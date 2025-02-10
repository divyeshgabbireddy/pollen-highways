import json
import math
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any

from flask import Flask, render_template, jsonify, request

# Set up logging for clarity.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

###############################################################################
# BACKEND OVERVIEW:
#
# This application integrates two live data feeds:
#
# 1. WindBorne’s balloon API: Returns 24 hourly snapshots of balloon positions.
#    Each balloon’s 24‐hour trajectory (its path in the atmosphere) is used to 
#    detect rapid descents below 30 km altitude.
#
# 2. Open‑Meteo API: Provides ground weather data (surface temperature and 
#    cloud cover) at any location.
#
# Our pipeline:
# - Retrieve and align the balloon trajectories from the past 24 hours.
# - Filter to trajectories whose final altitude is under 30 km and that are 
#   descending rapidly.
# - Cluster geographically nearby final positions into “frost events.”
# - For each event, fetch local ground conditions and compute a frost risk.
#
# The frost risk is calculated as:
#
#   risk = (|avg_descent_rate| / 1.0) *
#          (max(0, (2 - ground_temperature)) / 2.0) *
#          (1 - (cloud_cover / 100)) * 100
#
# where:
# - avg_descent_rate is the average descent (km/h) of the balloons in the cluster.
# - ground_temperature is the current surface temperature (°C).
# - cloud_cover is the percent cloud cover.
#
# Note: If the computed risk exceeds 100, it is capped at 100. Thus, for extremely
# low temperatures (e.g. -46°C), even a modest descent rate can yield a 100% risk.
#
# This combined approach gives us early, data‐driven insight into potential frost 
# events—a critical improvement over traditional methods that rely solely on 
# surface or satellite data. Such early warnings can help agriculture avoid millions
# in crop losses.
###############################################################################

@dataclass
class WindborneReading:
    latitude: float
    longitude: float
    altitude: float  # in km
    timestamp: str
    descent_rate: float = 0.0  # computed later

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great‐circle distance between two lat/lon points (in km)."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class DataFetcher:
    def __init__(self):
        self.windborne_base_url = "https://a.windbornesystems.com/treasure"
        self.owm_url = "https://api.open-meteo.com/v1/forecast"

    async def fetch_url(self, session: aiohttp.ClientSession, url: str, params: Dict[str, Any] = None) -> Any:
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return None
                text = await response.text()
                if not text.strip():
                    logger.warning(f"Empty response from {url}")
                    return None
                try:
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(text)
                    return data
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing issue from {url}: {str(e)}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def fetch_aligned_balloon_data(self) -> Dict[int, List[WindborneReading]]:
        """
        Retrieve 24 snapshots and align them into balloon trajectories (each trajectory
        is the ordered list of positions for a single balloon over 24 hours).
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for hour in range(24):
                url = f"{self.windborne_base_url}/{hour:02d}.json"
                tasks.append(self.fetch_url(session, url))
            results = await asyncio.gather(*tasks)

        now = datetime.now()
        valid_snapshots = []
        for hour, res in enumerate(results):
            if res and isinstance(res, list):
                snapshot = []
                timestamp = (now - timedelta(hours=hour)).isoformat()
                for reading in res:
                    try:
                        lat, lon, alt = map(float, reading[:3])
                        snapshot.append(WindborneReading(
                            latitude=lat,
                            longitude=lon,
                            altitude=alt,
                            timestamp=timestamp
                        ))
                    except Exception as e:
                        logger.debug(f"Skipping malformed reading: {e}")
                        continue
                valid_snapshots.append((hour, snapshot))

        if not valid_snapshots:
            return {}

        # Order snapshots from oldest (23 hrs ago) to newest (0 hrs ago)
        valid_snapshots.sort(key=lambda x: x[0], reverse=True)
        min_count = min(len(s[1]) for s in valid_snapshots)
        trajectories = {}
        for j in range(min_count):
            traj = []
            for _, snapshot in valid_snapshots:
                traj.append(snapshot[j])
            trajectories[j] = traj

        return trajectories

    async def fetch_ground_conditions(self, lat: float, lon: float) -> Dict[str, float]:
        """Fetch current ground temperature and cloud cover from Open‑Meteo."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "hourly": "cloudcover",
            "timezone": "auto"
        }
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_url(session, self.owm_url, params=params)
            ground_temp = 10.0
            cloud_cover = 50.0
            if data:
                if "current_weather" in data:
                    ground_temp = data["current_weather"].get("temperature", 10.0)
                    current_time = data["current_weather"].get("time")
                else:
                    current_time = None

                if "hourly" in data and "cloudcover" in data["hourly"] and "time" in data["hourly"]:
                    times = data["hourly"]["time"]
                    ccover = data["hourly"]["cloudcover"]
                    if current_time and current_time in times:
                        idx = times.index(current_time)
                        cloud_cover = ccover[idx]
                    else:
                        cloud_cover = ccover[0]
            return {
                "ground_temperature": ground_temp,
                "cloud_cover": cloud_cover
            }

class FrostRiskAnalyzer:
    def __init__(self,
                 altitude_range: tuple = (0, 30),
                 cluster_distance_km: float = 100,
                 min_cluster_size: int = 2,
                 descent_threshold: float = -0.1):
        """
        Parameters:
          - altitude_range: Only consider balloons ending below altitude_range[1] (km)
          - cluster_distance_km: Maximum distance (km) to group balloon endpoints
          - min_cluster_size: Minimum number of balloons to form a frost alert
          - descent_threshold: Average descent (km/h) must be below this threshold
        """
        self.altitude_range = altitude_range
        self.cluster_distance_km = cluster_distance_km
        self.min_cluster_size = min_cluster_size
        self.descent_threshold = descent_threshold
        self.data_fetcher = DataFetcher()

    def compute_trajectory_metrics(self, traj: List[WindborneReading]) -> Dict[str, float]:
        """
        Calculate:
          - horizontal_speed: distance between first and last positions (km/h)
          - descent_rate: average change in altitude (km/h)
        """
        if len(traj) < 2:
            return {"horizontal_speed": 0.0, "descent_rate": 0.0}
        first = traj[0]
        last = traj[-1]
        distance = haversine(first.latitude, first.longitude, last.latitude, last.longitude)
        time_diff = len(traj) - 1  # approx. hours between snapshots
        horizontal_speed = distance / time_diff
        descent_rate = (last.altitude - first.altitude) / time_diff
        return {
            "horizontal_speed": horizontal_speed,
            "descent_rate": descent_rate
        }

    async def analyze_frost_risk(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analysis pipeline:
         1. Fetch and align balloon trajectories.
         2. Filter for trajectories ending below 30 km with descent rates below the threshold.
         3. Cluster nearby endpoints into frost events.
         4. For each event, fetch ground conditions and compute risk.
         5. Attach the full trajectory for each balloon.
        """
        trajectories = await self.data_fetcher.fetch_aligned_balloon_data()
        now = datetime.now()
        filtered = {}

        for idx, traj in trajectories.items():
            reading_time = datetime.fromisoformat(traj[-1].timestamp)
            if (now - reading_time) <= timedelta(hours=hours):
                metrics = self.compute_trajectory_metrics(traj)
                final_alt = traj[-1].altitude
                if final_alt < self.altitude_range[1] and metrics["descent_rate"] < self.descent_threshold:
                    filtered[idx] = {"trajectory": traj, "metrics": metrics}

        # Build representatives that carry full trajectory data.
        representatives = []
        for idx, item in filtered.items():
            traj = item["trajectory"]
            rep = traj[-1]
            rep.descent_rate = item["metrics"]["descent_rate"]
            # Attach the full trajectory as a list of dicts.
            rep.full_trajectory = [{"lat": r.latitude, "lon": r.longitude, "alt": r.altitude} for r in traj]
            representatives.append(rep)

        # Cluster final readings.
        clusters = self._cluster_readings(representatives)

        frost_events = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue
            centroid_lat = sum(r.latitude for r in cluster) / len(cluster)
            centroid_lon = sum(r.longitude for r in cluster) / len(cluster)
            avg_descent_rate = sum(r.descent_rate for r in cluster) / len(cluster)

            ground_data = await self.data_fetcher.fetch_ground_conditions(centroid_lat, centroid_lon)
            ground_temp = ground_data.get("ground_temperature", 10.0)
            cloud_cover = ground_data.get("cloud_cover", 50.0)

            # Frost risk formula:
            # risk = (|avg_descent_rate| / 1.0) *
            #        (max(0, (2 - ground_temp)) / 2.0) *
            #        (1 - (cloud_cover / 100)) * 100
            # Note: The result is capped at 100.
            risk = (abs(avg_descent_rate) / 1.0) * (max(0, (2 - ground_temp)) / 2.0) * (1 - (cloud_cover / 100)) * 100
            risk = min(100, risk)

            member_data = []
            for r in cluster:
                member_data.append({
                    "latitude": round(r.latitude, 2),
                    "longitude": round(r.longitude, 2),
                    "altitude": round(r.altitude, 2),
                    "descent_rate": round(r.descent_rate, 2),
                    "trajectory": r.full_trajectory  # full 24-hour path
                })

            frost_events.append({
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "avg_descent_rate": round(avg_descent_rate, 2),
                "balloon_count": len(cluster),
                "ground_temperature": round(ground_temp, 1),
                "cloud_cover": round(cloud_cover, 1),
                "frost_risk": round(risk, 1),
                "members": member_data
            })

        return {
            "frost_events": frost_events,
            "total_trajectories": len(filtered),
            "timestamp": now.isoformat(),
            "hours": hours
        }

    def _cluster_readings(self, readings: List[WindborneReading]) -> List[List[WindborneReading]]:
        """Naively group readings if they are within 'cluster_distance_km' of one another."""
        clusters = []
        used = [False] * len(readings)
        for i, r in enumerate(readings):
            if used[i]:
                continue
            cluster = [r]
            used[i] = True
            for j, r2 in enumerate(readings):
                if not used[j]:
                    if haversine(r.latitude, r.longitude, r2.latitude, r2.longitude) <= self.cluster_distance_km:
                        cluster.append(r2)
                        used[j] = True
            clusters.append(cluster)
        return clusters

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    hours = request.args.get('hours', default=24, type=int)
    analyzer = FrostRiskAnalyzer(
        altitude_range=(0, 30),
        cluster_distance_km=100,
        min_cluster_size=2,
        descent_threshold=-0.1
    )
    data = asyncio.run(analyzer.analyze_frost_risk(hours=hours))
    return jsonify(data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
