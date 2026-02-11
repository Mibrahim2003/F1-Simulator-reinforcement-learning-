"""
Race Manager for F1 Racing Simulator.

Handles multi-agent racing logic: position tracking, lap timing,
race state management, and leaderboard functionality.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class RaceState(Enum):
    """Current state of the race."""
    WAITING = "waiting"       # Pre-race, cars on grid
    COUNTDOWN = "countdown"   # Race countdown
    RACING = "racing"         # Race in progress
    FINISHED = "finished"     # Race complete


@dataclass
class LapTime:
    """Represents a single lap time."""
    lap_number: int
    time_seconds: float
    
    def __str__(self):
        mins = int(self.time_seconds // 60)
        secs = self.time_seconds % 60
        return f"{mins}:{secs:05.2f}"


@dataclass
class RacerStats:
    """Statistics for a single racer."""
    racer_id: int
    name: str
    color: Tuple[int, int, int]
    
    # Race progress
    current_lap: int = 0
    lap_progress: float = 0.0  # 0-1 within current lap
    total_progress: float = 0.0  # Total laps + progress
    
    # Timing
    lap_times: List[LapTime] = field(default_factory=list)
    best_lap_time: Optional[float] = None
    last_lap_time: Optional[float] = None
    lap_start_time: float = 0.0
    
    # Status
    position: int = 1
    finished: bool = False
    finish_time: Optional[float] = None
    dnf: bool = False  # Did Not Finish
    crashes: int = 0
    
    def get_best_lap_str(self) -> str:
        if self.best_lap_time is None:
            return "--:--.--"
        mins = int(self.best_lap_time // 60)
        secs = self.best_lap_time % 60
        return f"{mins}:{secs:05.2f}"
    
    def get_last_lap_str(self) -> str:
        if self.last_lap_time is None:
            return "--:--.--"
        mins = int(self.last_lap_time // 60)
        secs = self.last_lap_time % 60
        return f"{mins}:{secs:05.2f}"


class RaceManager:
    """
    Manages a multi-agent race.
    
    Handles:
    - Race state (countdown, racing, finished)
    - Position tracking based on progress
    - Lap time recording
    - Leaderboard generation
    """
    
    # Color palette for racers
    RACER_COLORS = [
        (255, 0, 0),      # Red
        (0, 100, 255),    # Blue  
        (0, 200, 100),    # Green
        (255, 200, 0),    # Yellow
        (200, 0, 200),    # Purple
        (255, 100, 0),    # Orange
        (0, 200, 200),    # Cyan
        (200, 200, 200),  # White
    ]
    
    RACER_NAMES = [
        "VERSTAPPEN", "HAMILTON", "LECLERC", "NORRIS",
        "SAINZ", "RUSSELL", "PEREZ", "ALONSO"
    ]
    
    def __init__(
        self,
        num_racers: int = 4,
        total_laps: int = 3,
        countdown_seconds: float = 6.0
    ):
        self.num_racers = num_racers
        self.total_laps = total_laps
        self.countdown_seconds = countdown_seconds
        
        # Race state
        self.state = RaceState.WAITING
        self.race_start_time: float = 0.0
        self.countdown_start_time: float = 0.0
        
        # F1 lights state: 0 = no lights, 1-5 = number of red lights on, 6 = lights out (GO!)
        self.lights_state: int = 0
        self.lights_out_time: float = 0.0  # When lights went out
        
        # Racer tracking
        self.racers: Dict[int, RacerStats] = {}
        self._init_racers()
    
    def _init_racers(self):
        """Initialize racer stats."""
        for i in range(self.num_racers):
            self.racers[i] = RacerStats(
                racer_id=i,
                name=self.RACER_NAMES[i % len(self.RACER_NAMES)],
                color=self.RACER_COLORS[i % len(self.RACER_COLORS)],
                position=i + 1  # Starting grid position
            )
    
    def reset(self):
        """Reset for a new race."""
        self.state = RaceState.WAITING
        self.race_start_time = 0.0
        self.countdown_start_time = 0.0
        self.lights_state = 0
        self.lights_out_time = 0.0
        self._init_racers()
    
    def start_countdown(self):
        """Begin the F1 lights sequence."""
        self.state = RaceState.COUNTDOWN
        self.countdown_start_time = time.time()
        self.lights_state = 0
    
    def get_countdown_remaining(self) -> float:
        """Get remaining countdown time."""
        if self.state != RaceState.COUNTDOWN:
            return 0.0
        elapsed = time.time() - self.countdown_start_time
        return max(0.0, self.countdown_seconds - elapsed)
    
    def get_lights_state(self) -> int:
        """
        Get current F1 lights state.
        
        Returns:
            0 = waiting, 1-5 = red lights on (one per second), 6 = LIGHTS OUT (GO!)
        """
        if self.state != RaceState.COUNTDOWN and self.state != RaceState.RACING:
            return 0
        
        elapsed = time.time() - self.countdown_start_time
        
        # Each light turns on every 1 second
        if elapsed < 1.0:
            return 1
        elif elapsed < 2.0:
            return 2
        elif elapsed < 3.0:
            return 3
        elif elapsed < 4.0:
            return 4
        elif elapsed < 5.0:
            return 5
        else:
            return 6  # LIGHTS OUT!
    
    def update(self, racer_progress: Dict[int, Tuple[float, bool]]):
        """
        Update race state.
        
        Args:
            racer_progress: Dict mapping racer_id to (total_progress, crashed)
        """
        current_time = time.time()
        
        # Handle countdown to race start
        if self.state == RaceState.COUNTDOWN:
            if self.get_countdown_remaining() <= 0:
                self.state = RaceState.RACING
                self.race_start_time = current_time
                # Start lap timers
                for racer in self.racers.values():
                    racer.lap_start_time = current_time
        
        if self.state != RaceState.RACING:
            return
        
        # Update each racer
        for racer_id, (progress, crashed) in racer_progress.items():
            if racer_id not in self.racers:
                continue
            
            racer = self.racers[racer_id]
            
            if racer.finished or racer.dnf:
                continue
            
            # Track crashes
            if crashed:
                racer.crashes += 1
            
            # Calculate lap progress
            old_lap = racer.current_lap
            new_lap = int(progress)
            lap_progress = progress - new_lap
            
            # Check for lap completion
            if new_lap > old_lap and old_lap >= 0:
                # Completed a lap
                lap_time = current_time - racer.lap_start_time
                racer.lap_times.append(LapTime(old_lap + 1, lap_time))
                racer.last_lap_time = lap_time
                
                if racer.best_lap_time is None or lap_time < racer.best_lap_time:
                    racer.best_lap_time = lap_time
                
                racer.lap_start_time = current_time
                
                # Check for race finish
                if new_lap >= self.total_laps:
                    racer.finished = True
                    racer.finish_time = current_time - self.race_start_time
            
            racer.current_lap = min(new_lap, self.total_laps)
            racer.lap_progress = lap_progress
            racer.total_progress = progress
        
        # Update positions based on total progress
        self._update_positions()
        
        # Check if all racers finished
        if all(r.finished or r.dnf for r in self.racers.values()):
            self.state = RaceState.FINISHED
    
    def _update_positions(self):
        """Update race positions based on progress."""
        # Sort racers by: finished first (by finish time), then by total progress
        # DNF racers go last
        def sort_key(r):
            if r.dnf:
                return (0, 0, 0)  # DNF at bottom
            elif r.finished:
                # Finished: earlier finish time = better (lower = earlier)
                return (2, -r.finish_time if r.finish_time else 0, r.total_progress)
            else:
                # Racing: higher progress = better
                return (1, r.total_progress, 0)
        
        sorted_racers = sorted(
            self.racers.values(),
            key=sort_key,
            reverse=True
        )
        
        for pos, racer in enumerate(sorted_racers, 1):
            racer.position = pos
    
    def mark_dnf(self, racer_id: int):
        """Mark a racer as Did Not Finish."""
        if racer_id in self.racers:
            self.racers[racer_id].dnf = True
    
    def get_leaderboard(self) -> List[RacerStats]:
        """Get racers sorted by position."""
        return sorted(self.racers.values(), key=lambda r: r.position)
    
    def get_race_time(self) -> float:
        """Get elapsed race time."""
        if self.state == RaceState.RACING:
            return time.time() - self.race_start_time
        elif self.state == RaceState.FINISHED:
            # Return winner's time
            for racer in self.racers.values():
                if racer.position == 1 and racer.finish_time:
                    return racer.finish_time
        return 0.0
    
    def format_time(self, seconds: float) -> str:
        """Format time as M:SS.ss"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
    
    def get_racer(self, racer_id: int) -> Optional[RacerStats]:
        """Get a racer by ID."""
        return self.racers.get(racer_id)
