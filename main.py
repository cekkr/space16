import math
import os
import time
import random
from dataclasses import dataclass, field
import curses
from typing import List, Tuple, Callable, Dict
import pyfiglet

@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def rotate_y(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector3D(
            self.x * cos_a + self.z * sin_a,
            self.y,
            -self.x * sin_a + self.z * cos_a
        )

    def rotate_x(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector3D(
            self.x,
            self.y * cos_a - self.z * sin_a,
            self.y * sin_a + self.z * cos_a
        )

###
###
###

# Arte ASCII pre-generata per diverse dimensioni di pianeti
PLANET_SIZES = {
    'tiny': [
        '  ○  ',
        ' ⚬⚬⚬ ',
        '  ○  '
    ],
    'small': [
        '   ◯◯   ',
        '  ◯◯◯◯  ',
        ' ◯◯◯◯◯ ',
        '  ◯◯◯◯  ',
        '   ◯◯   '
    ],
    'medium': [
        '    ████    ',
        '  ████████  ',
        ' ██████████ ',
        '████████████',
        ' ██████████ ',
        '  ████████  ',
        '    ████    '
    ]
}


@dataclass
class CelestialObject:
    position: Vector3D
    type: str = 'star'  # 'star' o 'planet'
    size: float = 1.0  # Dimensione relativa
    features: dict = field(default_factory=dict)  # Caratteristiche procedurali

    def __post_init__(self):
        if not self.features:
            self.generate_features()

    def generate_features(self):
        """Genera caratteristiche casuali per il corpo celeste."""
        self.features = {
            'brightness': random.uniform(0.1, 1.0),
            'color_base': random.choice(['⋅', '·', ':', '°', 'o', 'O']),
            'surface_type': random.choice(['rocky', 'gaseous', 'ice']),
            'bands': random.randint(2, 5),
            'pattern_density': random.uniform(0.3, 0.7),
            'rotation_angle': random.uniform(0, 2 * math.pi)
        }

    def get_char_at_distance(self, distance: float) -> Tuple[str, List[str]]:
        """Restituisce la rappresentazione appropriata basata sulla distanza."""
        if distance > 50:
            # Stella lontana
            brightness = self.features['brightness']
            if brightness > 0.8:
                return '*', []
            elif brightness > 0.5:
                return '+', []
            elif brightness > 0.2:
                return '.', []
            return '·', []
        elif distance > 30:
            # Pianeta piccolo
            return '@', PLANET_SIZES['tiny']
        elif distance > 15:
            # Pianeta medio
            return 'O', PLANET_SIZES['small']
        else:
            # Pianeta dettagliato
            return '#', self.generate_detailed_view()

    def generate_detailed_view(self) -> List[str]:
        """Genera una vista dettagliata del pianeta."""
        size = 20
        view = []
        surface = self.features['surface_type']
        density = self.features['pattern_density']
        bands = self.features['bands']

        for y in range(size):
            row = ""
            angle = 2 * math.pi * y / size

            # Calcola l'intensità della riga basata sulla posizione y
            y_factor = 1.0 - abs(y - size / 2) / (size / 2)
            width = int(size * y_factor)
            padding = (size - width) // 2

            for x in range(size):
                if x < padding or x >= padding + width:
                    row += " "
                    continue

                # Genera pattern procedurali basati sul tipo di superficie
                if surface == 'rocky':
                    val = noise_2d(x / 5 + self.features['rotation_angle'], y / 5) * density
                    char = '█' if val > 0.6 else '▓' if val > 0.4 else '▒' if val > 0.2 else '░'
                elif surface == 'gaseous':
                    band_val = math.sin(y * bands * math.pi / size)
                    char = '═' if band_val > 0.3 else '─' if band_val > 0 else ' '
                else:  # ice
                    val = noise_2d(x / 3, y / 3) * density
                    char = '❄' if val > 0.7 else '•' if val > 0.4 else '·'

                row += char

            view.append(row)

        return view

def noise_2d(x: float, y: float) -> float:
    """Semplice funzione di rumore 2D per generare texture procedurali."""
    return (math.sin(x * 12.9898 + y * 78.233) * 43758.5453123) % 1.0

###
###
###

@dataclass
class GameObject:
    position: Vector3D
    vertices: List[Vector3D] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    rotation_y: float = 0.0
    rotation_x: float = 0.0
    velocity: Vector3D = field(default_factory=lambda: Vector3D(0, 0, 0))
    target_rotation_x: float = 0.0  # Rotazione target per interpolazione smooth
    target_rotation_y: float = 0.0

    def update(self, delta_time: float):
        self.position = self.position + (self.velocity * delta_time)

        # Interpola smoothly verso la rotazione target
        rotation_speed = 5.0 * delta_time
        self.rotation_x += (self.target_rotation_x - self.rotation_x) * rotation_speed
        self.rotation_y += (self.target_rotation_y - self.rotation_y) * rotation_speed

        # Limita la rotazione verticale
        self.rotation_x = max(min(self.rotation_x, math.pi / 2), -math.pi / 2)
        self.target_rotation_x = max(min(self.target_rotation_x, math.pi / 2), -math.pi / 2)

        # Applica frizione
        friction = 0.99
        self.velocity = self.velocity * pow(friction, delta_time * 60)

    def emergency_brake(self, delta_time: float):
        """Applica una forte decelerazione per fermare la nave."""
        brake_force = 5.0 * delta_time
        self.velocity = self.velocity * (1.0 - brake_force)

    def get_forward_vector(self) -> Vector3D:
        # Calcola il vettore direzione considerando entrambe le rotazioni
        cos_pitch = math.cos(self.rotation_x)
        return Vector3D(
            math.sin(self.rotation_y) * cos_pitch,
            -math.sin(self.rotation_x),
            math.cos(self.rotation_y) * cos_pitch
        )

    def apply_thrust(self, amount: float):
        forward = self.get_forward_vector()
        self.velocity = self.velocity + (forward * amount)

    def accelerate(self, amount: float):
        # Accelera nella direzione corrente
        forward = self.get_forward_vector()
        self.velocity = self.velocity + (forward * amount)

    def get_transformed_vertices(self) -> List[Vector3D]:
        # Applica rotazione e traslazione ai vertici
        transformed = []
        for vertex in self.vertices:
            rotated = vertex.rotate_y(self.rotation_y)
            transformed.append(rotated + self.position)
        return transformed

###
###
###

class HUDElement:
    def __init__(self, position: Tuple[int, int], render_func: Callable):
        self.position = position
        self.render_func = render_func
        self.visible = True


class HUDSystem:
    def __init__(self):
        self.elements: Dict[str, HUDElement] = {}

    def add_element(self, name: str, element: HUDElement):
        self.elements[name] = element

    def remove_element(self, name: str):
        if name in self.elements:
            del self.elements[name]

    def set_visibility(self, name: str, visible: bool):
        if name in self.elements:
            self.elements[name].visible = visible

    def render(self, stdscr, game_state: dict):
        for name, element in self.elements.items():
            if element.visible:
                try:
                    text = element.render_func(game_state)
                    y, x = element.position
                    if isinstance(text, list):
                        for i, line in enumerate(text):
                            stdscr.addstr(y + i, x, line)
                    else:
                        stdscr.addstr(y, x, text)
                except curses.error:
                    pass

###
###
###

@dataclass
class Star:
    position: Vector3D
    brightness: float  # Da 0 a 1

    def get_char(self) -> str:
        if self.brightness > 0.8:
            return '*'
        elif self.brightness > 0.5:
            return '+'
        elif self.brightness > 0.2:
            return '.'
        else:
            return '·'


class Camera:
    def __init__(self, target: GameObject = None):
        self.position = Vector3D(0, 0, -10)
        self.target = target
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.distance = 15.0
        self.height_offset = 5.0
        self.lag_factor = 0.1  # Fattore di ritardo per il movimento smooth

    def update(self, delta_time: float):
        if self.target:
            # Interpola rotazione
            self.rotation_x += (self.target.rotation_x - self.rotation_x) * self.lag_factor
            self.rotation_y += (self.target.rotation_y - self.rotation_y) * self.lag_factor

            # Calcola posizione target della camera
            forward = self.target.get_forward_vector()
            target_pos = self.target.position + (forward * -self.distance)
            target_pos.y += self.height_offset

            # Interpola posizione
            self.position = Vector3D(
                self.position.x + (target_pos.x - self.position.x) * self.lag_factor,
                self.position.y + (target_pos.y - self.position.y) * self.lag_factor,
                self.position.z + (target_pos.z - self.position.z) * self.lag_factor
            )

class SpaceGameEngine:
    def __init__(self):
        self.screen_height, self.screen_width = curses.LINES - 1, curses.COLS - 1
        self.objects: List[GameObject] = []
        self.celestial_objects: List[CelestialObject] = []
        self.player_ship: GameObject = None
        self.camera: Camera = None
        self.last_frame_time = time.time()
        self.fov = 60.0
        self.near_plane = 0.1
        self.far_plane = 1000.0

        self.hud = HUDSystem()
        self.setup_hud()
        self.generate_celestial_objects(1000)

    def setup_hud(self):
        # Velocità
        self.hud.add_element("velocity", HUDElement(
            (0, 0),
            lambda state: f"Velocity: {state['velocity']:.1f} u/s"
        ))

        # Rotazione
        self.hud.add_element("rotation", HUDElement(
            (1, 0),
            lambda state: [
                f"Yaw: {math.degrees(state['rotation_y']):.1f}°",
                f"Pitch: {math.degrees(state['rotation_x']):.1f}°"
            ]
        ))

        # FOV
        self.hud.add_element("fov", HUDElement(
            (3, 0),
            lambda state: f"FOV: {state['fov']:.1f}°"
        ))

        # Controlli
        self.hud.add_element("controls", HUDElement(
            (5, 0),
            lambda state: [
                "Controls:",
                "W/S     - Thrust forward/back",
                "A/D     - Turn left/right",
                "R/F     - Pitch up/down",
                "O/P     - Adjust FOV",
                "Q       - Quit"
            ]
        ))

        self.hud.add_element("brake", HUDElement(
            (11, 0),
            lambda state: "EMERGENCY BRAKE ACTIVE" if state.get('braking', False) else ""
        ))

    def generate_celestial_objects(self, count: int):
        """Genera un mix di stelle e pianeti."""
        self.celestial_objects = []
        for _ in range(count):
            pos = Vector3D(
                random.uniform(-200, 200),
                random.uniform(-200, 200),
                random.uniform(0, 400)
            )

            # 20% di probabilità di essere un pianeta
            obj_type = 'planet' if random.random() < 0.2 else 'star'
            size = random.uniform(1.0, 5.0) if obj_type == 'planet' else 1.0

            self.celestial_objects.append(CelestialObject(pos, obj_type, size))

    def project_point(self, point: Vector3D) -> Tuple[int, int]:
        if not self.camera:
            return (-1, -1)

        # Calcola la posizione relativa alla camera
        relative_point = point - self.camera.position

        # Applica la rotazione della camera
        rotated = relative_point.rotate_y(-self.camera.rotation_y).rotate_x(-self.camera.rotation_x)

        if rotated.z <= self.near_plane:
            return (-1, -1)

        # Proiezione prospettica con FOV corretto
        fov_rad = math.radians(self.fov)
        aspect_ratio = self.screen_width / self.screen_height

        scale_y = 1.0 / math.tan(fov_rad / 2)
        scale_x = scale_y / aspect_ratio

        screen_x = rotated.x / rotated.z * scale_x
        screen_y = rotated.y / rotated.z * scale_y

        x = int(self.screen_width / 2 * (1 + screen_x))
        y = int(self.screen_height / 2 * (1 - screen_y))

        return (x, y)

    def render_celestial_object(self, stdscr, obj: CelestialObject, projected_pos: Tuple[int, int]):
        if not self.camera:
            return

        # Calcola la distanza dall'oggetto
        distance = math.sqrt(
            (obj.position.x - self.camera.position.x) ** 2 +
            (obj.position.y - self.camera.position.y) ** 2 +
            (obj.position.z - self.camera.position.z) ** 2
        )

        # Calcola la dimensione apparente dell'oggetto
        apparent_size = obj.size / distance

        # Non renderizzare oggetti troppo piccoli
        if apparent_size < 0.25:
            return

        char, detailed_view = obj.get_char_at_distance(distance)

        if not detailed_view:
            if (0 <= projected_pos[0] < self.screen_width and
                    0 <= projected_pos[1] < self.screen_height):
                try:
                    stdscr.addch(projected_pos[1], projected_pos[0], char)
                except curses.error:
                    pass
        else:
            # Calcola la dimensione del rendering basata sulla distanza
            size_factor = min(5.0, max(1.0, 20.0 / distance))
            scaled_view = self.scale_detailed_view(detailed_view, size_factor)

            half_height = len(scaled_view) // 2
            half_width = len(scaled_view[0]) // 2

            for y, row in enumerate(scaled_view):
                screen_y = projected_pos[1] - half_height + y
                if not (0 <= screen_y < self.screen_height):
                    continue

                for x, char in enumerate(row):
                    screen_x = projected_pos[0] - half_width + x
                    if not (0 <= screen_x < self.screen_width):
                        continue

                    try:
                        if char != ' ':
                            stdscr.addch(screen_y, screen_x, char)
                    except curses.error:
                        pass

    def scale_detailed_view(self, view: List[str], factor: float) -> List[str]:
        """Scala la vista dettagliata del pianeta."""
        if factor <= 1.0:
            return view

        new_view = []
        for row in view:
            new_row = ""
            for char in row:
                new_row += char * int(factor)
            new_view.extend([new_row] * int(factor))

    def update(self, stdscr):
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Aggiorna dimensioni schermo
        new_height, new_width = curses.LINES - 1, curses.COLS - 1
        if new_height != self.screen_height or new_width != self.screen_width:
            self.screen_height, self.screen_width = new_height, new_width
            stdscr.clear()

        # Gestione input
        key = stdscr.getch()
        braking = False

        if key != -1 and self.player_ship:
            thrust = 20.0 * delta_time
            rotation_speed = 2.0

            if key == ord('w'):
                self.player_ship.apply_thrust(thrust)
            elif key == ord('s'):
                self.player_ship.apply_thrust(-thrust)
            elif key == ord('a'):
                self.player_ship.target_rotation_y += rotation_speed * delta_time
            elif key == ord('d'):
                self.player_ship.target_rotation_y -= rotation_speed * delta_time
            elif key == ord('r'):
                self.player_ship.target_rotation_x -= rotation_speed * delta_time
            elif key == ord('f'):
                self.player_ship.target_rotation_x += rotation_speed * delta_time
            elif key == ord(' '):  # Freno di emergenza
                self.player_ship.emergency_brake(delta_time)
                braking = True
            elif key == ord('o'):
                self.fov = max(30.0, self.fov - 5.0)
            elif key == ord('p'):
                self.fov = min(120.0, self.fov + 5.0)
            elif key == ord('q'):
                return False

        # Aggiorna camera e oggetti
        if self.camera:
            self.camera.update(delta_time)

        for obj in self.objects:
            obj.update(delta_time)

        # Gestione oggetti celesti infiniti
        if self.player_ship:
            # Rimuovi oggetti troppo lontani
            self.celestial_objects = [obj for obj in self.celestial_objects
                                      if (obj.position - self.player_ship.position).z < 400]

            # Aggiungi nuovi oggetti
            while len(self.celestial_objects) < 100:
                forward = self.player_ship.get_forward_vector()
                random_offset = Vector3D(
                    random.uniform(-200, 200),
                    random.uniform(-200, 200),
                    0
                )
                pos = self.player_ship.position + (forward * 400) + random_offset

                obj_type = 'planet' if random.random() < 0.2 else 'star'
                size = random.uniform(1.0, 5.0) if obj_type == 'planet' else 1.0

                self.celestial_objects.append(CelestialObject(pos, obj_type, size))

            velocity_magnitude = math.sqrt(
                self.player_ship.velocity.x ** 2 +
                self.player_ship.velocity.y ** 2 +
                self.player_ship.velocity.z ** 2
            )

            game_state = {
                'velocity': velocity_magnitude,
                'rotation_y': self.player_ship.rotation_y,
                'rotation_x': self.player_ship.rotation_x,
                'fov': self.fov,
                'braking': braking
            }

            self.hud.render(stdscr, game_state)

        return True

    def render_frame(self, stdscr) -> None:
        stdscr.clear()

        # Rendering degli oggetti celesti
        for obj in self.celestial_objects:
            projected = self.project_point(obj.position)
            if projected[0] >= 0:
                self.render_celestial_object(stdscr, obj, projected)

        # Rendering degli oggetti
        for obj in self.objects:
            transformed_vertices = obj.get_transformed_vertices()
            projected_vertices = [self.project_point(vertex) for vertex in transformed_vertices]

            for edge in obj.edges:
                start = projected_vertices[edge[0]]
                end = projected_vertices[edge[1]]

                if (start[0] >= 0 and start[1] >= 0 and
                        end[0] >= 0 and end[1] >= 0 and
                        start[0] < self.screen_width and start[1] < self.screen_height and
                        end[0] < self.screen_width and end[1] < self.screen_height):
                    self.draw_line(stdscr, start[0], start[1], end[0], end[1])

        # Aggiorna e renderizza HUD
        if self.player_ship:
            velocity_magnitude = math.sqrt(
                self.player_ship.velocity.x ** 2 +
                self.player_ship.velocity.y ** 2 +
                self.player_ship.velocity.z ** 2
            )

            game_state = {
                'velocity': velocity_magnitude,
                'rotation_y': self.player_ship.rotation_y,
                'rotation_x': self.player_ship.rotation_x,
                'fov': self.fov
            }

            self.hud.render(stdscr, game_state)

        stdscr.refresh()

    def create_player_ship(self) -> GameObject:
        # Definizione semplificata di una X-Wing
        vertices = [
            Vector3D(-2, -1, -2),  # Base posteriore
            Vector3D(2, -1, -2),
            Vector3D(0, 1, -2),
            Vector3D(0, -1, 2),  # Muso
            Vector3D(-3, 0, -1),  # Ali
            Vector3D(3, 0, -1),
            Vector3D(-3, 0, 1),
            Vector3D(3, 0, 1),
        ]

        edges = [
            (0, 1), (0, 2), (1, 2),  # Base posteriore
            (0, 3), (1, 3), (2, 3),  # Connessioni al muso
            (4, 6), (5, 7),  # Ali
            (4, 0), (5, 1),  # Connessioni ali-base
            (6, 3), (7, 3)  # Connessioni ali-muso
        ]

        ship = GameObject(
            position=Vector3D(0, 0, 0),
            vertices=vertices,
            edges=edges,
            rotation_y=0.0,
            velocity=Vector3D(0, 0, 0)
        )
        self.player_ship = ship
        return ship

    def draw_line(self, stdscr, x1: int, y1: int, x2: int, y2: int) -> None:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
                    stdscr.addch(y, x, '#')
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
                    stdscr.addch(y, x, '#')
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)

    engine = SpaceGameEngine()

    # Crea la nave del giocatore
    player_ship = engine.create_player_ship()
    engine.objects.append(player_ship)
    engine.player_ship = player_ship

    # Inizializza la camera
    engine.camera = Camera(player_ship)

    running = True
    while running:
        running = engine.update(stdscr)
        engine.render_frame(stdscr)
        time.sleep(0.033)


if __name__ == "__main__":
    curses.wrapper(main)