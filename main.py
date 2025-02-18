import math
import os
import time
import random
from dataclasses import dataclass, field
import curses
from typing import List, Tuple, Callable, Dict

import pyfiglet

# Importa Pygame se disponibile
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

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

def generate_asteroid_shape():
    """Genera forme casuali per gli asteroidi usando pyfiglet e modifiche casuali."""
    shapes = {
        'small': [],
        'medium': [],
        'large': []
    }

    # Genera una forma base casuale
    base_chars = ['@', '#', '$', '%', '&']
    base_char = random.choice(base_chars)

    # Usa pyfiglet per generare arte ASCII di base
    figlet = pyfiglet.Figlet(font=random.choice(['small', 'mini']))
    base_art = figlet.renderText(base_char).split('\n')

    # Modifica casualmente la forma
    def modify_shape(art, scale):
        modified = []
        for line in art:
            if random.random() < 0.3:  # 30% chance di modificare la linea
                line = line.replace(' ', random.choice([' ', '.']))
            modified.append(line)
        return modified

    # Genera versioni di diverse dimensioni
    shapes['small'] = modify_shape(base_art, 0.5)
    shapes['medium'] = modify_shape(base_art, 1.0)
    shapes['large'] = modify_shape(base_art, 1.5)

    return shapes

@dataclass
class CelestialObject:
    position: Vector3D
    type: str = 'star'  # 'star', 'planet' o 'asteroid'
    size: float = 1.0
    features: dict = field(default_factory=dict)

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
            'rotation_angle': random.uniform(0, 2 * math.pi),
            'color': random.choice([
                curses.COLOR_RED,
                curses.COLOR_YELLOW,
                curses.COLOR_BLUE,
                curses.COLOR_CYAN,
                curses.COLOR_MAGENTA
            ]) if self.type != 'asteroid' else curses.COLOR_WHITE,
            'asteroid_shape': generate_asteroid_shape() if self.type == 'asteroid' else None
        }

    def get_char_at_distance(self, distance: float) -> Tuple[str, List[str], int]:
        # Calcola dimensione apparente
        apparent_size = self.size / max(distance, 0.1)
        color = self.features['color']

        if apparent_size < 0.25:
            return '', [], color
        elif distance > 50:
            brightness = self.features['brightness']
            if brightness > 0.8:
                return '*', [], color
            elif brightness > 0.5:
                return '+', [], color
            elif brightness > 0.2:
                return '.', [], color
            return '·', [], color
        else:
            size = int(20 + (50 / max(distance, 1.0)))
            return '#', self.generate_detailed_view(size), color

    def generate_detailed_view(self, size: int) -> List[str]:
        if self.type == 'asteroid':
            return self.generate_asteroid_view(size)

        view = []
        surface = self.features['surface_type']
        density = self.features['pattern_density']
        bands = self.features['bands']

        for y in range(size):
            row = ""
            y_factor = 1.0 - abs(y - size / 2) / (size / 2)
            width = int(size * y_factor)
            padding = (size - width) // 2

            for x in range(size):
                if x < padding or x >= padding + width:
                    row += " "
                    continue

                if surface == 'rocky':
                    val = (noise_2d(x / 3 + self.features['rotation_angle'], y / 3) * 0.5 +
                           noise_2d(x / 7 + self.features['rotation_angle'], y / 7) * 0.3 +
                           noise_2d(x / 13, y / 13) * 0.2) * density
                    char = '█' if val > 0.7 else '▓' if val > 0.5 else '▒' if val > 0.3 else '░'
                elif surface == 'gaseous':
                    band_val = (math.sin(y * bands * math.pi / size) +
                                math.sin(y * (bands + 1) * math.pi / size) * 0.5) * density
                    char = '═' if band_val > 0.5 else '─' if band_val > 0 else ' '
                else:  # ice
                    val = (noise_2d(x / 4, y / 4) * 0.6 +
                           noise_2d(x / 8, y / 8) * 0.4) * density
                    char = '❄' if val > 0.8 else '•' if val > 0.5 else '·'

                row += char
            view.append(row)
        return view

    def generate_asteroid_view(self, size: int) -> List[str]:
        if size < 10:
            return self.features['asteroid_shape']['small']
        elif size < 20:
            return self.features['asteroid_shape']['medium']
        else:
            return self.features['asteroid_shape']['large']

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
    target_rotation_x: float = 0.0
    target_rotation_y: float = 0.0

    def update(self, delta_time: float):
        # Aggiorna posizione basata sulla velocità attuale
        self.position = self.position + (self.velocity * delta_time)

        # Interpola rotazione più lentamente senza limiti
        rotation_speed = 3.0 * delta_time
        self.rotation_x = self.target_rotation_x
        self.rotation_y = self.target_rotation_y

        # Normalizza rotazioni
        self.rotation_y = self.rotation_y % (2 * math.pi)
        self.rotation_x = max(min(self.rotation_x, math.pi / 2), -math.pi / 2)  # Limita la rotazione verticale

        # Attrito molto ridotto
        friction = 0.999
        self.velocity = self.velocity * pow(friction, delta_time * 60)

    def emergency_brake(self, delta_time: float):
        brake_force = 2.0 * delta_time
        self.velocity = self.velocity * (1.0 - brake_force)

    def get_forward_vector(self) -> Vector3D:
        # Calcola direzione considerando entrambe le rotazioni
        return Vector3D(
            math.sin(self.rotation_y) * math.cos(self.rotation_x),
            math.sin(self.rotation_x),
            math.cos(self.rotation_y) * math.cos(self.rotation_x)
        )

    def apply_thrust(self, amount: float):
        forward = self.get_forward_vector()
        # Aumenta significativamente la potenza della spinta
        self.velocity = self.velocity + (forward * (amount * 2.0))

    def accelerate(self, amount: float):
        # Accelera nella direzione corrente
        forward = self.get_forward_vector()
        self.velocity = self.velocity + (forward * amount)

    def get_transformed_vertices(self) -> List[Vector3D]:
        # Applica rotazioni e traslazioni ai vertici
        transformed = []
        for vertex in self.vertices:
            rotated_y = vertex.rotate_y(self.rotation_y)
            rotated_x = rotated_y.rotate_x(self.rotation_x)  # Applica la rotazione verticale
            transformed.append(rotated_x + self.position)
        return transformed

###
###
###

class HUDElement:
    def __init__(self, position: tuple[int, int], render_func: callable):
        self.position = position
        self.render_func = render_func
        self.visible = True
        self.font_size = 16
        self.color = (255, 255, 255)  # Default white color for Pygame


class HUDSystem:
    def __init__(self, is_pygame):
        self.elements: dict[str, HUDElement] = {}
        self.pygame_font = None
        self.is_pygame_initialized = False
        self.is_pygame = is_pygame

    def init_pygame(self, font_name=None):
        """Initialize Pygame font system"""
        import pygame
        pygame.font.init()
        self.pygame_font = pygame.font.SysFont(font_name if font_name else 'arial', 16)
        self.is_pygame_initialized = True

    def add_element(self, name: str, element: HUDElement):
        self.elements[name] = element

    def remove_element(self, name: str):
        if name in self.elements:
            del self.elements[name]

    def set_visibility(self, name: str, visible: bool):
        if name in self.elements:
            self.elements[name].visible = visible

    def render(self, surface, game_state: dict):
        """Render HUD elements on either curses or pygame surface"""
        import curses
        
        # Check if we're using pygame or curses
        is_pygame = self.is_pygame
        
        if is_pygame:
            if not self.is_pygame_initialized:
                self.init_pygame()
            self._render_pygame(surface, game_state)
        else:
            self._render_curses(surface, game_state)

    def _render_curses(self, stdscr, game_state: dict):
        """Render HUD elements using curses"""
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

    def _render_pygame(self, screen, game_state: dict):
        """Render HUD elements using Pygame"""
        for name, element in self.elements.items():
            if element.visible:
                text = element.render_func(game_state)
                if isinstance(text, list):
                    for i, line in enumerate(text):
                        text_surface = self.pygame_font.render(
                            line, 
                            True, 
                            element.color
                        )
                        screen.blit(
                            text_surface, 
                            (element.position[1], element.position[0] + (i * element.font_size))
                        )
                else:
                    text_surface = self.pygame_font.render(
                        text, 
                        True, 
                        element.color
                    )
                    screen.blit(
                        text_surface, 
                        (element.position[1], element.position[0])
                    )

    def set_element_color(self, name: str, color: tuple[int, int, int]):
        """Set color for a specific HUD element (Pygame only)"""
        if name in self.elements:
            self.elements[name].color = color

    def set_element_font_size(self, name: str, size: int):
        """Set font size for a specific HUD element (Pygame only)"""
        if name in self.elements:
            self.elements[name].font_size = size
            if self.is_pygame_initialized:
                import pygame
                self.pygame_font = pygame.font.SysFont(None, size)

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
        self.distance = 30.0  # Aumentata la distanza base dalla nave
        self.height_offset = 8.0  # Aumentato l'offset verticale
        self.lag_factor = 0.1  # Aumentato leggermente per un movimento più fluido
        self.velocity = Vector3D(0, 0, 0)  # Aggiunta velocità della camera

    def update(self, delta_time: float):
        if self.target:
            # Interpola rotazione
            target_rotation_x = self.target.rotation_x
            target_rotation_y = self.target.rotation_y

            # Normalizza le rotazioni per evitare salti improvvisi
            while target_rotation_y - self.rotation_y > math.pi:
                target_rotation_y -= 2 * math.pi
            while target_rotation_y - self.rotation_y < -math.pi:
                target_rotation_y += 2 * math.pi

            self.rotation_x += (target_rotation_x - self.rotation_x) * self.lag_factor
            self.rotation_y += (target_rotation_y - self.rotation_y) * self.lag_factor

            # Calcola la posizione target considerando la velocità della nave
            forward = self.target.get_forward_vector()
            up = Vector3D(
                math.sin(self.target.rotation_y) * math.sin(self.target.rotation_x),
                math.cos(self.target.rotation_x),
                math.cos(self.target.rotation_y) * math.sin(self.target.rotation_x)
            )

            # Aggiungi un offset basato sulla velocità della nave
            velocity_magnitude = math.sqrt(
                self.target.velocity.x ** 2 +
                self.target.velocity.y ** 2 +
                self.target.velocity.z ** 2
            )

            # La distanza della camera aumenta con la velocità
            dynamic_distance = self.distance + (velocity_magnitude * 0.5)

            # Calcola la posizione target
            target_pos = self.target.position
            target_pos = target_pos + (forward * -dynamic_distance)
            target_pos = target_pos + (up * self.height_offset)

            # Aggiungi un leggero offset nella direzione della velocità per dare senso di movimento
            velocity_offset = self.target.velocity * 0.2
            target_pos = target_pos + velocity_offset

            # Interpola la posizione della camera
            self.position = Vector3D(
                self.position.x + (target_pos.x - self.position.x) * self.lag_factor,
                self.position.y + (target_pos.y - self.position.y) * self.lag_factor,
                self.position.z + (target_pos.z - self.position.z) * self.lag_factor
            )

class SpaceGameEngine:
    def __init__(self, use_pygame=False):
        self.use_pygame = use_pygame and PYGAME_AVAILABLE

        if self.use_pygame:
            pygame.init()
            self.screen_width, self.screen_height = 800, 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Space Exploration")
        else:
            self.screen_height, self.screen_width = curses.LINES - 1, curses.COLS - 1
            self.init_colors()

        self.objects: List[GameObject] = []
        self.celestial_objects: List[CelestialObject] = []
        self.player_ship: GameObject = None
        self.camera: Camera = None
        self.last_frame_time = time.time()
        self.fov = 60.0
        self.near_plane = 0.00001
        self.far_plane = 100000.0

        self.hud = HUDSystem(is_pygame=use_pygame)
        self.setup_hud()
        self.generate_celestial_objects(10000)

    def generate_celestial_objects(self, count: int):
        """Genera un mix di stelle e pianeti."""
        self.celestial_objects = []
        for _ in range(count):
            pos = Vector3D(
                random.uniform(-200, 200),  # Ridotto il range
                random.uniform(-200, 200),  # Ridotto il range
                random.uniform(-400, 400)  # Ridotto il range
            )

            # Aumentata la probabilità di pianeti
            obj_type = 'planet' if random.random() < 0.2 else 'star'
            size = random.uniform(2.0, 5.0) if obj_type == 'planet' else 1.0

            self.celestial_objects.append(CelestialObject(pos, obj_type, size))

    def project_point(self, point: Vector3D) -> Tuple[int, int]:
        if not self.camera:
            return (-1, -1)

        # Calcola la posizione relativa alla camera
        relative_point = point - self.camera.position

        # Applica la rotazione della camera
        rotated = relative_point.rotate_y(-self.camera.rotation_y).rotate_x(-self.camera.rotation_x)

        # Modifica: rimuoviamo il controllo sul near_plane e aumentiamo il far_plane
        if rotated.z <= 0 or rotated.z > self.far_plane:
            return (-1, -1)

        # Proiezione prospettica con FOV corretto
        fov_rad = math.radians(self.fov)
        aspect_ratio = self.screen_width / self.screen_height

        # Modificato il fattore di scala per migliorare la visibilità a lunga distanza
        scale_y = 1.0 / math.tan(fov_rad / 2)
        scale_x = scale_y / aspect_ratio

        # Rimuoviamo il distance_scale che causava problemi con gli oggetti lontani
        screen_x = rotated.x / rotated.z * scale_x
        screen_y = rotated.y / rotated.z * scale_y

        # Converti in coordinate schermo
        x = int(self.screen_width / 2 * (1 + screen_x))
        y = int(self.screen_height / 2 * (1 - screen_y))

        return (x, y)

    def render_celestial_object(self, surface, obj: CelestialObject, projected_pos: Tuple[int, int]):
        if not self.camera:
            return

        # Calcola la distanza dall'oggetto alla camera
        relative_pos = Vector3D(
            obj.position.x - self.camera.position.x,
            obj.position.y - self.camera.position.y,
            obj.position.z - self.camera.position.z
        )

        # Applica la rotazione della camera per ottenere la vera distanza prospettica
        rotated_pos = relative_pos.rotate_y(-self.camera.rotation_y).rotate_x(-self.camera.rotation_x)
        distance = abs(rotated_pos.z)  # Usiamo la Z rotata per la distanza prospettica

        # Calcola la dimensione apparente
        apparent_size = obj.size / max(distance, 0.1)

        # Implementazione del dithering: se l'oggetto è molto piccolo ma ancora visibile
        dithering_threshold = 0.1  # 10% di un pixel
        if apparent_size < 0.25 and apparent_size > dithering_threshold:
            if obj.type == 'star':
                brightness = obj.features['brightness']
                # Applica dithering solo se l'oggetto è abbastanza luminoso
                if brightness > 0.3 and random.random() < apparent_size * brightness * 4:
                    color = self._get_pygame_color(obj.features['color'])
                    # Riduce l'intensità del colore in base alla dimensione apparente
                    color = tuple(int(c * (apparent_size * 4)) for c in color)
                    if self.use_pygame:
                        pygame.draw.circle(surface, color, projected_pos, 1)
            return

        char, detailed_view, color = obj.get_char_at_distance(distance)

        if self.use_pygame:
            # Modifica il rendering per Pygame
            if not detailed_view:
                # Rendering semplificato per oggetti distanti
                size = max(1, int(apparent_size * 15))  # Aumentato il fattore di scala
                color_rgb = self._get_pygame_color(color)

                # Aggiungi effetto di luminosità per le stelle
                if obj.type == 'star':
                    brightness = obj.features['brightness']
                    # Aggiungi glow per stelle luminose
                    if brightness > 0.5:
                        glow_size = size * 2
                        glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                        for radius in range(glow_size, 0, -1):
                            alpha = int(255 * (radius / glow_size) * 0.3 * brightness)
                            glow_color = (*color_rgb, alpha)
                            pygame.draw.circle(glow_surface, glow_color, (glow_size, glow_size), radius)
                        surface.blit(glow_surface,
                                     (projected_pos[0] - glow_size, projected_pos[1] - glow_size),
                                     special_flags=pygame.BLEND_ADD)

                pygame.draw.circle(surface, color_rgb, projected_pos, size)
            else:
                # Rendering dettagliato per oggetti vicini
                size = max(5, int(apparent_size * 30))
                color_rgb = self._get_pygame_color(color)
                pygame.draw.circle(surface, color_rgb, projected_pos, size)

                if obj.type == 'planet':
                    self._add_planet_details(surface, projected_pos, size, obj, color_rgb)

    def _get_pygame_color(self, curses_color):
        """Converte i colori curses in colori RGB per Pygame."""
        color_map = {
            curses.COLOR_RED: (255, 0, 0),
            curses.COLOR_YELLOW: (255, 255, 0),
            curses.COLOR_BLUE: (0, 0, 255),
            curses.COLOR_CYAN: (0, 255, 255),
            curses.COLOR_MAGENTA: (255, 0, 255),
            curses.COLOR_WHITE: (255, 255, 255)
        }
        return color_map.get(curses_color, (255, 255, 255))

    def _add_planet_details(self, surface, pos, size, obj, color_rgb):
        """Aggiunge dettagli al rendering del pianeta in Pygame."""
        if obj.features['surface_type'] == 'rocky':
            for _ in range(int(size / 2)):
                offset_x = random.randint(-size, size)
                offset_y = random.randint(-size, size)
                if offset_x * offset_x + offset_y * offset_y < size * size:
                    crater_pos = (pos[0] + offset_x, pos[1] + offset_y)
                    crater_size = random.randint(1, max(2, int(size / 5)))
                    darker_color = tuple(max(0, c - 50) for c in color_rgb)
                    pygame.draw.circle(surface, darker_color, crater_pos, crater_size)
        elif obj.features['surface_type'] == 'gaseous':
            for i in range(obj.features['bands']):
                y_offset = -size + (i * size * 2 // obj.features['bands'])
                rect = pygame.Rect(pos[0] - size, pos[1] + y_offset, size * 2, size // obj.features['bands'])
                lighter_color = tuple(min(255, c + 30) for c in color_rgb)
                pygame.draw.ellipse(surface, lighter_color, rect)

    def _render_detailed_view(self, surface, scaled_view, projected_pos, color_pair):
        """Renderizza la vista dettagliata per curses."""
        half_height = len(scaled_view) // 2
        half_width = len(scaled_view[0]) // 2 if scaled_view else 0

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
                        surface.addch(screen_y, screen_x, char, color_pair)
                except curses.error:
                    pass

    def setup_hud(self):
        # Velocità
        velocity_element = HUDElement(
            (10, 10),  # Posizione adattata per Pygame
            lambda state: f"Velocity: {state['velocity']:.1f} u/s"
        )
        velocity_element.color = (0, 255, 0)  # Verde per Pygame
        self.hud.add_element("velocity", velocity_element)

        # Controlli
        controls_element = HUDElement(
            (100, 10),
            lambda state: [
                "Controls:",
                "W/S     - Thrust forward/back",
                "A/D     - Turn left/right",
                "R/F     - Pitch up/down",
                "O/P     - Adjust FOV",
                "Q       - Quit"
            ]
        )
        controls_element.color = (200, 200, 200)  # Grigio chiaro per Pygame
        self.hud.add_element("controls", controls_element)        

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

        self.hud.add_element("brake", HUDElement(
            (11, 0),
            lambda state: "EMERGENCY BRAKE ACTIVE" if state.get('braking', False) else ""
        ))

    def init_colors(self):
        """Inizializza le coppie di colori per curses."""
        curses.start_color()
        curses.use_default_colors()

        # Inizializza le coppie di colori base
        for i in range(1, 8):
            curses.init_pair(i, i, -1)

    def _render_celestial_object_pygame(self, screen, obj: CelestialObject, projected_pos: Tuple[int, int],
                                    distance: float, apparent_size: float):
        """Render celestial object using Pygame"""
        # Converti i colori curses in colori RGB per Pygame
        color_map = {
            curses.COLOR_RED: (255, 0, 0),
            curses.COLOR_YELLOW: (255, 255, 0),
            curses.COLOR_BLUE: (0, 0, 255),
            curses.COLOR_CYAN: (0, 255, 255),
            curses.COLOR_MAGENTA: (255, 0, 255),
            curses.COLOR_WHITE: (255, 255, 255)
        }

        # Ottieni il colore RGB in base alle caratteristiche dell'oggetto
        base_color = color_map.get(obj.features['color'], (255, 255, 255))

        # Calcola la dimensione base dell'oggetto
        base_size = max(2, int((obj.size * 20) / distance))

        if obj.type == 'star':
            # Rendering stelle
            brightness = obj.features['brightness']
            color = tuple(int(c * brightness) for c in base_color)
            pygame.draw.circle(screen, color, projected_pos, base_size)
            
            # Aggiungi bagliore per stelle luminose
            if brightness > 0.7:
                glow_size = base_size * 2
                glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                for radius in range(glow_size, 0, -1):
                    alpha = int(255 * (radius / glow_size) * 0.3)
                    glow_color = (*color, alpha)
                    pygame.draw.circle(glow_surface, glow_color, (glow_size, glow_size), radius)
                screen.blit(glow_surface, 
                          (projected_pos[0] - glow_size, projected_pos[1] - glow_size), 
                          special_flags=pygame.BLEND_ADD)

        elif obj.type == 'planet':
            # Rendering pianeti
            if distance < 50:  # Rendering dettagliato per pianeti vicini
                size = int(base_size * 1.5)
                
                # Disegna il corpo principale del pianeta
                pygame.draw.circle(screen, base_color, projected_pos, size)
                
                # Aggiungi dettagli in base al tipo di superficie
                if obj.features['surface_type'] == 'rocky':
                    for _ in range(int(size / 2)):
                        crater_pos = (
                            projected_pos[0] + random.randint(-size, size),
                            projected_pos[1] + random.randint(-size, size)
                        )
                        if (crater_pos[0] - projected_pos[0])**2 + (crater_pos[1] - projected_pos[1])**2 < size**2:
                            crater_size = random.randint(1, max(2, int(size/5)))
                            pygame.draw.circle(screen, (max(0, c-50) for c in base_color), crater_pos, crater_size)
                
                elif obj.features['surface_type'] == 'gaseous':
                    for i in range(obj.features['bands']):
                        band_height = size // obj.features['bands']
                        band_y = projected_pos[1] - size + (i * band_height * 2)
                        band_color = tuple(min(255, c + 30) for c in base_color)
                        pygame.draw.ellipse(screen, band_color, 
                                         (projected_pos[0] - size, 
                                          band_y, 
                                          size * 2, 
                                          band_height))
            else:
                # Rendering semplificato per pianeti distanti
                pygame.draw.circle(screen, base_color, projected_pos, base_size)

    def _render_celestial_object_curses(self, stdscr, obj: CelestialObject, projected_pos: Tuple[int, int],
                                        distance: float, apparent_size: float):
        """Render celestial object using curses (original implementation)"""
        char, detailed_view, color = obj.get_char_at_distance(distance)
        color_pair = curses.color_pair(color)

        if not detailed_view:
            if (0 <= projected_pos[0] < self.screen_width and
                    0 <= projected_pos[1] < self.screen_height):
                try:
                    stdscr.addch(projected_pos[1], projected_pos[0], char, color_pair)
                except curses.error:
                    pass
        else:
            size_factor = min(5.0, max(1.0, 20.0 / distance))
            scaled_view = self.scale_detailed_view(detailed_view, size_factor)

            half_height = len(scaled_view) // 2
            half_width = len(scaled_view[0]) // 2 if scaled_view else 0

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
                            stdscr.addch(screen_y, screen_x, char, color_pair)
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

    def update(self, surface):
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if not self.use_pygame:
            # Gestione input curses
            key = surface.getch()
            key_char = key
        else:
            # Gestione input pygame
            keys = pygame.key.get_pressed()
            key_char = None
            if keys[pygame.K_w]: key_char = ord('w')
            elif keys[pygame.K_s]: key_char = ord('s')
            elif keys[pygame.K_a]: key_char = ord('a')
            elif keys[pygame.K_d]: key_char = ord('d')
            elif keys[pygame.K_r]: key_char = ord('r')
            elif keys[pygame.K_f]: key_char = ord('f')
            elif keys[pygame.K_SPACE]: key_char = ord(' ')
            elif keys[pygame.K_o]: key_char = ord('o')
            elif keys[pygame.K_p]: key_char = ord('p')
            elif keys[pygame.K_q]: key_char = ord('q')

        braking = False

        if self.player_ship and (key_char != -1 if not self.use_pygame else key_char is not None):
            thrust = 20.0 * delta_time
            rotation_speed = 2.0

            if key_char == ord('w'):
                self.player_ship.apply_thrust(thrust)
            elif key_char == ord('s'):
                self.player_ship.apply_thrust(-thrust)
            elif key_char == ord('a'):
                self.player_ship.target_rotation_y += rotation_speed * delta_time
            elif key_char == ord('d'):
                self.player_ship.target_rotation_y -= rotation_speed * delta_time
            elif key_char == ord('r'):
                self.player_ship.target_rotation_x -= rotation_speed * delta_time
            elif key_char == ord('f'):
                self.player_ship.target_rotation_x += rotation_speed * delta_time
            elif key_char == ord(' '):
                self.player_ship.emergency_brake(delta_time)
                braking = True
            elif key_char == ord('o'):
                self.fov = max(30.0, self.fov - 5.0)
            elif key_char == ord('p'):
                self.fov = min(120.0, self.fov + 5.0)
            elif key_char == ord('q'):
                return False

        # Aggiorna camera e oggetti
        if self.camera:
            self.camera.update(delta_time)

        for obj in self.objects:
            obj.update(delta_time)

        # Aggiorna la generazione degli oggetti celesti
        if self.player_ship:
            # Rimuovi oggetti troppo lontani
            self.celestial_objects = [obj for obj in self.celestial_objects
                                      if (obj.position - self.player_ship.position).z < 400]

            # Aggiungi nuovi oggetti
            while len(self.celestial_objects) < 100:  # Ridotto il numero di oggetti
                forward = self.player_ship.get_forward_vector()
                random_offset = Vector3D(
                    random.uniform(-200, 200),
                    random.uniform(-200, 200),
                    0
                )
                pos = self.player_ship.position + (forward * 400) + random_offset

                obj_type = 'planet' if random.random() < 0.2 else 'star'
                size = random.uniform(2.0, 5.0) if obj_type == 'planet' else 1.0

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

            if self.use_pygame:
                self.render_frame(surface)
            else:
                self.hud.render(surface, game_state)

        return True

    def render_frame(self, stdscr=None) -> None:
        if self.use_pygame:
            self.screen.fill((0, 0, 0))  # Pulisci lo schermo

            # Rendering oggetti celesti
            for obj in self.celestial_objects:
                projected = self.project_point(obj.position)
                if projected[0] >= 0:
                    self.render_celestial_object(self.screen, obj, projected)

            # Rendering oggetti
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
                        pygame.draw.line(self.screen, (255, 255, 255), start, end)

            # Rendering HUD
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

                self.hud.render(self.screen, game_state)

            pygame.display.flip()  # Aggiorna lo schermo
        else:
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

def main(stdscr, use_pygame=False):    
    if not use_pygame:
        curses.curs_set(0)
        stdscr.nodelay(1)

    engine = SpaceGameEngine(use_pygame)

    # Crea la nave del giocatore
    player_ship = engine.create_player_ship()
    engine.objects.append(player_ship)
    engine.player_ship = player_ship

    # Inizializza la camera
    engine.camera = Camera(player_ship)

    if use_pygame:
        screen = engine.screen
        clock = pygame.time.Clock()
        target_fps = 60  # Aumentato a 60 FPS per una migliore fluidità
    else:
        screen = stdscr

    running = True
    while running:
        if use_pygame:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            # Limita il framerate
            clock.tick(target_fps)
            
        else:
            time.sleep(0.033)  # Circa 30 FPS per curses

        # Aggiorna la logica di gioco
        running = engine.update(screen)

        # Renderizza il frame (per curses - per pygame è già gestito in update)
        if not use_pygame:
            engine.render_frame(screen)

    if use_pygame:
        pygame.quit()


if __name__ == "__main__":
    if PYGAME_AVAILABLE and False:
        use_pygame = input("Use Pygame? (y/n): ").lower() == 'y'
    else:
        use_pygame = PYGAME_AVAILABLE

    #use_pygame = False

    if use_pygame:
        main(None, use_pygame)
    else:
        curses.wrapper(main)