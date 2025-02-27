import pygame
import numpy as np
import math
from collections import defaultdict

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return Vector3()
    
    def to_tuple(self):
        return (self.x, self.y, self.z)
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])


class Matrix4x4:
    def __init__(self):
        self.m = np.identity(4)
    
    @staticmethod
    def translation(x, y, z):
        mat = Matrix4x4()
        mat.m[0, 3] = x
        mat.m[1, 3] = y
        mat.m[2, 3] = z
        return mat
    
    @staticmethod
    def rotation_x(angle):
        mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        mat.m[1, 1] = c
        mat.m[1, 2] = -s
        mat.m[2, 1] = s
        mat.m[2, 2] = c
        return mat
    
    @staticmethod
    def rotation_y(angle):
        mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        mat.m[0, 0] = c
        mat.m[0, 2] = s
        mat.m[2, 0] = -s
        mat.m[2, 2] = c
        return mat
    
    @staticmethod
    def rotation_z(angle):
        mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        mat.m[0, 0] = c
        mat.m[0, 1] = -s
        mat.m[1, 0] = s
        mat.m[1, 1] = c
        return mat
    
    @staticmethod
    def scale(x, y, z):
        mat = Matrix4x4()
        mat.m[0, 0] = x
        mat.m[1, 1] = y
        mat.m[2, 2] = z
        return mat
    
    @staticmethod
    def perspective(fov, aspect, near, far):
        mat = Matrix4x4()
        tan_half_fov = math.tan(fov / 2)
        mat.m[0, 0] = 1 / (aspect * tan_half_fov)
        mat.m[1, 1] = 1 / tan_half_fov
        mat.m[2, 2] = -(far + near) / (far - near)
        mat.m[2, 3] = -2 * far * near / (far - near)
        mat.m[3, 2] = -1
        mat.m[3, 3] = 0
        return mat
    
    def __mul__(self, other):
        result = Matrix4x4()
        result.m = np.matmul(self.m, other.m)
        return result
    
    def transform_vector(self, v):
        v4 = np.array([v.x, v.y, v.z, 1])
        result = np.matmul(self.m, v4)
        if result[3] != 0:
            result = result / result[3]
        return Vector3(result[0], result[1], result[2])


class Vertex:
    def __init__(self, position, color=(255, 255, 255)):
        self.position = position
        self.color = color
        self.transformed = Vector3()
        self.projected = (0, 0)
        self.z = 0  # for depth sorting


class Edge:
    def __init__(self, start_idx, end_idx, color=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.color = color


class Triangle:
    def __init__(self, a, b, c, color=None, fill_color=None):
        self.vertices = [a, b, c]
        self.color = color
        self.fill_color = fill_color
        self.center = Vector3()  # for depth sorting
        self.normal = Vector3()
        self.is_visible = True


class Mesh:
    def __init__(self, vertices=None, edges=None, triangles=None):
        self.vertices = vertices or []
        self.edges = edges or []
        self.triangles = triangles or []
        self.position = Vector3()
        self.rotation = Vector3()
        self.scale = Vector3(1, 1, 1)
        self.color = (255, 255, 255)
        self.fill_color = None
        self.is_visible = True
    
    @staticmethod
    def create_cube(size=1.0, color=(255, 255, 255), fill_color=None):
        mesh = Mesh()
        half_size = size / 2
        
        # Define vertices
        vertices = [
            Vector3(-half_size, -half_size, -half_size),  # 0: bottom-left-back
            Vector3(half_size, -half_size, -half_size),   # 1: bottom-right-back
            Vector3(half_size, half_size, -half_size),    # 2: top-right-back
            Vector3(-half_size, half_size, -half_size),   # 3: top-left-back
            Vector3(-half_size, -half_size, half_size),   # 4: bottom-left-front
            Vector3(half_size, -half_size, half_size),    # 5: bottom-right-front
            Vector3(half_size, half_size, half_size),     # 6: top-right-front
            Vector3(-half_size, half_size, half_size)     # 7: top-left-front
        ]
        
        for v in vertices:
            mesh.vertices.append(Vertex(v, color))
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]
        
        for start_idx, end_idx in edges:
            mesh.edges.append(Edge(start_idx, end_idx, color))
        
        # Define triangles (two per face for a total of 12)
        triangles = [
            # Back face
            (0, 1, 2), (0, 2, 3),
            # Front face
            (4, 6, 5), (4, 7, 6),
            # Left face
            (0, 3, 7), (0, 7, 4),
            # Right face
            (1, 5, 6), (1, 6, 2),
            # Top face
            (3, 2, 6), (3, 6, 7),
            # Bottom face
            (0, 4, 5), (0, 5, 1)
        ]
        
        for a, b, c in triangles:
            mesh.triangles.append(Triangle(a, b, c, color, fill_color))
        
        mesh.color = color
        mesh.fill_color = fill_color
        return mesh
    
    @staticmethod
    def create_sphere(radius=1.0, segments=16, color=(255, 255, 255), fill_color=None):
        mesh = Mesh()
        
        # Generate vertices
        for j in range(segments + 1):
            theta = j * math.pi / segments
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            for i in range(segments * 2):
                phi = i * math.pi / segments
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                
                x = radius * sin_theta * cos_phi
                y = radius * cos_theta
                z = radius * sin_theta * sin_phi
                
                mesh.vertices.append(Vertex(Vector3(x, y, z), color))
        
        # Generate triangles
        for j in range(segments):
            for i in range(segments * 2):
                next_i = (i + 1) % (segments * 2)
                
                p1 = j * (segments * 2) + i
                p2 = j * (segments * 2) + next_i
                p3 = (j + 1) * (segments * 2) + i
                p4 = (j + 1) * (segments * 2) + next_i
                
                # Skip degenerate triangles at the poles
                if j != 0:
                    mesh.triangles.append(Triangle(p1, p2, p3, color, fill_color))
                
                if j != segments - 1:
                    mesh.triangles.append(Triangle(p2, p4, p3, color, fill_color))
                
                # Add edges for wireframe
                mesh.edges.append(Edge(p1, p2, color))
                mesh.edges.append(Edge(p1, p3, color))
                
                if j == segments - 1:
                    mesh.edges.append(Edge(p3, p4, color))
        
        mesh.color = color
        mesh.fill_color = fill_color
        return mesh

    @staticmethod
    def create_ellipsoid(radius_x=1.0, radius_y=1.0, radius_z=1.0, segments=16, color=(255, 255, 255), fill_color=None):
        mesh = Mesh.create_sphere(1.0, segments, color, fill_color)
        
        # Scale the sphere to create an ellipsoid
        for vertex in mesh.vertices:
            vertex.position.x *= radius_x
            vertex.position.y *= radius_y
            vertex.position.z *= radius_z
        
        return mesh
    
    def apply_boolean_operation(self, other_mesh, operation="subtract", threshold=0.01):
        if operation not in ["subtract", "union", "intersect"]:
            raise ValueError("Operation must be one of: subtract, union, intersect")
        
        result_mesh = Mesh()
        
        # First, we need to triangulate both meshes to a higher resolution for better boolean operations
        self_triangulated = self.triangulate(2)
        other_triangulated = other_mesh.triangulate(2)
        
        # Depending on the operation, mark triangles as visible or invisible
        for triangle in self_triangulated.triangles:
            # Calculate the center of the triangle
            center = Vector3()
            for vertex_idx in triangle.vertices:
                vertex_pos = self_triangulated.vertices[vertex_idx].position
                center = center + vertex_pos
            center = center * (1/3)
            
            # Check if this point is inside the other mesh
            is_inside_other = other_triangulated.is_point_inside(center)
            
            if operation == "subtract":
                triangle.is_visible = not is_inside_other
            elif operation == "union":
                triangle.is_visible = True
            elif operation == "intersect":
                triangle.is_visible = is_inside_other
        
        for triangle in other_triangulated.triangles:
            # Calculate the center of the triangle
            center = Vector3()
            for vertex_idx in triangle.vertices:
                vertex_pos = other_triangulated.vertices[vertex_idx].position
                center = center + vertex_pos
            center = center * (1/3)
            
            # Check if this point is inside the other mesh
            is_inside_self = self_triangulated.is_point_inside(center)
            
            if operation == "subtract":
                # In subtract operation, we only add the boundary of the subtracted region
                if not is_inside_self:
                    continue
                triangle.is_visible = False
            elif operation == "union":
                triangle.is_visible = not is_inside_self
            elif operation == "intersect":
                triangle.is_visible = is_inside_self
        
        # Create a new mesh with only the visible triangles
        result_mesh = Mesh()
        
        # Add all visible triangles from the first mesh
        vertex_map = {}
        for i, triangle in enumerate(self_triangulated.triangles):
            if triangle.is_visible:
                new_vertices = []
                for vertex_idx in triangle.vertices:
                    vertex = self_triangulated.vertices[vertex_idx]
                    # Avoid duplicate vertices
                    vertex_key = (vertex.position.x, vertex.position.y, vertex.position.z)
                    if vertex_key not in vertex_map:
                        vertex_map[vertex_key] = len(result_mesh.vertices)
                        result_mesh.vertices.append(Vertex(vertex.position, vertex.color))
                    new_vertices.append(vertex_map[vertex_key])
                
                result_mesh.triangles.append(Triangle(
                    new_vertices[0], 
                    new_vertices[1], 
                    new_vertices[2], 
                    self_triangulated.color, 
                    self_triangulated.fill_color
                ))
        
        # For subtract operation, add the intersection boundary
        if operation == "subtract":
            # Add edges at the intersection boundary
            for triangle in other_triangulated.triangles:
                if not triangle.is_visible:
                    for i in range(3):
                        start_idx = triangle.vertices[i]
                        end_idx = triangle.vertices[(i+1)%3]
                        
                        start_vertex = other_triangulated.vertices[start_idx].position
                        end_vertex = other_triangulated.vertices[end_idx].position
                        
                        # Check if this edge is at the boundary (one point inside, one outside)
                        start_inside = self_triangulated.is_point_inside(start_vertex)
                        end_inside = self_triangulated.is_point_inside(end_vertex)
                        
                        if start_inside != end_inside:
                            # Find and add intersection point
                            # (simplified - in a real implementation you'd need ray-triangle intersection)
                            t = 0.5  # Simple approximation
                            intersection = start_vertex + (end_vertex - start_vertex) * t
                            
                            # Add edges for the intersection
                            v1 = len(result_mesh.vertices)
                            result_mesh.vertices.append(Vertex(intersection, (255, 0, 0)))  # Red for intersection
                            
                            # Find closest point already in result_mesh
                            closest_dist = float('inf')
                            closest_idx = -1
                            for i, v in enumerate(result_mesh.vertices[:-1]):  # Skip the one we just added
                                dist = (v.position - intersection).length()
                                if dist < closest_dist and dist > threshold:
                                    closest_dist = dist
                                    closest_idx = i
                            
                            if closest_idx != -1:
                                result_mesh.edges.append(Edge(v1, closest_idx, (255, 0, 0)))
        
        return result_mesh
    
    def triangulate(self, subdivisions=1):
        if subdivisions <= 0:
            return self
        
        result = Mesh()
        result.vertices = self.vertices.copy()
        result.color = self.color
        result.fill_color = self.fill_color
        
        # Subdivide each triangle
        for triangle in self.triangles:
            a_idx, b_idx, c_idx = triangle.vertices
            a = self.vertices[a_idx].position
            b = self.vertices[b_idx].position
            c = self.vertices[c_idx].position
            
            # Find midpoints
            ab = a + (b - a) * 0.5
            bc = b + (c - b) * 0.5
            ca = c + (a - c) * 0.5
            
            # Create new vertices
            ab_idx = len(result.vertices)
            result.vertices.append(Vertex(ab, self.color))
            
            bc_idx = len(result.vertices)
            result.vertices.append(Vertex(bc, self.color))
            
            ca_idx = len(result.vertices)
            result.vertices.append(Vertex(ca, self.color))
            
            # Create 4 new triangles
            result.triangles.append(Triangle(a_idx, ab_idx, ca_idx, self.color, self.fill_color))
            result.triangles.append(Triangle(b_idx, bc_idx, ab_idx, self.color, self.fill_color))
            result.triangles.append(Triangle(c_idx, ca_idx, bc_idx, self.color, self.fill_color))
            result.triangles.append(Triangle(ab_idx, bc_idx, ca_idx, self.color, self.fill_color))
        
        # Recursively subdivide if needed
        if subdivisions > 1:
            result = result.triangulate(subdivisions - 1)
        
        return result
    
    def is_point_inside(self, point):
        # A simple ray casting algorithm to determine if a point is inside a mesh
        # This is a simplified version and might not work well for all meshes
        
        # Cast a ray in the +X direction and count intersections
        intersections = 0
        ray_dir = Vector3(1, 0, 0)
        ray_origin = point
        
        for triangle in self.triangles:
            a = self.vertices[triangle.vertices[0]].position
            b = self.vertices[triangle.vertices[1]].position
            c = self.vertices[triangle.vertices[2]].position
            
            # Calculate triangle normal
            edge1 = b - a
            edge2 = c - a
            normal = edge1.cross(edge2).normalize()
            
            # Check if ray and plane are parallel
            ndotdir = normal.dot(ray_dir)
            if abs(ndotdir) < 1e-6:
                continue
            
            # Calculate distance to plane
            d = -normal.dot(a)
            t = -(normal.dot(ray_origin) + d) / ndotdir
            
            # Check if intersection is behind the ray
            if t < 0:
                continue
            
            # Calculate intersection point
            intersection = ray_origin + ray_dir * t
            
            # Check if intersection point is inside the triangle
            edge1 = b - a
            edge2 = c - a
            edge3 = a - c
            
            p0 = intersection - a
            p1 = intersection - b
            p2 = intersection - c
            
            c0 = edge1.cross(p0)
            c1 = edge2.cross(p1)
            c2 = edge3.cross(p2)
            
            if normal.dot(c0) > 0 and normal.dot(c1) > 0 and normal.dot(c2) > 0:
                intersections += 1
        
        # If number of intersections is odd, point is inside
        return intersections % 2 == 1


class Sphere:
    def __init__(self, center=None, radius=1.0, color=(255, 255, 255), fill_color=None):
        self.center = center or Vector3()
        self.radius = radius
        self.radius_x = radius
        self.radius_y = radius
        self.radius_z = radius
        self.color = color
        self.fill_color = fill_color
        self.segments = 32  # Detail level for rendering
        self.is_visible = True
        self.boolean_operations = []  # List of spheres and operations to apply
    
    def set_ellipsoid(self, radius_x, radius_y, radius_z):
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.radius_z = radius_z
        return self
    
    def apply_boolean(self, sphere, operation="subtract"):
        if operation not in ["subtract", "union", "intersect"]:
            raise ValueError("Operation must be one of: subtract, union, intersect")
        
        self.boolean_operations.append((sphere, operation))
        return self
    
    def is_point_inside(self, point):
        # Transform point to ellipsoid space
        local_point = Vector3(
            (point.x - self.center.x) / self.radius_x,
            (point.y - self.center.y) / self.radius_y,
            (point.z - self.center.z) / self.radius_z
        )
        
        # Check if point is inside base ellipsoid
        is_inside = local_point.length() <= 1.0
        
        # Apply boolean operations
        for sphere, operation in self.boolean_operations:
            point_in_other = sphere.is_point_inside(point)
            
            if operation == "subtract":
                is_inside = is_inside and not point_in_other
            elif operation == "union":
                is_inside = is_inside or point_in_other
            elif operation == "intersect":
                is_inside = is_inside and point_in_other
        
        return is_inside
    
    def generate_mesh(self):
        # Generate a mesh representation of the sphere with boolean operations
        base_mesh = Mesh.create_ellipsoid(
            self.radius_x, self.radius_y, self.radius_z, 
            self.segments, self.color, self.fill_color
        )
        
        # Apply translation
        for vertex in base_mesh.vertices:
            vertex.position = vertex.position + self.center
        
        # Apply boolean operations
        result_mesh = base_mesh
        for sphere, operation in self.boolean_operations:
            other_mesh = sphere.generate_mesh()
            result_mesh = result_mesh.apply_boolean_operation(other_mesh, operation)
        
        return result_mesh


class Camera:
    def __init__(self, position=None, target=None, up=None):
        self.position = position or Vector3(0, 0, -5)
        self.target = target or Vector3(0, 0, 0)
        self.up = up or Vector3(0, 1, 0)
        self.fov = math.pi / 3  # 60 degrees
        self.aspect = 16 / 9
        self.near = 0.1
        self.far = 1000.0
    
    def get_view_matrix(self):
        z = (self.position - self.target).normalize()
        x = self.up.cross(z).normalize()
        y = z.cross(x)
        
        view = Matrix4x4()
        view.m[0, 0] = x.x
        view.m[0, 1] = y.x
        view.m[0, 2] = z.x
        view.m[0, 3] = 0
        
        view.m[1, 0] = x.y
        view.m[1, 1] = y.y
        view.m[1, 2] = z.y
        view.m[1, 3] = 0
        
        view.m[2, 0] = x.z
        view.m[2, 1] = y.z
        view.m[2, 2] = z.z
        view.m[2, 3] = 0
        
        view.m[3, 0] = -x.dot(self.position)
        view.m[3, 1] = -y.dot(self.position)
        view.m[3, 2] = -z.dot(self.position)
        view.m[3, 3] = 1
        
        return view
    
    def get_projection_matrix(self):
        return Matrix4x4.perspective(self.fov, self.aspect, self.near, self.far)


class Engine3D:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D CPU Engine")
        self.clock = pygame.time.Clock()
        
        self.meshes = []
        self.spheres = []
        self.camera = Camera()
        
        self.view_matrix = Matrix4x4()
        self.projection_matrix = Matrix4x4()
        
        self.wireframe_mode = True
        self.fill_mode = True
        self.occlusion_culling = True
        self.z_buffer = np.full((width, height), float('inf'))
        
        # For mouse control
        self.mouse_prev_pos = (0, 0)
        self.mouse_pressed = False
        
        self.running = False
        self.fps = 60
    
    def run(self):
        """Main game loop"""
        self.running = True
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.mouse_pressed = True
                        self.mouse_prev_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        self.mouse_pressed = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_pressed:
                        self.handle_mouse_motion(event.pos)
            
            # Update scene
            self.update()
            
            # Render scene
            self.render_scene()
            
            # Cap the frame rate
            self.clock.tick(self.fps)
            
            # Display FPS
            fps = int(self.clock.get_fps())
            pygame.display.set_caption(f"3D CPU Engine - FPS: {fps}")
        
        pygame.quit()
    
    def handle_keydown(self, key):
        """Handle keyboard input"""
        speed = 0.1
        
        # Camera movement
        if key == pygame.K_w:
            direction = (self.camera.target - self.camera.position).normalize()
            self.camera.position = self.camera.position + direction * speed
            self.camera.target = self.camera.target + direction * speed
        elif key == pygame.K_s:
            direction = (self.camera.target - self.camera.position).normalize()
            self.camera.position = self.camera.position - direction * speed
            self.camera.target = self.camera.target - direction * speed
        elif key == pygame.K_a:
            direction = (self.camera.target - self.camera.position).normalize()
            right = self.camera.up.cross(direction).normalize()
            self.camera.position = self.camera.position - right * speed
            self.camera.target = self.camera.target - right * speed
        elif key == pygame.K_d:
            direction = (self.camera.target - self.camera.position).normalize()
            right = self.camera.up.cross(direction).normalize()
            self.camera.position = self.camera.position + right * speed
            self.camera.target = self.camera.target + right * speed
        elif key == pygame.K_q:
            self.camera.position = self.camera.position + self.camera.up * speed
            self.camera.target = self.camera.target + self.camera.up * speed
        elif key == pygame.K_e:
            self.camera.position = self.camera.position - self.camera.up * speed
            self.camera.target = self.camera.target - self.camera.up * speed
        
        # Rendering modes
        elif key == pygame.K_f:
            self.fill_mode = not self.fill_mode
        elif key == pygame.K_l:
            self.wireframe_mode = not self.wireframe_mode
        elif key == pygame.K_o:
            self.occlusion_culling = not self.occlusion_culling
        
        # Close application
        elif key == pygame.K_ESCAPE:
            self.running = False
    
    def handle_mouse_motion(self, pos):
        """Handle mouse motion for camera rotation"""
        dx = pos[0] - self.mouse_prev_pos[0]
        dy = pos[1] - self.mouse_prev_pos[1]
        self.mouse_prev_pos = pos
        
        # Sensitivity factor
        sensitivity = 0.005
        
        # Calculate new target position based on rotation around camera position
        direction = (self.camera.target - self.camera.position).normalize()
        
        # Rotate left/right (yaw)
        yaw_matrix = Matrix4x4.rotation_y(dx * sensitivity)
        direction = yaw_matrix.transform_vector(direction)
        
        # Rotate up/down (pitch)
        right = self.camera.up.cross(direction).normalize()
        pitch_matrix = Matrix4x4()
        pitch_matrix.m[0:3, 0:3] = np.array([
            [math.cos(dy * sensitivity), 0, math.sin(dy * sensitivity)],
            [0, 1, 0],
            [-math.sin(dy * sensitivity), 0, math.cos(dy * sensitivity)]
        ])
        direction = pitch_matrix.transform_vector(direction)
        
        # Update camera target
        self.camera.target = self.camera.position + direction
    
    def update(self):
        """Update all objects in the scene"""
        # This can be extended to update animations, physics, etc.
        pass
    
    def add_mesh(self, mesh):
        self.meshes.append(mesh)
        return len(self.meshes) - 1  # Return index for reference
    
    def add_sphere(self, sphere):
        self.spheres.append(sphere)
        return len(self.spheres) - 1  # Return index for reference
    
    def render_sphere(self, sphere):
        """Render a sphere directly with circle approximation for wireframe"""
        if not sphere.is_visible:
            return
        
        # Generate a mesh representation for complex spheres (with boolean operations)
        if sphere.boolean_operations:
            mesh = sphere.generate_mesh()
            self.render_mesh(mesh)
            return
        
        # For simple spheres/ellipsoids, use a more efficient circle-based approach
        # Transform center point
        center_world = Vector3(sphere.center.x, sphere.center.y, sphere.center.z)
        center_view = self.view_matrix.transform_vector(center_world)
        center_proj = self.projection_matrix.transform_vector(center_view)
        
        # Convert to screen coordinates
        screen_x = int((center_proj.x + 1) * self.width / 2)
        screen_y = int((1 - center_proj.y) * self.height / 2)
        center_screen = (screen_x, screen_y)
        
        # Calculate depth for z-buffer
        depth = center_view.z
        
        # Calculate radius in screen space (approximation)
        # This is a simplified calculation and might not be accurate for all camera positions
        dist_to_camera = (center_world - self.camera.position).length()
        screen_radius = int(sphere.radius * self.height / (dist_to_camera * math.tan(self.camera.fov / 2)))
        
        # For ellipsoids, calculate different radii (very simplified)
        if sphere.radius_x != sphere.radius_y or sphere.radius_x != sphere.radius_z:
            # Calculate average radius for now
            avg_radius = (sphere.radius_x + sphere.radius_y + sphere.radius_z) / 3
            screen_radius = int(avg_radius * self.height / (dist_to_camera * math.tan(self.camera.fov / 2)))
            
            # Draw as a simple circle for now - can be improved for true ellipsoid rendering
            if self.fill_mode and sphere.fill_color:
                if self.occlusion_culling:
                    self.draw_filled_circle_with_z_buffer(center_screen, screen_radius, depth, sphere.fill_color)
                else:
                    pygame.draw.circle(self.screen, sphere.fill_color, center_screen, screen_radius)
            
            if self.wireframe_mode:
                if self.occlusion_culling:
                    self.draw_circle_with_z_buffer(center_screen, screen_radius, depth, sphere.color)
                else:
                    pygame.draw.circle(self.screen, sphere.color, center_screen, screen_radius, 1)
            
            return
        
        # Draw the sphere
        if self.fill_mode and sphere.fill_color:
            if self.occlusion_culling:
                self.draw_filled_circle_with_z_buffer(center_screen, screen_radius, depth, sphere.fill_color)
            else:
                pygame.draw.circle(self.screen, sphere.fill_color, center_screen, screen_radius)
        
        if self.wireframe_mode:
            if self.occlusion_culling:
                self.draw_circle_with_z_buffer(center_screen, screen_radius, depth, sphere.color)
            else:
                pygame.draw.circle(self.screen, sphere.color, center_screen, screen_radius, 1)
    
    def draw_circle_with_z_buffer(self, center, radius, depth, color):
        """Draw a circle wireframe with z-buffer for hidden surface removal"""
        x0, y0 = center
        
        # Bresenham's circle algorithm
        x = radius
        y = 0
        decision = 1 - x
        
        while y <= x:
            # Draw 8 octants
            points = [
                (x0 + x, y0 + y), (x0 - x, y0 + y),
                (x0 + x, y0 - y), (x0 - x, y0 - y),
                (x0 + y, y0 + x), (x0 - y, y0 + x),
                (x0 + y, y0 - x), (x0 - y, y0 - x)
            ]
            
            for px, py in points:
                if 0 <= px < self.width and 0 <= py < self.height:
                    if depth < self.z_buffer[px, py]:
                        self.z_buffer[px, py] = depth
                        self.screen.set_at((px, py), color)
            
            y += 1
            if decision <= 0:
                decision += 2 * y + 1
            else:
                x -= 1
                decision += 2 * (y - x) + 1
    
    def draw_filled_circle_with_z_buffer(self, center, radius, depth, color):
        """Draw a filled circle with z-buffer for hidden surface removal"""
        x0, y0 = center
        
        # Scan through a square bounding the circle
        for y in range(max(0, y0 - radius), min(self.height, y0 + radius + 1)):
            for x in range(max(0, x0 - radius), min(self.width, x0 + radius + 1)):
                # Check if point is within the circle
                if (x - x0)**2 + (y - y0)**2 <= radius**2:
                    if depth < self.z_buffer[x, y]:
                        self.z_buffer[x, y] = depth
                        self.screen.set_at((x, y), color)


# Example usage
def main():
    """Example usage of the 3D engine"""
    # Create engine
    engine = Engine3D(800, 600)
    
    # Create a simple scene
    # Add a cube
    cube = Mesh.create_cube(1.0, (255, 255, 255), (100, 100, 100))
    cube.position = Vector3(0, 0, 0)
    engine.add_mesh(cube)
    
    # Add a sphere
    sphere = Sphere(Vector3(2, 0, 0), 1.0, (255, 255, 0), (100, 100, 0))
    engine.add_sphere(sphere)
    
    # Add an ellipsoid
    ellipsoid = Sphere(Vector3(-2, 0, 0), 1.0, (0, 255, 255), (0, 100, 100))
    ellipsoid.set_ellipsoid(0.75, 1.5, 0.75)
    engine.add_sphere(ellipsoid)
    
    # Add a sphere with boolean operations (cheese-like with holes)
    cheese = Sphere(Vector3(0, 2, 0), 1.0, (255, 165, 0), (200, 150, 50))
    
    # Add holes to the cheese
    hole1 = Sphere(Vector3(0.3, 2.3, 0.3), 0.4, (255, 165, 0))
    hole2 = Sphere(Vector3(-0.4, 1.8, 0.2), 0.3, (255, 165, 0))
    hole3 = Sphere(Vector3(0.1, 1.7, -0.5), 0.35, (255, 165, 0))
    
    # Apply boolean operations
    cheese.apply_boolean(hole1, "subtract")
    cheese.apply_boolean(hole2, "subtract")
    cheese.apply_boolean(hole3, "subtract")
    
    engine.add_sphere(cheese)
    
    # Add a mesh with boolean operations
    base_mesh = Mesh.create_cube(1.0, (255, 0, 255), (150, 0, 150))
    base_mesh.position = Vector3(0, -2, 0)
    
    cut_sphere = Sphere(Vector3(0, -2, 0), 0.8, (255, 0, 0))
    cut_mesh = cut_sphere.generate_mesh()
    
    # Create a mesh with a spherical hole in it
    boolean_mesh = base_mesh.apply_boolean_operation(cut_mesh, "subtract")
    engine.add_mesh(boolean_mesh)
    
    # Set up camera
    engine.camera.position = Vector3(0, 0, -5)
    engine.camera.target = Vector3(0, 0, 0)
    
    # Run the engine
    engine.run()


# Advanced demo with animated scene
def demo_solar_system():
    """Solar system demo with planets and moons"""
    # Create engine
    engine = Engine3D(1024, 768)
    
    # Sun at the center
    sun = Sphere(Vector3(0, 0, 0), 2.0, (255, 215, 0), (255, 165, 0))
    sun_idx = engine.add_sphere(sun)
    
    # Earth
    earth = Sphere(Vector3(8, 0, 0), 1.0, (70, 130, 180), (30, 100, 150))
    earth_idx = engine.add_sphere(earth)
    
    # Moon
    moon = Sphere(Vector3(10, 0, 0), 0.3, (200, 200, 200), (150, 150, 150))
    moon_idx = engine.add_sphere(moon)
    
    # Mars with craters
    mars = Sphere(Vector3(12, 0, 0), 0.8, (205, 92, 92), (160, 70, 70))
    
    # Add craters to Mars
    crater1 = Sphere(Vector3(12.3, 0.3, 0.3), 0.25, (205, 92, 92))
    crater2 = Sphere(Vector3(11.8, 0.4, -0.2), 0.2, (205, 92, 92))
    crater3 = Sphere(Vector3(12.1, -0.3, 0.5), 0.3, (205, 92, 92))
    
    mars.apply_boolean(crater1, "subtract")
    mars.apply_boolean(crater2, "subtract")
    mars.apply_boolean(crater3, "subtract")
    
    mars_idx = engine.add_sphere(mars)
    
    # Jupiter (gas giant - ellipsoidal)
    jupiter = Sphere(Vector3(16, 0, 0), 1.8, (255, 165, 0), (220, 140, 60))
    jupiter.set_ellipsoid(1.8, 1.7, 1.8)  # Slightly flattened at poles
    jupiter_idx = engine.add_sphere(jupiter)
    
    # Saturn with rings
    saturn = Sphere(Vector3(22, 0, 0), 1.5, (210, 180, 140), (180, 160, 120))
    saturn.set_ellipsoid(1.5, 1.4, 1.5)  # Slightly flattened at poles
    saturn_idx = engine.add_sphere(saturn)
    
    # Saturn's rings (simplified as a very flat ellipsoid)
    rings = Sphere(Vector3(22, 0, 0), 3.0, (210, 180, 140))
    rings.set_ellipsoid(3.0, 0.05, 3.0)
    
    # Cut out the center of the rings
    ring_hole = Sphere(Vector3(22, 0, 0), 1.8, (0, 0, 0))
    rings.apply_boolean(ring_hole, "subtract")
    
    rings_idx = engine.add_sphere(rings)
    
    # Camera setup for solar system view
    engine.camera.position = Vector3(0, 15, -30)
    engine.camera.target = Vector3(0, 0, 0)
    
    # Animation function
    def update_scene():
        # Get time for animation
        time = pygame.time.get_ticks() / 1000.0
        
        # Rotate planets around the sun
        earth_angle = time * 0.5
        earth.center.x = 8 * math.cos(earth_angle)
        earth.center.z = 8 * math.sin(earth_angle)
        
        # Moon around earth
        moon_angle = time * 2.0
        moon.center.x = earth.center.x + 2 * math.cos(moon_angle)
        moon.center.z = earth.center.z + 2 * math.sin(moon_angle)
        
        # Mars orbit
        mars_angle = time * 0.3
        mars.center.x = 12 * math.cos(mars_angle)
        mars.center.z = 12 * math.sin(mars_angle)
        
        # Update Mars craters
        crater1.center.x = mars.center.x + 0.3
        crater1.center.y = mars.center.y + 0.3
        crater1.center.z = mars.center.z + 0.3
        
        crater2.center.x = mars.center.x - 0.2
        crater2.center.y = mars.center.y + 0.4
        crater2.center.z = mars.center.z - 0.2
        
        crater3.center.x = mars.center.x + 0.1
        crater3.center.y = mars.center.y - 0.3
        crater3.center.z = mars.center.z + 0.5
        
        # Jupiter orbit
        jupiter_angle = time * 0.2
        jupiter.center.x = 16 * math.cos(jupiter_angle)
        jupiter.center.z = 16 * math.sin(jupiter_angle)
        
        # Saturn orbit
        saturn_angle = time * 0.15
        saturn.center.x = 22 * math.cos(saturn_angle)
        saturn.center.z = 22 * math.sin(saturn_angle)
        
        # Update rings position to follow Saturn
        rings.center.x = saturn.center.x
        rings.center.z = saturn.center.z
        
        ring_hole.center.x = saturn.center.x
        ring_hole.center.z = saturn.center.z
    
    # Override engine's update method
    engine.update = update_scene
    
    # Run the engine
    engine.run()


if __name__ == "__main__":
    # Choose which demo to run
    # main()  # Basic demo
    demo_solar_system()  # Advanced solar system demo
    
    def transform_mesh(self, mesh):
        world_matrix = Matrix4x4.translation(mesh.position.x, mesh.position.y, mesh.position.z)
        world_matrix = world_matrix * Matrix4x4.rotation_x(mesh.rotation.x)
        world_matrix = world_matrix * Matrix4x4.rotation_y(mesh.rotation.y)
        world_matrix = world_matrix * Matrix4x4.rotation_z(mesh.rotation.z)
        world_matrix = world_matrix * Matrix4x4.scale(mesh.scale.x, mesh.scale.y, mesh.scale.z)
        
        view_proj_matrix = self.view_matrix * self.projection_matrix
        
        # Transform vertices
        for i, vertex in enumerate(mesh.vertices):
            # Apply world transform
            transformed = world_matrix.transform_vector(vertex.position)
            
            # Calculate normal for triangles
            for triangle in mesh.triangles:
                if i in triangle.vertices:
                    a = mesh.vertices[triangle.vertices[0]].position
                    b = mesh.vertices[triangle.vertices[1]].position
                    c = mesh.vertices[triangle.vertices[2]].position
                    
                    edge1 = b - a
                    edge2 = c - a
                    triangle.normal = edge1.cross(edge2).normalize()
                    
                    # Calculate center for depth sorting
                    triangle.center = (a + b + c) * (1/3)
                    triangle.center = world_matrix.transform_vector(triangle.center)
            
            # Apply view and projection
            projected = view_proj_matrix.transform_vector(transformed)
            
            # Store transformed and projected coordinates
            vertex.transformed = transformed
            vertex.z = projected.z
            
            # Convert to screen coordinates
            screen_x = int((projected.x + 1) * self.width / 2)
            screen_y = int((1 - projected.y) * self.height / 2)
            vertex.projected = (screen_x, screen_y)
    
    def render_scene(self):
        """Render all meshes and spheres in the scene"""
        # Clear the screen and Z-buffer
        self.screen.fill((0, 0, 0))
        self.z_buffer = np.full((self.width, self.height), float('inf'))
        
        # Prepare view and projection matrices
        self.view_matrix = self.camera.get_view_matrix()
        self.projection_matrix = self.camera.get_projection_matrix()
        
        # Render all meshes
        for mesh in self.meshes:
            self.render_mesh(mesh)
        
        # Render all spheres
        for sphere in self.spheres:
            self.render_sphere(sphere)
        
        # Update the display
        pygame.display.flip()
    
    def render_mesh(self, mesh):
        if not mesh.is_visible:
            return
        
        # Transform the mesh
        self.transform_mesh(mesh)
        
        # Backface culling and depth sorting for triangles
        visible_triangles = []
        
        for triangle in mesh.triangles:
            if not triangle.is_visible:
                continue
            
            # Check if triangle is facing camera
            view_dir = (self.camera.position - triangle.center).normalize()
            if triangle.normal.dot(view_dir) <= 0:
                continue
            
            # Add to visible triangles for depth sorting
            visible_triangles.append(triangle)
        
        # Sort triangles by depth (painter's algorithm)
        visible_triangles.sort(key=lambda t: (self.camera.position - t.center).length(), reverse=True)
        
        # Draw triangles
        for triangle in visible_triangles:
            vertices = [mesh.vertices[idx] for idx in triangle.vertices]
            points = [v.projected for v in vertices]
            z_values = [v.z for v in vertices]
            
            # Fill triangle if fill mode is enabled and a fill color is specified
            if self.fill_mode and (triangle.fill_color or mesh.fill_color):
                color = triangle.fill_color or mesh.fill_color
                if len(points) == 3:  # Ensure we have 3 points for a triangle
                    if self.occlusion_culling:
                        self.fill_triangle_with_z_buffer(points, z_values, color)
                    else:
                        pygame.draw.polygon(self.screen, color, points)
            
            # Draw wireframe if enabled
            if self.wireframe_mode:
                color = triangle.color or mesh.color
                
                # Draw triangle edges
                for i in range(3):
                    start = points[i]
                    end = points[(i + 1) % 3]
                    start_z = z_values[i]
                    end_z = z_values[(i + 1) % 3]
                    
                    # Check if line is within screen bounds
                    if (0 <= start[0] < self.width and 0 <= start[1] < self.height and
                        0 <= end[0] < self.width and 0 <= end[1] < self.height):
                        if self.occlusion_culling:
                            self.draw_line_with_z_buffer(start, end, start_z, end_z, color)
                        else:
                            pygame.draw.line(self.screen, color, start, end, 1)
        
        # Draw edges (only if not drawing triangles or for special effects)
        if self.wireframe_mode and not mesh.triangles:
            for edge in mesh.edges:
                start_vertex = mesh.vertices[edge.start_idx]
                end_vertex = mesh.vertices[edge.end_idx]
                
                start = start_vertex.projected
                end = end_vertex.projected
                start_z = start_vertex.z
                end_z = end_vertex.z
                
                # Use edge color or mesh color
                color = edge.color or mesh.color
                
                # Check if line is within screen bounds
                if (0 <= start[0] < self.width and 0 <= start[1] < self.height and
                    0 <= end[0] < self.width and 0 <= end[1] < self.height):
                    if self.occlusion_culling:
                        self.draw_line_with_z_buffer(start, end, start_z, end_z, color)
                    else:
                        pygame.draw.line(self.screen, color, start, end, 1)
    
    def fill_triangle_with_z_buffer(self, points, z_values, color):
        """Fill a triangle with z-buffer for hidden surface removal"""
        # Compute bounding box of the triangle
        min_x = max(0, min(p[0] for p in points))
        max_x = min(self.width - 1, max(p[0] for p in points))
        min_y = max(0, min(p[1] for p in points))
        max_y = min(self.height - 1, max(p[1] for p in points))
        
        # Convert points to NumPy arrays for easier calculation
        p0 = np.array([points[0][0], points[0][1]])
        p1 = np.array([points[1][0], points[1][1]])
        p2 = np.array([points[2][0], points[2][1]])
        
        # Precompute edge functions
        edge01 = np.array([p1[1] - p0[1], p0[0] - p1[0]])
        edge12 = np.array([p2[1] - p1[1], p1[0] - p2[0]])
        edge20 = np.array([p0[1] - p2[1], p2[0] - p0[0]])
        
        # Constants for edge functions
        c01 = -np.dot(edge01, p0)
        c12 = -np.dot(edge12, p1)
        c20 = -np.dot(edge20, p2)
        
        # Area of the triangle
        area = 0.5 * (edge01[1] * edge20[0] - edge01[0] * edge20[1])
        
        # Skip degenerate triangles
        if abs(area) < 1e-6:
            return
        
        # Loop over all pixels in the bounding box
        for y in range(int(min_y), int(max_y) + 1):
            for x in range(int(min_x), int(max_x) + 1):
                p = np.array([x, y])
                
                # Evaluate edge functions at this pixel
                w0 = (edge12[0] * (p[0] - p1[0]) + edge12[1] * (p[1] - p1[1])) / (2 * area)
                w1 = (edge20[0] * (p[0] - p2[0]) + edge20[1] * (p[1] - p2[1])) / (2 * area)
                w2 = (edge01[0] * (p[0] - p0[0]) + edge01[1] * (p[1] - p0[1])) / (2 * area)
                
                # Check if point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Interpolate z value
                    z = w0 * z_values[0] + w1 * z_values[1] + w2 * z_values[2]
                    
                    # Check z-buffer - only draw if closer than what's already there
                    if z < self.z_buffer[x, y]:
                        self.z_buffer[x, y] = z
                        self.screen.set_at((x, y), color)
    
    def draw_line_with_z_buffer(self, start, end, start_z, end_z, color):
        """Draw a line with z-buffer for hidden surface removal"""
        # Bresenham's line algorithm with z-buffer
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steep = dy > dx
        
        # If the line is steep, we transpose the coordinates
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            dx, dy = dy, dx
        
        # Ensure we always draw from left to right
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            start_z, end_z = end_z, start_z
        
        # Calculate step direction and error
        step_y = 1 if y0 < y1 else -1
        error = dx // 2
        y = y0
        
        # Calculate z-step for interpolation
        z_step = (end_z - start_z) / dx if dx > 0 else 0
        z = start_z
        
        # Draw the line
        for x in range(x0, x1 + 1):
            # If steep, we transpose back
            actual_x, actual_y = (y, x) if steep else (x, y)
            
            # Check if point is within screen bounds
            if 0 <= actual_x < self.width and 0 <= actual_y < self.height:
                # Check z-buffer
                if z < self.z_buffer[actual_x, actual_y]:
                    self.z_buffer[actual_x, actual_y] = z
                    self.screen.set_at((actual_x, actual_y), color)
            
            # Update error and y coordinate
            error -= dy
            if error < 0:
                y += step_y
                error += dx
            
            # Update z value
            z += z_step