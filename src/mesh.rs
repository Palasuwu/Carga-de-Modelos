//! Mallas: vertices + indices de triangulos. Incluye la generacion procedural
//! de esferas (icosfera por subdivision) y del anillo del gigante gaseoso.

use std::collections::HashMap;

use nalgebra_glm::Vec3;

#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
    /// Dato extra por vertice (en la nave: 1 = material emisivo de motor).
    pub extra: f32,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<[u32; 3]>,
    /// Esfera envolvente en espacio local, para descartar el objeto completo.
    pub bound_center: Vec3,
    pub bound_radius: f32,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<[u32; 3]>) -> Mesh {
        let mut min = Vec3::repeat(f32::MAX);
        let mut max = Vec3::repeat(f32::MIN);
        for v in &vertices {
            min = min.inf(&v.position);
            max = max.sup(&v.position);
        }
        let bound_center = (min + max) * 0.5;
        let bound_radius = vertices
            .iter()
            .map(|v| (v.position - bound_center).norm())
            .fold(0.0, f32::max);
        Mesh { vertices, indices, bound_center, bound_radius }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }
}

/// Esfera unitaria generada proceduralmente: se parte de un icosaedro (20
/// triangulos) y cada nivel divide cada triangulo en 4, empujando los puntos
/// medios a la superficie de la esfera. Nivel n => 20 * 4^n triangulos.
pub fn icosphere(subdivisions: u32) -> Mesh {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut points: Vec<Vec3> = [
        (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
        (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
        (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();

    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    for _ in 0..subdivisions {
        // Cache de puntos medios para que triangulos vecinos compartan vertices
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, points: &mut Vec<Vec3>| -> u32 {
            let key = (a.min(b), a.max(b));
            *midpoints.entry(key).or_insert_with(|| {
                points.push(((points[a as usize] + points[b as usize]) * 0.5).normalize());
                (points.len() - 1) as u32
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for &[a, b, c] in &faces {
            let ab = midpoint(a, b, &mut points);
            let bc = midpoint(b, c, &mut points);
            let ca = midpoint(c, a, &mut points);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        faces = next;
    }

    // Todas las caras en sentido antihorario visto desde afuera
    for f in &mut faces {
        let (a, b, c) = (points[f[0] as usize], points[f[1] as usize], points[f[2] as usize]);
        if (b - a).cross(&(c - a)).dot(&(a + b + c)) < 0.0 {
            f.swap(1, 2);
        }
    }

    let vertices = points
        .iter()
        .map(|&p| Vertex { position: p, normal: p, color: Vec3::repeat(1.0), extra: 0.0 })
        .collect();
    Mesh::new(vertices, faces)
}

/// Anillo plano en el plano XZ entre `inner` y `outer` (radios locales).
pub fn ring(inner: f32, outer: f32, segments: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let up = Vec3::new(0.0, 1.0, 0.0);
    for i in 0..=segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        for r in [inner, outer] {
            vertices.push(Vertex { position: Vec3::new(c * r, 0.0, s * r), normal: up, color: Vec3::repeat(1.0), extra: 0.0 });
        }
    }
    for i in 0..segments {
        let k = i * 2;
        indices.push([k, k + 1, k + 3]);
        indices.push([k, k + 3, k + 2]);
    }
    Mesh::new(vertices, indices)
}

/// Recalcula normales suaves sumando las normales de las caras vecinas.
pub fn recompute_normals(mesh: &mut Mesh) {
    for v in &mut mesh.vertices {
        v.normal = Vec3::zeros();
    }
    for &[a, b, c] in &mesh.indices {
        let (pa, pb, pc) = (
            mesh.vertices[a as usize].position,
            mesh.vertices[b as usize].position,
            mesh.vertices[c as usize].position,
        );
        let n = (pb - pa).cross(&(pc - pa));
        for i in [a, b, c] {
            mesh.vertices[i as usize].normal += n;
        }
    }
    for v in &mut mesh.vertices {
        v.normal = v.normal.try_normalize(1e-12).unwrap_or(v.position.normalize());
    }
}

/// Deformacion procedural de la geometria: mueve cada vertice de una
/// icosfera con `f` y recalcula normales y esfera envolvente.
pub fn deformed_sphere(subdivisions: u32, f: impl Fn(Vec3) -> Vec3) -> Mesh {
    let base = icosphere(subdivisions);
    let vertices = base
        .vertices
        .iter()
        .map(|v| Vertex { position: f(v.position), ..*v })
        .collect();
    let mut mesh = Mesh::new(vertices, base.indices);
    recompute_normals(&mut mesh);
    mesh
}

/// Toro (dona) con radio mayor `major` y radio del tubo `minor`.
/// `nu` segmentos alrededor del anillo y `nv` alrededor del tubo.
pub fn torus(major: f32, minor: f32, nu: u32, nv: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..nu {
        let u = i as f32 / nu as f32 * std::f32::consts::TAU;
        let (su, cu) = u.sin_cos();
        for j in 0..nv {
            let v = j as f32 / nv as f32 * std::f32::consts::TAU;
            let (sv, cv) = v.sin_cos();
            let normal = Vec3::new(cu * cv, sv, su * cv);
            let position = Vec3::new(cu * major, 0.0, su * major) + normal * minor;
            vertices.push(Vertex { position, normal, color: Vec3::repeat(1.0), extra: 0.0 });
        }
    }
    for i in 0..nu {
        for j in 0..nv {
            let a = i * nv + j;
            let b = ((i + 1) % nu) * nv + j;
            let c = ((i + 1) % nu) * nv + (j + 1) % nv;
            let d = i * nv + (j + 1) % nv;
            indices.push([a, d, c]);
            indices.push([a, c, b]);
        }
    }
    // Asegura el orden antihorario visto desde afuera del tubo
    let mesh = Mesh::new(vertices, indices);
    let [a, b, c] = mesh.indices[0];
    let (pa, pb, pc) = (mesh.vertices[a as usize].position, mesh.vertices[b as usize].position, mesh.vertices[c as usize].position);
    if (pb - pa).cross(&(pc - pa)).dot(&mesh.vertices[a as usize].normal) < 0.0 {
        let indices = mesh.indices.iter().map(|&[a, b, c]| [a, c, b]).collect();
        return Mesh::new(mesh.vertices, indices);
    }
    mesh
}
