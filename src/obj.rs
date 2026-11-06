use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use nalgebra_glm::Vec3;

use crate::mesh::{Mesh, Vertex};

struct Material {
    diffuse: Vec3,
    emissive: bool,
}

/// Carga un .obj (lineas v, f, usemtl, mtllib) y lo convierte en una malla
/// con sombreado plano: cada triangulo lleva su propia normal y el color
/// difuso (Kd) de su material. El modelo se recentra en el origen.
pub fn load_obj(path: &str) -> io::Result<Mesh> {
    let text = fs::read_to_string(path)?;
    let dir = Path::new(path).parent().unwrap_or(Path::new("."));

    let mut materials: HashMap<String, Material> = HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut current = (Vec3::repeat(0.8), false);
    let mut faces: Vec<(Vec<usize>, Vec3, bool)> = Vec::new();

    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("mtllib") => {
                if let Some(name) = parts.next() {
                    // Si falta el .mtl el modelo se dibuja gris, no es un error fatal
                    if let Ok(m) = load_mtl(&dir.join(name)) {
                        materials.extend(m);
                    }
                }
            }
            Some("usemtl") => {
                let name = parts.next().unwrap_or("");
                current = match materials.get(name) {
                    Some(m) => (m.diffuse, m.emissive),
                    None => (Vec3::repeat(0.8), false),
                };
            }
            Some("v") => {
                let c: Vec<f32> = parts.take(3).filter_map(|p| p.parse().ok()).collect();
                if c.len() == 3 {
                    positions.push(Vec3::new(c[0], c[1], c[2]));
                }
            }
            Some("f") => {
                let mut idx = Vec::new();
                // Cada token puede ser "v", "v/vt", "v//vn" o "v/vt/vn"; solo nos importa v
                for token in parts {
                    if let Ok(i) = token.split('/').next().unwrap_or("").parse::<i64>() {
                        // Los indices del .obj empiezan en 1; los negativos cuentan desde el final
                        let i = if i > 0 { i - 1 } else { positions.len() as i64 + i };
                        idx.push(i as usize);
                    }
                }
                if idx.len() >= 3 {
                    faces.push((idx, current.0, current.1));
                }
            }
            _ => {}
        }
    }

    let mut min = Vec3::repeat(f32::MAX);
    let mut max = Vec3::repeat(f32::MIN);
    for p in &positions {
        min = min.inf(p);
        max = max.sup(p);
    }
    let center = (min + max) * 0.5;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (idx, color, emissive) in &faces {
        // Caras de 4+ vertices se parten en abanico: (0,1,2), (0,2,3)...
        for i in 1..idx.len() - 1 {
            let a = positions[idx[0]] - center;
            let b = positions[idx[i]] - center;
            let c = positions[idx[i + 1]] - center;
            let n = (b - a).cross(&(c - a));
            if n.norm_squared() < 1e-12 {
                continue; // triangulo degenerado
            }
            let n = n.normalize();
            let base = vertices.len() as u32;
            for p in [a, b, c] {
                vertices.push(Vertex { position: p, normal: n, color: *color, extra: if *emissive { 1.0 } else { 0.0 } });
            }
            indices.push([base, base + 1, base + 2]);
        }
    }

    Ok(Mesh::new(vertices, indices))
}

fn load_mtl(path: &Path) -> io::Result<HashMap<String, Material>> {
    let text = fs::read_to_string(path)?;
    let mut out = HashMap::new();
    let mut name = String::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("newmtl") => {
                name = parts.next().unwrap_or("").to_string();
                // Los materiales de "Brillo" (motores) se marcan como emisivos
                let emissive = name.to_lowercase().contains("brillo");
                out.insert(name.clone(), Material { diffuse: Vec3::repeat(0.8), emissive });
            }
            Some("Kd") => {
                let c: Vec<f32> = parts.take(3).filter_map(|p| p.parse().ok()).collect();
                if let (Some(m), 3) = (out.get_mut(&name), c.len()) {
                    m.diffuse = Vec3::new(c[0], c[1], c[2]);
                }
            }
            Some("Ke") => {
                let c: Vec<f32> = parts.take(3).filter_map(|p| p.parse().ok()).collect();
                if let Some(m) = out.get_mut(&name) {
                    if c.iter().any(|&v| v > 0.0) {
                        m.emissive = true;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}
