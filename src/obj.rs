use std::fs;
use std::io;

use nalgebra_glm::Vec3;

/// Una cara guarda los indices (base 0) de sus vertices. Puede tener 3 o mas.
pub struct Face {
    pub vertex_indices: Vec<usize>,
}

pub struct Obj {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<Face>,
}

impl Obj {
    /// Lee las lineas "v" y "f" de un archivo .obj; lo demas (normales,
    /// materiales, grupos) no hace falta para dibujar los triangulos.
    pub fn load(path: &str) -> io::Result<Obj> {
        let text = fs::read_to_string(path)?;
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for line in text.lines() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let coords: Vec<f32> = parts.take(3).filter_map(|p| p.parse().ok()).collect();
                    if coords.len() == 3 {
                        vertices.push(Vec3::new(coords[0], coords[1], coords[2]));
                    }
                }
                Some("f") => {
                    let mut vertex_indices = Vec::new();
                    // Cada token puede ser "v", "v/vt", "v//vn" o "v/vt/vn"; solo nos importa v
                    for token in parts {
                        let first = token.split('/').next().unwrap_or("");
                        if let Ok(index) = first.parse::<i64>() {
                            // Los indices del .obj empiezan en 1; los negativos cuentan desde el final
                            let index = if index > 0 { index - 1 } else { vertices.len() as i64 + index };
                            vertex_indices.push(index as usize);
                        }
                    }
                    if vertex_indices.len() >= 3 {
                        faces.push(Face { vertex_indices });
                    }
                }
                _ => {}
            }
        }

        Ok(Obj { vertices, faces })
    }
}
