mod bmp;
mod framebuffer;
mod line;
mod obj;
mod triangle;

use std::env;

use minifb::{Key, Window, WindowOptions};
use nalgebra_glm::{Mat4, Vec3, Vec4, rotate};

use bmp::write_bmp_file;
use framebuffer::Framebuffer;
use obj::Obj;
use triangle::triangle;

const SCREEN_WIDTH: usize = 800;
const SCREEN_HEIGHT: usize = 600;

/// Gira el modelo para verlo en tres cuartos y lo escala para que quepa
/// centrado en la pantalla, ocupando cerca del 70% del espacio disponible.
/// Devuelve los vertices ya en coordenadas de pantalla.
fn fit_to_screen(vertices: &[Vec3]) -> Vec<Vec3> {
    let mut rotation = Mat4::identity();
    rotation = rotate(&rotation, 30f32.to_radians(), &Vec3::new(1.0, 0.0, 0.0));
    rotation = rotate(&rotation, (-40f32).to_radians(), &Vec3::new(0.0, 1.0, 0.0));

    let rotated: Vec<Vec3> = vertices
        .iter()
        .map(|v| (rotation * Vec4::new(v.x, v.y, v.z, 1.0)).xyz())
        .collect();

    let mut min = Vec3::repeat(f32::MAX);
    let mut max = Vec3::repeat(f32::MIN);
    for r in &rotated {
        min = min.inf(r);
        max = max.sup(r);
    }

    let center = (min + max) * 0.5;
    let size = max - min;
    let scale = 0.7 * (SCREEN_WIDTH as f32 / size.x).min(SCREEN_HEIGHT as f32 / size.y);

    rotated
        .iter()
        .map(|r| {
            // En pantalla la y crece hacia abajo, por eso se invierte
            Vec3::new(
                SCREEN_WIDTH as f32 / 2.0 + (r.x - center.x) * scale,
                SCREEN_HEIGHT as f32 / 2.0 - (r.y - center.y) * scale,
                r.z,
            )
        })
        .collect()
}

fn render(fb: &mut Framebuffer, model: &Obj, screen_vertices: &[Vec3]) {
    // Recorremos cada cara y tomamos sus indices. Si tiene mas de 3 vertices
    // la partimos en triangulos en abanico: (0,1,2), (0,2,3), (0,3,4)...
    for face in &model.faces {
        let idx = &face.vertex_indices;
        for i in 1..idx.len() - 1 {
            let a = &screen_vertices[idx[0]];
            let b = &screen_vertices[idx[i]];
            let c = &screen_vertices[idx[i + 1]];
            triangle(fb, a, b, c);
        }
    }
}

fn main() {
    // Uso: cargo run --release -- [modelo.obj] [--screenshot archivo.bmp]
    let mut model_path = String::from("models/nave_espacial.obj");
    let mut screenshot_path: Option<String> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--screenshot" {
            screenshot_path = args.next();
        } else {
            model_path = arg;
        }
    }

    let model = match Obj::load(&model_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("No se pudo abrir {}: {}", model_path, e);
            return;
        }
    };
    println!("Modelo cargado: {} vertices, {} caras", model.vertices.len(), model.faces.len());
    let screen_vertices = fit_to_screen(&model.vertices);

    let mut fb = Framebuffer::new(SCREEN_WIDTH, SCREEN_HEIGHT);
    fb.set_background_color((30, 34, 66));

    // Para la imagen del README: dibuja una vez, la guarda y termina
    if let Some(path) = screenshot_path {
        fb.clear();
        fb.set_current_color((255, 255, 0));
        render(&mut fb, &model, &screen_vertices);
        match write_bmp_file(&fb, &path) {
            Ok(()) => println!("Imagen generada: {}", path),
            Err(e) => eprintln!("Error al escribir {}: {}", path, e),
        }
        return;
    }

    let mut window = Window::new(
        "Software Renderer",
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        WindowOptions::default(),
    )
    .expect("No se pudo crear la ventana");
    window.set_target_fps(60);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        fb.clear();

        fb.set_current_color((255, 255, 0)); // pueden cambiar los colores
        render(&mut fb, &model, &screen_vertices);

        window
            .update_with_buffer(&fb.to_u32_buffer(), SCREEN_WIDTH, SCREEN_HEIGHT)
            .expect("No se pudo actualizar la ventana");
    }
}
