mod app;
mod bmp;
mod framebuffer;
mod hud;
mod line;
mod math;
mod mesh;
mod noise;
mod obj;
mod renderer;
mod scene;
mod shaders;
mod sky;

use std::env;
use std::time::Instant;

use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

use app::{Action, App, HEIGHT, Input, WIDTH};
use bmp::write_bmp_file;
use framebuffer::Framebuffer;
use obj::load_obj;
use scene::ShipInput;

struct Options {
    model_path: String,
    screenshot: Option<String>,
    time: f32,
    warp: Option<u8>,
    map: bool,
    simulate: f32,
    bench: Option<u32>,
    fly: f32,
    boost: bool,
}

fn parse_args() -> Options {
    let mut o = Options {
        model_path: "models/nave_espacial.obj".into(),
        screenshot: None,
        time: 0.0,
        warp: None,
        map: false,
        simulate: 2.0,
        bench: None,
        fly: 0.0,
        boost: false,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut num = || args.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
        match arg.as_str() {
            "--screenshot" => o.screenshot = args.next(),
            "--time" => o.time = num(),
            "--warp" => o.warp = Some(num() as u8),
            "--map" => o.map = true,
            "--simulate" => o.simulate = num(),
            "--bench" => o.bench = Some(num() as u32),
            "--fly" => o.fly = num(),
            "--boost" => o.boost = true,
            _ => o.model_path = arg,
        }
    }
    o
}

/// Modo sin ventana: simula unos segundos y guarda una imagen (o mide tiempos).
/// Sirve para generar las capturas del README.
fn run_headless(app: &mut App, fb: &mut Framebuffer, o: &Options) {
    let dt = 1.0 / 60.0;
    app.time = o.time;
    let mut first = Input::default();
    if let Some(k) = o.warp {
        first.actions.push(Action::Warp(k));
    }
    if o.map {
        first.actions.push(Action::ToggleMap);
    }
    app.update(dt, &first);
    let frames = (o.simulate / dt) as u32;
    for f in 0..frames {
        let mut input = Input::default();
        input.ship.forward = (f as f32 * dt) < o.fly;
        input.ship.boost = o.boost;
        app.update(dt, &input);
    }

    if let Some(n) = o.bench {
        let start = Instant::now();
        let mut input = Input::default();
        input.ship.forward = o.fly > 0.0;
        input.ship.boost = o.boost;
        for _ in 0..n {
            app.update(dt, &input);
            app.render(fb);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("{} frames, {:.2} ms por frame ({:.0} FPS)", n, ms, 1000.0 / ms);
    }

    app.render(fb);
    if let Some(path) = &o.screenshot {
        match write_bmp_file(fb, path) {
            Ok(()) => println!("Imagen generada: {}", path),
            Err(e) => eprintln!("Error al escribir {}: {}", path, e),
        }
    }
}

fn main() {
    let options = parse_args();
    let ship = match load_obj(&options.model_path) {
        Ok(mesh) => mesh,
        Err(e) => {
            eprintln!("No se pudo abrir {}: {}", options.model_path, e);
            return;
        }
    };
    println!("Nave cargada: {} triangulos", ship.triangle_count());

    let mut app = App::new(ship);
    let mut fb = Framebuffer::new(WIDTH, HEIGHT);

    if options.screenshot.is_some() || options.bench.is_some() {
        run_headless(&mut app, &mut fb, &options);
        return;
    }

    let mut window = Window::new("Sistema Solar - Software Renderer", WIDTH, HEIGHT, WindowOptions::default())
        .expect("No se pudo crear la ventana");
    window.set_target_fps(60);

    let mut last = Instant::now();
    let mut was_down = false;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;

        let down = |k: &[Key]| k.iter().any(|&key| window.is_key_down(key));
        let mut input = Input {
            ship: ShipInput {
                forward: down(&[Key::W, Key::Up]),
                back: down(&[Key::S, Key::Down]),
                left: down(&[Key::A, Key::Left]),
                right: down(&[Key::D, Key::Right]),
                up: down(&[Key::Space, Key::E]),
                down: down(&[Key::C, Key::Q]),
                boost: down(&[Key::LeftShift, Key::RightShift]),
            },
            ..Default::default()
        };
        let warp_keys = [
            (Key::Key0, 0), (Key::Key1, 1), (Key::Key2, 2), (Key::Key3, 3),
            (Key::Key4, 4), (Key::Key5, 5), (Key::Key6, 6),
        ];
        for (key, n) in warp_keys {
            if window.is_key_pressed(key, KeyRepeat::No) {
                input.actions.push(Action::Warp(n));
            }
        }
        let toggles: [(Key, fn() -> Action); 6] = [
            (Key::M, || Action::ToggleMap),
            (Key::Enter, || Action::ToggleMap),
            (Key::H, || Action::ToggleHelp),
            (Key::Tab, || Action::ToggleStats),
            (Key::L, || Action::ToggleLod),
            (Key::V, || Action::ToggleCulling),
        ];
        for (key, make) in toggles {
            if window.is_key_pressed(key, KeyRepeat::No) {
                input.actions.push(make());
            }
        }
        input.mouse = window.get_mouse_pos(MouseMode::Discard);
        let is_down = window.get_mouse_down(MouseButton::Left);
        input.click = is_down && !was_down;
        was_down = is_down;

        app.update(dt, &input);
        app.render(&mut fb);
        window
            .update_with_buffer(&fb.color, WIDTH, HEIGHT)
            .expect("No se pudo actualizar la ventana");
    }
}
