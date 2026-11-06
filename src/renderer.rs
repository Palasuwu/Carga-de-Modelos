//! Pipeline de graficas por software.
//!
//! Por cada modelo (`Frame::draw`):
//!   1. Descarte por frustum de la esfera envolvente del objeto.
//!   2. Vertex shader: modelo -> mundo -> vista -> clip (uniforms.model/view/projection).
//!   3. Ensamblado de triangulos y recorte contra el plano cercano.
//!   4. Division de perspectiva + viewport -> coordenadas de pantalla.
//!   5. Descarte de caras traseras y de triangulos fuera de pantalla.
//! Los triangulos resultantes se guardan junto con el indice de su "draw call"
//! (sus uniforms y su fragment shader). Al final (`Frame::finish`) se rasterizan:
//!   6. La pantalla se divide en bandas horizontales y los hilos van tomando
//!      bandas de una cola: coordenadas baricentricas, prueba de profundidad (z-buffer),
//!      interpolacion con correccion de perspectiva y fragment shader.
//! Cada banda escribe solo en sus propias filas, asi que no hay conflictos.

use std::sync::Mutex;
use std::thread;

use nalgebra_glm::{Mat3, Mat4, Vec3, Vec4};

use crate::framebuffer::{Framebuffer, pack_color};
use crate::mesh::{Mesh, Vertex};
use crate::sky::SkyBox;

/// Cantidad de "varyings" que viajan del vertex al fragment shader:
/// posicion de mundo (3), normal (3), posicion local (3), color (3), extra (1).
const NV: usize = 13;
const BAND_HEIGHT: usize = 8;

/// Variables globales del shader para el objeto que se esta dibujando.
#[derive(Clone, Copy)]
pub struct Uniforms<'a> {
    pub model: Mat4,
    pub normal_matrix: Mat3,
    pub view: Mat4,
    pub projection: Mat4,
    pub viewport: Mat4,
    pub time: f32,
    pub camera_pos: Vec3,
    pub light_pos: Vec3,
    /// Parametros libres por objeto (intensidad, radio, empuje...).
    pub params: [f32; 4],
    pub sky: Option<&'a SkyBox>,
}

impl<'a> Uniforms<'a> {
    /// Cambia la matriz de modelo "actual" y recalcula la matriz de normales.
    pub fn with_model(mut self, model: Mat4) -> Self {
        self.model = model;
        let m3 = Mat3::new(
            model[(0, 0)], model[(0, 1)], model[(0, 2)],
            model[(1, 0)], model[(1, 1)], model[(1, 2)],
            model[(2, 0)], model[(2, 1)], model[(2, 2)],
        );
        self.normal_matrix = m3.try_inverse().map(|m| m.transpose()).unwrap_or(m3);
        self
    }

    pub fn with_params(mut self, params: [f32; 4]) -> Self {
        self.params = params;
        self
    }
}

/// Lo que recibe el fragment shader por cada pixel cubierto.
pub struct Fragment {
    pub world_pos: Vec3,
    pub normal: Vec3,
    /// Posicion en el espacio del modelo: sirve para que los patrones
    /// procedurales queden "pegados" a la superficie aunque el planeta rote.
    pub local_pos: Vec3,
    pub color: Vec3,
    pub extra: f32,
}

/// Un fragment shader es un puntero a funcion: el shader "activo" se cambia
/// simplemente pasando otra funcion. Devolver None descarta el fragmento.
pub type FragmentShader = fn(&Fragment, &Uniforms) -> Option<Vec3>;

#[derive(Clone, Copy, PartialEq)]
pub enum CullMode {
    Back,
    None,
}

#[derive(Default, Clone, Copy)]
pub struct RenderStats {
    pub objects: u32,
    pub objects_culled: u32,
    pub tris_submitted: u32,
    pub tris_culled_frustum: u32,
    pub tris_backface: u32,
    pub tris_clipped: u32,
    pub tris_raster: u32,
    pub fragments: u64,
}

struct DrawCall<'a> {
    uniforms: Uniforms<'a>,
    shader: FragmentShader,
}

#[derive(Clone, Copy)]
struct ClipVertex {
    pos: Vec4,
    var: [f32; NV],
}

struct ScreenVertex {
    x: f32,
    y: f32,
    inv_w: f32,
    /// varyings ya multiplicados por 1/w (para la correccion de perspectiva)
    var: [f32; NV],
}

struct RasterTri {
    x: [f32; 3],
    y: [f32; 3],
    inv_w: [f32; 3],
    var: [[f32; NV]; 3],
    call: u32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

pub struct Renderer {
    width: usize,
    height: usize,
    tris: Vec<RasterTri>,
    clip_verts: Vec<ClipVertex>,
    bins: Vec<Vec<u32>>,
    threads: usize,
    pub stats: RenderStats,
    pub frustum_culling: bool,
}

pub struct Frame<'r, 'a> {
    renderer: &'r mut Renderer,
    calls: Vec<DrawCall<'a>>,
}

/// Vertex shader: transforma un vertice del modelo a clip space y prepara
/// los datos que se interpolaran sobre el triangulo.
fn vertex_shader(v: &Vertex, u: &Uniforms, view_proj: &Mat4) -> ClipVertex {
    let world = u.model * Vec4::new(v.position.x, v.position.y, v.position.z, 1.0);
    let clip = view_proj * world;
    let n = u.normal_matrix * v.normal;
    ClipVertex {
        pos: clip,
        var: [
            world.x, world.y, world.z,
            n.x, n.y, n.z,
            v.position.x, v.position.y, v.position.z,
            v.color.x, v.color.y, v.color.z,
            v.extra,
        ],
    }
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        Renderer {
            width,
            height,
            tris: Vec::new(),
            clip_verts: Vec::new(),
            bins: (0..height.div_ceil(BAND_HEIGHT)).map(|_| Vec::new()).collect(),
            threads,
            stats: RenderStats::default(),
            frustum_culling: true,
        }
    }

    pub fn begin<'r, 'a>(&'r mut self) -> Frame<'r, 'a> {
        self.tris.clear();
        self.stats = RenderStats::default();
        Frame { renderer: self, calls: Vec::new() }
    }
}

/// Prueba la esfera envolvente del modelo contra los 6 planos del frustum
/// (en espacio de vista). Si queda completamente fuera, no se procesa nada.
fn sphere_visible(mesh: &Mesh, u: &Uniforms) -> bool {
    let c = mesh.bound_center;
    let cv = u.view * u.model * Vec4::new(c.x, c.y, c.z, 1.0);
    let m = &u.model;
    let scale = (0..3)
        .map(|j| Vec3::new(m[(0, j)], m[(1, j)], m[(2, j)]).norm())
        .fold(0.0, f32::max);
    let r = mesh.bound_radius * scale;

    let p = &u.projection;
    let tan_x = 1.0 / p[(0, 0)];
    let tan_y = 1.0 / p[(1, 1)];
    let near = p[(2, 3)] / (p[(2, 2)] - 1.0);
    let far = p[(2, 3)] / (p[(2, 2)] + 1.0);

    if cv.z - r > -near || cv.z + r < -far {
        return false;
    }
    let nx = (1.0 + tan_x * tan_x).sqrt();
    let ny = (1.0 + tan_y * tan_y).sqrt();
    if (cv.x + cv.z * tan_x) / nx > r || (-cv.x + cv.z * tan_x) / nx > r {
        return false;
    }
    if (cv.y + cv.z * tan_y) / ny > r || (-cv.y + cv.z * tan_y) / ny > r {
        return false;
    }
    true
}

/// Recorta el triangulo contra el plano cercano (z = -w). Puede devolver
/// 0, 3 o 4 vertices (un cuadrilatero que luego se parte en 2 triangulos).
fn clip_near(tri: &[ClipVertex; 3], out: &mut [ClipVertex; 4]) -> usize {
    let mut n = 0;
    for i in 0..3 {
        let a = &tri[i];
        let b = &tri[(i + 1) % 3];
        let da = a.pos.z + a.pos.w;
        let db = b.pos.z + b.pos.w;
        if da >= 0.0 {
            out[n] = *a;
            n += 1;
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            let mut v = *a;
            v.pos = a.pos + (b.pos - a.pos) * t;
            for k in 0..NV {
                v.var[k] = a.var[k] + (b.var[k] - a.var[k]) * t;
            }
            out[n] = v;
            n += 1;
        }
    }
    n
}

/// true si los tres vertices estan del lado de afuera de un mismo plano.
fn trivially_outside(t: &[ClipVertex; 3]) -> bool {
    let all = |f: &dyn Fn(&Vec4) -> bool| t.iter().all(|v| f(&v.pos));
    all(&|p| p.x > p.w)
        || all(&|p| p.x < -p.w)
        || all(&|p| p.y > p.w)
        || all(&|p| p.y < -p.w)
        || all(&|p| p.z > p.w)
        || all(&|p| p.z < -p.w)
}

#[inline]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

impl<'r, 'a> Frame<'r, 'a> {
    /// Pasa un modelo por la parte geometrica del pipeline con los uniforms
    /// y el fragment shader dados. Devuelve false si el objeto fue descartado.
    pub fn draw(&mut self, mesh: &Mesh, uniforms: Uniforms<'a>, shader: FragmentShader, cull: CullMode) -> bool {
        let r = &mut *self.renderer;
        let tri_count = mesh.indices.len() as u32;
        r.stats.objects += 1;
        r.stats.tris_submitted += tri_count;

        if r.frustum_culling && !sphere_visible(mesh, &uniforms) {
            r.stats.objects_culled += 1;
            r.stats.tris_culled_frustum += tri_count;
            return false;
        }

        let call = self.calls.len() as u32;
        self.calls.push(DrawCall { uniforms, shader });
        let u = &self.calls[call as usize].uniforms;

        // Etapa de vertices
        let view_proj = u.projection * u.view;
        r.clip_verts.clear();
        r.clip_verts.extend(mesh.vertices.iter().map(|v| vertex_shader(v, u, &view_proj)));

        let (w, h) = (r.width as i32, r.height as i32);
        let mut poly = [r.clip_verts[0]; 4];
        for idx in &mesh.indices {
            let tri = [
                r.clip_verts[idx[0] as usize],
                r.clip_verts[idx[1] as usize],
                r.clip_verts[idx[2] as usize],
            ];
            if trivially_outside(&tri) {
                r.stats.tris_clipped += 1;
                continue;
            }
            let n = clip_near(&tri, &mut poly);
            if n < 3 {
                r.stats.tris_clipped += 1;
                continue;
            }

            // Division de perspectiva + viewport
            let sv: Vec<ScreenVertex> = poly[..n]
                .iter()
                .map(|cv| {
                    let inv_w = 1.0 / cv.pos.w;
                    let ndc = Vec4::new(cv.pos.x * inv_w, cv.pos.y * inv_w, cv.pos.z * inv_w, 1.0);
                    let s = u.viewport * ndc;
                    let mut var = cv.var;
                    for v in &mut var {
                        *v *= inv_w;
                    }
                    ScreenVertex { x: s.x, y: s.y, inv_w, var }
                })
                .collect();

            for k in 1..n - 1 {
                let (a, b, c) = (&sv[0], &sv[k], &sv[k + 1]);
                let area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
                if area.abs() < 1e-9 {
                    continue;
                }
                // En pantalla (y hacia abajo) las caras frontales quedan con area negativa
                if cull == CullMode::Back && area > 0.0 {
                    r.stats.tris_backface += 1;
                    continue;
                }
                let min_x = ((a.x.min(b.x).min(c.x) - 0.5).ceil() as i32).max(0);
                let max_x = ((a.x.max(b.x).max(c.x) - 0.5).floor() as i32).min(w - 1);
                let min_y = ((a.y.min(b.y).min(c.y) - 0.5).ceil() as i32).max(0);
                let max_y = ((a.y.max(b.y).max(c.y) - 0.5).floor() as i32).min(h - 1);
                if min_x > max_x || min_y > max_y {
                    r.stats.tris_clipped += 1;
                    continue;
                }
                r.stats.tris_raster += 1;
                r.tris.push(RasterTri {
                    x: [a.x, b.x, c.x],
                    y: [a.y, b.y, c.y],
                    inv_w: [a.inv_w, b.inv_w, c.inv_w],
                    var: [a.var, b.var, c.var],
                    call,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                });
            }
        }
        true
    }

    /// Rasteriza todo lo encolado, en paralelo por bandas de filas.
    pub fn finish(self, fb: &mut Framebuffer) -> RenderStats {
        let r = self.renderer;
        let calls = &self.calls;
        for bin in &mut r.bins {
            bin.clear();
        }
        for (i, t) in r.tris.iter().enumerate() {
            for b in (t.min_y as usize / BAND_HEIGHT)..=(t.max_y as usize / BAND_HEIGHT) {
                r.bins[b].push(i as u32);
            }
        }

        let width = r.width;
        let tris = &r.tris;
        let bins = &r.bins;
        let threads = r.threads.max(1);

        // Cola de bandas: cada hilo toma la siguiente banda libre. Asi los
        // nucleos rapidos procesan mas bandas y nadie espera a los lentos
        // (los Mac mezclan nucleos de rendimiento y de eficiencia).
        // Se recorren de abajo hacia arriba con pop(), el orden no importa.
        let queue: Vec<(usize, &mut [u32], &mut [f32])> = fb
            .color
            .chunks_mut(BAND_HEIGHT * width)
            .zip(fb.depth.chunks_mut(BAND_HEIGHT * width))
            .enumerate()
            .filter(|(b, _)| !bins[*b].is_empty())
            .map(|(b, (color, depth))| (b, color, depth))
            .collect();
        let queue = Mutex::new(queue);

        let fragments: u64 = thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let queue = &queue;
                    s.spawn(move || {
                        let mut count = 0;
                        loop {
                            let next = queue.lock().unwrap().pop();
                            let Some((b, color, depth)) = next else { break };
                            count += raster_band(tris, &bins[b], calls, b * BAND_HEIGHT, width, color, depth);
                        }
                        count
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });

        r.stats.fragments = fragments;
        r.stats
    }
}

fn raster_band(
    tris: &[RasterTri],
    bin: &[u32],
    calls: &[DrawCall],
    y0: usize,
    width: usize,
    color: &mut [u32],
    depth: &mut [f32],
) -> u64 {
    const EPS: f32 = -1e-5;
    let rows = color.len() / width;
    let (band_start, band_end) = (y0 as i32, (y0 + rows) as i32 - 1);
    let mut count = 0;
    let mut var = [0.0f32; NV];

    for &ti in bin {
        let t = &tris[ti as usize];
        let ys = t.min_y.max(band_start);
        let ye = t.max_y.min(band_end);
        if ys > ye {
            continue;
        }
        let call = &calls[t.call as usize];
        let [ax, bx, cx] = t.x;
        let [ay, by, cy] = t.y;
        let inv_area = 1.0 / edge(ax, ay, bx, by, cx, cy);
        // Funciones de borde normalizadas: w_i = A_i * px + B_i * py + C_i
        let (a0, b0) = ((by - cy) * inv_area, (cx - bx) * inv_area);
        let (a1, b1) = ((cy - ay) * inv_area, (ax - cx) * inv_area);
        let c0 = edge(bx, by, cx, cy, 0.0, 0.0) * inv_area;
        let c1 = edge(cx, cy, ax, ay, 0.0, 0.0) * inv_area;
        let [iw0, iw1, iw2] = t.inv_w;

        for y in ys..=ye {
            let py = y as f32 + 0.5;
            let px0 = t.min_x as f32 + 0.5;
            let mut w0 = a0 * px0 + b0 * py + c0;
            let mut w1 = a1 * px0 + b1 * py + c1;
            let row = (y - band_start) as usize * width;
            for x in t.min_x..=t.max_x {
                let w2 = 1.0 - w0 - w1;
                if w0 >= EPS && w1 >= EPS && w2 >= EPS {
                    // 1/w se interpola linealmente en pantalla: es nuestra profundidad
                    let z = w0 * iw0 + w1 * iw1 + w2 * iw2;
                    let di = row + x as usize;
                    if z > depth[di] {
                        let inv_z = 1.0 / z;
                        for k in 0..NV {
                            var[k] = (w0 * t.var[0][k] + w1 * t.var[1][k] + w2 * t.var[2][k]) * inv_z;
                        }
                        let n = Vec3::new(var[3], var[4], var[5]);
                        let frag = Fragment {
                            world_pos: Vec3::new(var[0], var[1], var[2]),
                            normal: n / n.norm().max(1e-8),
                            local_pos: Vec3::new(var[6], var[7], var[8]),
                            color: Vec3::new(var[9], var[10], var[11]),
                            extra: var[12],
                        };
                        if let Some(c) = (call.shader)(&frag, &call.uniforms) {
                            color[di] = pack_color(&c);
                            depth[di] = z;
                            count += 1;
                        }
                    }
                }
                w0 += a0;
                w1 += a1;
            }
        }
    }
    count
}
