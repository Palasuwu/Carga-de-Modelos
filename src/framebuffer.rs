use nalgebra_glm::Vec3;

/// Buffer de color (formato 0RGB que usa minifb) + buffer de profundidad.
///
/// La profundidad que guardamos es 1/w (w = distancia en el eje de la camara).
/// 1/w se interpola linealmente en pantalla y mantiene buena precision a lo
/// lejos. Mas grande = mas cerca; 0 = infinitamente lejos (valor de limpieza).
pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub color: Vec<u32>,
    pub depth: Vec<f32>,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Framebuffer {
            width,
            height,
            color: vec![0; width * height],
            depth: vec![0.0; width * height],
        }
    }

    pub fn clear(&mut self, color: u32) {
        self.color.fill(color);
        self.depth.fill(0.0);
    }

    #[inline]
    pub fn set(&mut self, x: i32, y: i32, color: u32) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.color[y as usize * self.width + x as usize] = color;
        }
    }

    /// Mezcla `color` sobre el pixel con opacidad `alpha` (0..1).
    #[inline]
    pub fn blend(&mut self, x: i32, y: i32, color: Vec3, alpha: f32) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            let i = y as usize * self.width + x as usize;
            let old = unpack_color(self.color[i]);
            self.color[i] = pack_color(&(old + (color - old) * alpha));
        }
    }

    /// Suma luz al pixel (para brillos como el halo del sol).
    #[inline]
    pub fn add(&mut self, i: usize, light: Vec3) {
        let old = unpack_color(self.color[i]);
        self.color[i] = pack_color(&(old + light));
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let c = self.color[y * self.width + x];
        ((c >> 16) as u8, (c >> 8) as u8, c as u8)
    }
}

/// Color en flotantes (0..1) a 0RGB. Valores > 1 se saturan.
#[inline]
pub fn pack_color(c: &Vec3) -> u32 {
    let r = (c.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let g = (c.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let b = (c.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    (r << 16) | (g << 8) | b
}

#[inline]
pub fn unpack_color(c: u32) -> Vec3 {
    Vec3::new(
        ((c >> 16) & 0xff) as f32 / 255.0,
        ((c >> 8) & 0xff) as f32 / 255.0,
        (c & 0xff) as f32 / 255.0,
    )
}
