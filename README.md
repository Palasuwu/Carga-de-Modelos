# Proyecto final: sistema solar con renderizador por software

Cuatro galaxias, conectadas por agujeros negros, que se exploran con la nave que modelamos en Blender. Todo el 3D (transformaciones, recorte, rasterizado, profundidad y shaders) está hecho a mano en Rust. `minifb` solo abre la ventana, lee el teclado y el mouse y muestra el framebuffer; cumple el mismo papel que raylib en el enunciado.

![Vuelo hacia el sol](render.png)

![Mapa del sistema](mapa.png)

## Ejecutar

```bash
cargo run --release
```

Hay que compilar en modo `--release`: en modo debug el renderizador es unas 20 veces más lento.

## Controles

| Tecla | Acción |
|---|---|
| `W` / `↑` | Acelerar |
| `S` / `↓` | Frenar (y reversa lenta) |
| `A` `D` / `←` `→` | Girar |
| `Espacio` / `E` | Subir |
| `C` / `Q` | Bajar |
| `Shift` + `W` | Turbo |
| `1`–`6`, `0` | Warp a los planetas de la galaxia actual (`0` = su estrella). La ayuda en pantalla (`H`) muestra los de cada galaxia |
| `9` | Warp frente al portal (agujero negro) de la galaxia |
| `M` / `Enter` | Abre o cierra el mapa del sistema. En el mapa, un clic en un planeta o en el espacio lleva la nave ahí |
| `L` | Activa o desactiva los niveles de detalle (LOD) |
| `V` | Activa o desactiva el descarte por frustum |
| `H` | Muestra u oculta los controles en pantalla |
| `Tab` | Muestra u oculta el panel de detalles (FPS y contadores de triángulos) |
| `Esc` | Salir |
| `Ctrl` + `Cmd` + `F` / botón verde | Pantalla completa (la imagen se escala conservando la proporción) |

## Recorrido sugerido para la demo

1. Al iniciar, la nave mira hacia el sol. Acelera con `W` y gira con `A`/`D`.
2. Presiona `2` para hacer warp a Verdalia. Al llegar se activa la **órbita local** y puedes rodear el planeta y su luna.
3. Aléjate con `W` + `Shift`.
4. Abre el mapa con `M` y haz clic en Bandus para ver sus anillos.
5. Presiona `9` para ir frente al portal y vuela hacia el agujero negro: su gravedad te jala, y al cruzarlo viajas a la siguiente galaxia.

## Cómo funciona el pipeline ([src/renderer.rs](src/renderer.rs))

Cada objeto se dibuja con `frame.draw(malla, uniforms, shader, cull)`. `uniforms.model` contiene la matriz del objeto actual. Antes de dibujar el siguiente objeto se cambian esa matriz y el shader activo, que es un puntero a función (`FragmentShader`).

1. **Descarte por frustum:** la esfera envolvente del objeto se prueba contra los 6 planos de la cámara. Si queda fuera, no se procesa ni un triángulo.
2. **Vertex shader:** transforma de modelo a mundo a vista a clip (`projection * view * model`) y prepara los datos que se interpolan (posición de mundo, normal, posición local y color).
3. **Recorte** contra el plano cercano (un triángulo puede convertirse en dos), seguido de la división de perspectiva y la matriz de viewport.
4. **Descarte de caras traseras** y de triángulos fuera de pantalla.
5. **Rasterizado en paralelo:** la pantalla se divide en bandas de 8 filas y los hilos van tomando bandas de una cola, así los núcleos rápidos hacen más trabajo. Para cada píxel se calculan las coordenadas baricéntricas y se hace la prueba de profundidad: el z-buffer guarda 1/w y lo más grande es lo más cercano. Luego se interpolan los datos con corrección de perspectiva y se llama al fragment shader.

Las matrices `look_at`, `perspective` y `viewport` están escritas a mano en [src/math.rs](src/math.rs).

## Escena

- **Sol y planetas** ([src/scene.rs](src/scene.rs)): matriz de modelo = `traslación(órbita) * inclinación * rotación(t) * escala(radio)`. La Luna orbita alrededor de Verdalia porque su posición parte de la de su planeta. Todo depende del tiempo transcurrido.
- **Nave:** se carga desde `models/nave_espacial.obj` con los colores de su `.mtl`, con sombreado plano. Los motores (material `Brillo_Motor`) son emisivos.
- **Cámara en tercera persona:** el objetivo de `look_at` es el centro de la nave y la cámara se mantiene a una distancia constante detrás de ella, siguiendo su orientación con un pequeño retraso.
- **Cielo** ([src/sky.rs](src/sky.rs)): una esfera de radio 4000 que contiene todo el sistema, con un fragment shader sin iluminación. Es nuestro shader de nebulosa (`nebula.frag`) portado a CPU, más dos capas de estrellas que parpadean. La nebulosa es costosa, así que se hornea en un cubemap animado: se mezclan dos cuadros clave mientras el siguiente se calcula unas filas por frame.
- **Efecto de turbo:** mientras se usa el turbo, el shader del cielo cambia. La nebulosa se deforma con ondas que viajan desde el punto de fuga, aparecen estelas de luz que salen del centro (carriles en coordenadas polares que se desplazan con el tiempo), los bordes se tiñen de azul y el FOV se abre un poco.
- **Warp con "pantalla de carga":** al viajar, un iris se cierra sobre la nave con estelas de velocidad y el FOV se abre. Con la pantalla en negro la nave se teletransporta al lado iluminado del destino y el iris se vuelve a abrir. Nunca se cambia de escena.

## Galaxias y portales ([src/galaxy.rs](src/galaxy.rs))

| Galaxia | Estrella | Planetas |
|---|---|---|
| 1. Vía Solar | Sol amarillo | Magma, Verdalia + Luna, Duna, Bandus (anillos), Glacia |
| 2. Nebulosa Violeta | Estrella azul | Cubix (cubo con retícula de neón), Espejo (cromo que refleja el cielo), Toroide (dona de caramelo), Cristal (picos de cristal) |
| 3. Jardín Esmeralda | Estrella verde | Cacahuate (dos lóbulos con selva y ríos), Ojo (un ojo que te sigue y parpadea), Burbuja (película iridiscente), Coral (celdas bioluminiscentes) |
| 4. Abismo Carmesí | Gigante roja | Vacío (planeta hecho con el shader del espacio), Tablero (ajedrez que se tuerce), Erizo (picos incandescentes), Halo (anillo de plasma) |

- **Cielo propio por galaxia:** cada una usa el shader de nebulosa con otra paleta, más nubes de nebulosa horneadas una sola vez.
- **Formas raras:** se generan deformando la icosfera antes de rasterizar: superelipsoide para el cubo, cintura angosta para el cacahuate y conos de Voronoi para los picos. El toro tiene su propio generador. Todas tienen niveles de detalle.
- **Agujero negro** (`black_hole_shader` en [src/exotic.rs](src/exotic.rs)): por cada píxel se traza un rayo. Si cruza el disco de acreción antes de acercarse al centro, se ve el disco. Si pasa por el horizonte, se ve negro. Si no, se dobla hacia el centro y con el rayo doblado se busca la parte de atrás del disco o se lee el cielo deformado (lente gravitacional). El disco gira más rápido por dentro y el lado que se acerca a la cámara brilla más.
- **Portal y carga:** solo hay una galaxia en memoria. Al cruzar el horizonte se lanza un hilo que construye la siguiente galaxia (horneado del cielo y mallas, ~150 ms) mientras se ve el túnel hiperespacial. El túnel dura al menos 1.6 s, así que la carga nunca se nota. Al terminar se reemplaza la galaxia y se libera la anterior. La cuarta galaxia regresa a la primera.
- **El mapa y el warp usan la galaxia actual:** desde una galaxia solo se puede viajar a sus propios planetas.

## Características del menú que solicitamos evaluar

| Característica | Dónde verla |
|---|---|
| Planetas distinguibles (22 cuerpos en 4 galaxias) | Teclas `1`–`6` en cada galaxia |
| Shaders propios: sol, magma, verdalia, duna, bandus, anillos, glacia, luna, nave, cielo, agujero negro y 12 planetas exóticos | [src/shaders.rs](src/shaders.rs), [src/exotic.rs](src/exotic.rs) |
| Deformación de la geometría de un planeta (cubo, cacahuate, picos) | `deformed_sphere` en [src/mesh.rs](src/mesh.rs) |
| Atracción gravitatoria que modifica la trayectoria de la nave | Acércate al portal; `apply_portal_gravity` en [src/app.rs](src/app.rs) |
| Shaders animados: plasma del sol, lava que pulsa, nubes de Verdalia, bandas de Bandus, nebulosa, estrellas y estelas del turbo | Cualquier vista; `Shift` + `W` para el turbo |
| Etapa propia de generación/subdivisión de geometría (icosfera) | `icosphere` en [src/mesh.rs](src/mesh.rs) |
| Modo de control local (la nave acompaña al planeta en su órbita) | HUD: "ORBITA LOCAL" |
| Aceleración, frenado y giro progresivos e independientes de los FPS | [src/scene.rs](src/scene.rs), `Ship::update` |
| Colisión básica con planetas y sol | Intenta atravesar un planeta |
| Descarte de objetos fuera de la vista | Contadores del HUD; compara con `V` |
| Niveles de detalle según la distancia (20 a 20 480 triángulos) | Contadores del HUD; compara con `L` |

El panel de detalles (`Tab`) muestra los triángulos enviados, los descartados por frustum, las caras traseras, los fuera de pantalla, los rasterizados y los fragmentos sombreados.

## Parámetros fáciles de cambiar en vivo

- Radio, órbita, velocidades y parámetros de shader de cada cuerpo: `solar_system()` en [src/scene.rs](src/scene.rs). Por ejemplo, `params[0]` del sol es la velocidad de su animación y `params[0]` de Magma es la intensidad de la lava.
- Semillas del ruido: el último argumento de `fbm`, `perlin` y `voronoi` en [src/shaders.rs](src/shaders.rs).
- Física de la nave: las constantes `THRUST`, `DRAG`, `TURN_MAX`, etc. en [src/scene.rs](src/scene.rs).
- Velocidad de la nebulosa: `NEBULA_SPEED` en [src/sky.rs](src/sky.rs).

## Capturas y pruebas sin ventana

```bash
# Imagen después de hacer warp a Bandus (tecla 4) en t = 20 s
cargo run --release -- --screenshot shot.bmp --time 20 --warp 4
# Mapa del sistema
cargo run --release -- --screenshot mapa.bmp --map
# Otra galaxia (0 a 3) y la animacion del portal
cargo run --release -- --screenshot g3.bmp --galaxy 2 --warp 2
cargo run --release -- --screenshot tunel.bmp --jump --simulate 1.5
# Medir el tiempo por frame
cargo run --release -- --bench 120
```

## Recursos externos y uso de IA

- [minifb](https://crates.io/crates/minifb) para la ventana y la entrada, y [nalgebra-glm](https://crates.io/crates/nalgebra-glm) para los vectores y matrices.
- El shader de nebulosa está basado en un shader inspirado en Balatro de un proyecto personal anterior (`myShader.js`), portado a Rust.

Claude para ver errores the phasing entre el planeta y la nave 
