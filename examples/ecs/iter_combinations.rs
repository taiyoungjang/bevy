//! Shows how to iterate over combinations of query results.

use bevy::{pbr::AmbientLight, prelude::*, time::FixedTimestep};
use rand::{thread_rng, Rng};

#[derive(Debug, Hash, PartialEq, Eq, Clone, StageLabel)]
struct FixedUpdateStage;

const DELTA_TIME: f64 = 0.01;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(AmbientLight {
            brightness: 0.03,
            ..default()
        })
        .add_startup_system(generate_bodies)
        .add_stage_after(
            CoreStage::Update,
            FixedUpdateStage,
            SystemStage::parallel()
                .with_run_criteria(FixedTimestep::step(DELTA_TIME))
                .with_system(interact_bodies)
                .with_system(integrate),
        )
        .add_system(look_at_star)
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}

const GRAVITY_CONSTANT: f64 = 0.001;
const NUM_BODIES: usize = 100;

#[derive(Component, Default)]
struct Mass(f64);
#[derive(Component, Default)]
struct Acceleration(DVec3);
#[derive(Component, Default)]
struct LastPos(DVec3);
#[derive(Component)]
struct Star;

#[derive(Bundle, Default)]
struct BodyBundle {
    pbr: PbrBundle,
    mass: Mass,
    last_pos: LastPos,
    acceleration: Acceleration,
}

fn generate_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(
        Mesh::try_from(shape::Icosphere {
            radius: 1.0,
            subdivisions: 3,
        })
        .unwrap(),
    );

    let color_range = 0.5..1.0;
    let vel_range = -0.5..0.5;

    let mut rng = thread_rng();
    for _ in 0..NUM_BODIES {
        let radius: f64 = rng.gen_range(0.1..0.7);
        let mass_value = radius.powi(3) * 10.;

        let position =DVec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        )
        .normalize()
            * rng.gen_range(0.2f64..1.0).cbrt()
            * 15.;

        commands.spawn(BodyBundle {
            pbr: PbrBundle {
                transform: Transform {
                    translation: position,
                    scale: DVec3::splat(radius),
                    ..default()
                },
                mesh: mesh.clone(),
                material: materials.add(
                    Color::rgb(
                        rng.gen_range(color_range.clone()),
                        rng.gen_range(color_range.clone()),
                        rng.gen_range(color_range.clone()),
                    )
                    .into(),
                ),
                ..default()
            },
            mass: Mass(mass_value),
            acceleration: Acceleration(DVec3::ZERO),
            last_pos: LastPos(
                position
                    - DVec3::new(
                        rng.gen_range(vel_range.clone()),
                        rng.gen_range(vel_range.clone()),
                        rng.gen_range(vel_range.clone()),
                    ) * DELTA_TIME,
            ),
        });
    }

    // add bigger "star" body in the center
    let star_radius = 1.;
    commands
        .spawn((
            BodyBundle {
                pbr: PbrBundle {
                    transform: Transform::from_scale(DVec3::splat(star_radius)),
                    mesh: meshes.add(
                        Mesh::try_from(shape::Icosphere {
                            radius: 1.0,
                            subdivisions: 5,
                        })
                        .unwrap(),
                    ),
                    material: materials.add(StandardMaterial {
                        base_color: Color::ORANGE_RED,
                        emissive: (Color::ORANGE_RED * 2.),
                        ..default()
                    }),
                    ..default()
                },
                mass: Mass(500.0),
                ..default()
            },
            Star,
        ))
        .with_children(|p| {
            p.spawn(PointLightBundle {
                point_light: PointLight {
                    color: Color::WHITE,
                    intensity: 400.0,
                    range: 100.0,
                    radius: star_radius as f32,
                    ..default()
                },
                ..default()
            });
        });
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 10.5, -30.0).looking_at(DVec3::ZERO, DVec3::Y),
        ..default()
    });
}

fn interact_bodies(mut query: Query<(&Mass, &GlobalTransform, &mut Acceleration)>) {
    let mut iter = query.iter_combinations_mut();
    while let Some([(Mass(m1), transform1, mut acc1), (Mass(m2), transform2, mut acc2)]) =
        iter.fetch_next()
    {
        let delta = transform2.translation() - transform1.translation();
        let distance_sq: f64 = delta.length_squared();

        let f = GRAVITY_CONSTANT / distance_sq;
        let force_unit_mass = delta * f;
        acc1.0 += force_unit_mass * *m2;
        acc2.0 -= force_unit_mass * *m1;
    }
}

fn integrate(mut query: Query<(&mut Acceleration, &mut Transform, &mut LastPos)>) {
    let dt_sq = (DELTA_TIME * DELTA_TIME) as f64;
    for (mut acceleration, mut transform, mut last_pos) in &mut query {
        // verlet integration
        // x(t+dt) = 2x(t) - x(t-dt) + a(t)dt^2 + O(dt^4)

        let new_pos = transform.translation * 2.0 - last_pos.0 + acceleration.0 * dt_sq;
        acceleration.0 = DVec3::ZERO;
        last_pos.0 = transform.translation;
        transform.translation = new_pos;
    }
}

fn look_at_star(
    mut camera: Query<&mut Transform, (With<Camera>, Without<Star>)>,
    star: Query<&Transform, With<Star>>,
) {
    let mut camera = camera.single_mut();
    let star = star.single();
    let new_rotation = camera
        .looking_at(star.translation, DVec3::Y)
        .rotation
        .lerp(camera.rotation, 0.1);
    camera.rotation = new_rotation;
}
