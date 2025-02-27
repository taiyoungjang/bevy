use std::f64::consts::*;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::NotShadowCaster,
    prelude::*,
};
use rand::{thread_rng, Rng};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_startup_system(setup)
        .add_system(light_sway)
        .add_system(movement)
        .run();
}

#[derive(Component)]
struct Movable;

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // ground plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 100.0 })),
        material: materials.add(StandardMaterial {
            base_color: Color::GREEN,
            perceptual_roughness: 1.0,
            ..default()
        }),
        ..default()
    });

    // cubes
    let mut rng = thread_rng();
    for _ in 0..100 {
        let x = rng.gen_range(-5.0..5.0);
        let y = rng.gen_range(-5.0..5.0);
        let z = rng.gen_range(-5.0..5.0);
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Cube { size: 0.5 })),
                material: materials.add(StandardMaterial {
                    base_color: Color::BLUE,
                    ..default()
                }),
                transform: Transform::from_xyz(x, y, z),
                ..default()
            },
            Movable,
        ));
    }

    // ambient light
    commands.insert_resource(AmbientLight {
        color: Color::rgb(0.0, 1.0, 1.0),
        brightness: 0.14,
    });

    for x in 0..4 {
        for z in 0..4 {
            let x = x as f64 - 2.0;
            let z = z as f64 - 2.0;
            // red spot_light
            commands
                .spawn(SpotLightBundle {
                    transform: Transform::from_xyz(1.0 + x, 2.0, z)
                        .looking_at(DVec3::new(1.0 + x, 0.0, z), DVec3::X),
                    spot_light: SpotLight {
                        intensity: 200.0, // lumens
                        color: Color::WHITE,
                        shadows_enabled: true,
                        inner_angle: PI as f32 / 4.0 * 0.85,
                        outer_angle: PI as f32 / 4.0,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|builder| {
                    builder.spawn(PbrBundle {
                        mesh: meshes.add(Mesh::from(shape::UVSphere {
                            radius: 0.05,
                            ..default()
                        })),
                        material: materials.add(StandardMaterial {
                            base_color: Color::RED,
                            emissive: Color::rgba_linear(1.0, 0.0, 0.0, 0.0),
                            ..default()
                        }),
                        ..default()
                    });
                    builder.spawn((
                        PbrBundle {
                            transform: Transform::from_translation(DVec3::Z * -0.1),
                            mesh: meshes.add(Mesh::from(shape::UVSphere {
                                radius: 0.1,
                                ..default()
                            })),
                            material: materials.add(StandardMaterial {
                                base_color: Color::MAROON,
                                emissive: Color::rgba_linear(0.125, 0.0, 0.0, 0.0),
                                ..default()
                            }),
                            ..default()
                        },
                        NotShadowCaster,
                    ));
                });
        }
    }

    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-4.0, 5.0, 10.0).looking_at(DVec3::ZERO, DVec3::Y),
        ..default()
    });
}

fn light_sway(time: Res<Time>, mut query: Query<(&mut Transform, &mut SpotLight)>) {
    for (mut transform, mut angles) in query.iter_mut() {
        transform.rotation = DQuat::from_euler(
            EulerRot::XYZ,
            -FRAC_PI_2 + (time.elapsed_seconds_f64() * 0.67 * 3.0).sin() * 0.5,
            (time.elapsed_seconds_f64() * 3.0).sin() * 0.5,
            0.0,
        );
        let angle = ((time.elapsed_seconds_f64() * 1.2).sin() + 1.0) * (FRAC_PI_4 - 0.1);
        angles.inner_angle = angle as f32 * 0.8;
        angles.outer_angle = angle as f32;
    }
}

fn movement(
    input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Movable>>,
) {
    for mut transform in &mut query {
        let mut direction = DVec3::ZERO;
        if input.pressed(KeyCode::Up) {
            direction.z -= 1.0;
        }
        if input.pressed(KeyCode::Down) {
            direction.z += 1.0;
        }
        if input.pressed(KeyCode::Left) {
            direction.x -= 1.0;
        }
        if input.pressed(KeyCode::Right) {
            direction.x += 1.0;
        }
        if input.pressed(KeyCode::PageUp) {
            direction.y += 1.0;
        }
        if input.pressed(KeyCode::PageDown) {
            direction.y -= 1.0;
        }

        transform.translation += time.delta_seconds_f64() * 2.0 * direction;
    }
}
