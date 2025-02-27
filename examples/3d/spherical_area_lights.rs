//! Demonstrates how lighting is affected by different radius of point lights.

use bevy::prelude::*;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.0, 2.5, 5.0).looking_at(DVec3::ZERO, DVec3::Y),
        ..default()
    });

    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 100.0 })),
        material: materials.add(StandardMaterial {
            base_color: Color::rgb(0.2, 0.2, 0.2),
            perceptual_roughness: 0.08,
            ..default()
        }),
        ..default()
    });

    const COUNT: usize = 6;
    let position_range = -4.0f64..4.0f64;
    let radius_range = 0.0f64..0.8f64;
    let pos_len = position_range.end - position_range.start;
    let radius_len = radius_range.end - radius_range.start;
    let mesh = meshes.add(Mesh::from(shape::UVSphere {
        sectors: 128,
        stacks: 64,
        ..default()
    }));

    for i in 0..COUNT {
        let percent = i as f64 / COUNT as f64;
        let radius = radius_range.start + percent * radius_len;

        // sphere light
        commands
            .spawn(PbrBundle {
                mesh: mesh.clone(),
                material: materials.add(StandardMaterial {
                    base_color: Color::rgb(0.5, 0.5, 1.0),
                    unlit: true,
                    ..default()
                }),
                transform: Transform::from_xyz(position_range.start + percent * pos_len, 0.6, 0.0)
                    .with_scale(DVec3::splat(radius)),
                ..default()
            })
            .with_children(|children| {
                children.spawn(PointLightBundle {
                    point_light: PointLight {
                        intensity: 1500.0,
                        radius: radius as f32,
                        color: Color::rgb(0.2, 0.2, 1.0),
                        ..default()
                    },
                    ..default()
                });
            });
    }
}
