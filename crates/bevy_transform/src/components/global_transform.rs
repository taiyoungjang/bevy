use std::ops::Mul;

use super::Transform;
use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_math::{Affine3A, DAffine3, DMat4, DQuat, DVec3, Mat4, Vec4, Vec3, Vec3A};
use bevy_reflect::{std_traits::ReflectDefault, FromReflect, Reflect};

/// Describe the position of an entity relative to the reference frame.
///
/// * To place or move an entity, you should set its [`Transform`].
/// * To get the global transform of an entity, you should get its [`GlobalTransform`].
/// * For transform hierarchies to work correctly, you must have both a [`Transform`] and a [`GlobalTransform`].
///   * You may use the [`TransformBundle`](crate::TransformBundle) to guarantee this.
///
/// ## [`Transform`] and [`GlobalTransform`]
///
/// [`Transform`] is the position of an entity relative to its parent position, or the reference
/// frame if it doesn't have a [`Parent`](bevy_hierarchy::Parent).
///
/// [`GlobalTransform`] is the position of an entity relative to the reference frame.
///
/// [`GlobalTransform`] is updated from [`Transform`] in the systems labeled
/// [`TransformPropagate`](crate::TransformSystem::TransformPropagate).
///
/// This system runs in stage [`CoreStage::PostUpdate`](crate::CoreStage::PostUpdate). If you
/// update the [`Transform`] of an entity in this stage or after, you will notice a 1 frame lag
/// before the [`GlobalTransform`] is updated.
///
/// # Examples
///
/// - [`global_vs_local_translation`]
///
/// [`global_vs_local_translation`]: https://github.com/bevyengine/bevy/blob/latest/examples/transforms/global_vs_local_translation.rs
#[derive(Component, Debug, PartialEq, Clone, Copy, Reflect, FromReflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[reflect(Component, Default, PartialEq)]
pub struct GlobalTransform(DAffine3);

macro_rules! impl_local_axis {
    ($pos_name: ident, $neg_name: ident, $axis: ident) => {
        #[doc=std::concat!("Return the local ", std::stringify!($pos_name), " vector (", std::stringify!($axis) ,").")]
        #[inline]
        pub fn $pos_name(&self) -> DVec3 {
            (self.0.matrix3 * DVec3::$axis).normalize()
        }

        #[doc=std::concat!("Return the local ", std::stringify!($neg_name), " vector (-", std::stringify!($axis) ,").")]
        #[inline]
        pub fn $neg_name(&self) -> DVec3 {
            -self.$pos_name()
        }
    };
}

impl GlobalTransform {
    /// An identity [`GlobalTransform`] that maps all points in space to themselves.
    pub const IDENTITY: Self = Self(DAffine3::IDENTITY);

    #[doc(hidden)]
    #[inline]
    pub fn from_xyz(x: f64, y: f64, z: f64) -> Self {
        Self::from_translation(DVec3::new(x, y, z))
    }

    #[doc(hidden)]
    #[inline]
    pub fn from_translation(translation: DVec3) -> Self {
        GlobalTransform(DAffine3::from_translation(translation))
    }

    #[doc(hidden)]
    #[inline]
    pub fn from_rotation(rotation: DQuat) -> Self {
        GlobalTransform(DAffine3::from_rotation_translation(rotation, DVec3::ZERO))
    }

    #[doc(hidden)]
    #[inline]
    pub fn from_scale(scale: DVec3) -> Self {
        GlobalTransform(DAffine3::from_scale(scale))
    }

    /// Returns the 3d affine transformation matrix as a [`Mat4`].
    #[inline]
    pub fn compute_matrix(&self) -> DMat4 { DMat4::from(self.0) }

    /// Returns the 3d affine transformation matrix as a [`Mat4`].
    #[inline]
    pub fn compute_matrix_mat4(&self) -> Mat4 {
        let dmat = DMat4::from(self.0);
        let x_axis = dmat.x_axis;
        let y_axis = dmat.y_axis;
        let z_axis = dmat.z_axis;
        let w_axis = dmat.w_axis;
        Mat4::from_cols(
            Vec4::new(x_axis.x as f32,x_axis.y as f32,x_axis.z as f32,x_axis.w as f32),
            Vec4::new(y_axis.x as f32,y_axis.y as f32,y_axis.z as f32,y_axis.w as f32),
            Vec4::new(z_axis.x as f32,z_axis.y as f32,z_axis.z as f32,z_axis.w as f32),
            Vec4::new(w_axis.x as f32,w_axis.y as f32,w_axis.z as f32,w_axis.w as f32)
        )
    }
    /// Returns the 3d affine transformation matrix as an [`Affine3A`].
    #[inline]
    pub fn affine(&self) -> Affine3A {
        Affine3A::from_mat4( self.compute_matrix_mat4() )
    }

    /// Returns the 3d affine transformation matrix as an [`DAffine3`].
    #[inline]
    pub fn daffine(&self) -> DAffine3 {
        self.0
    }
    /// Returns the transformation as a [`Transform`].
    ///
    /// The transform is expected to be non-degenerate and without shearing, or the output
    /// will be invalid.
    #[inline]
    pub fn compute_transform(&self) -> Transform {
        let (scale, rotation, translation) = self.0.to_scale_rotation_translation();
        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`.
    ///
    /// The transform is expected to be non-degenerate and without shearing, or the output
    /// will be invalid.
    #[inline]
    pub fn to_scale_rotation_translation(&self) -> (DVec3, DQuat, DVec3) {
        self.0.to_scale_rotation_translation()
    }

    impl_local_axis!(right, left, X);
    impl_local_axis!(up, down, Y);
    impl_local_axis!(back, forward, Z);

    /// Get the translation as a [`DVec3`].
    #[inline]
    pub fn translation(&self) -> DVec3 {
        self.0.translation.into()
    }

    /// Get the translation as a [`DVec3`].
    #[inline]
    pub fn translation_vec3(&self) -> Vec3 {
        Vec3::new( self.0.translation.x as f32, self.0.translation.y as f32, self.0.translation.z as f32)
    }

    /// Mutably access the internal translation.
    #[inline]
    pub fn translation_mut(&mut self) -> &mut DVec3 {
        &mut self.0.translation
    }

    /// Get the translation as a [`Vec3A`].
    #[inline]
    pub fn translation_vec3a(&self) -> DVec3 {
        self.0.translation
    }

    /// Get an upper bound of the radius from the given `extents`.
    #[inline]
    pub fn radius_vec3(&self, extents: DVec3) -> f64 {
        (self.0.matrix3 * extents).length()
    }

    /// Get an upper bound of the radius from the given `extents`.
    #[inline]
    pub fn radius_vec3a(&self, extents: Vec3A) -> f32 {
        (self.affine().matrix3 * extents).length()
    }

    /// Transforms the given `point`, applying shear, scale, rotation and translation.
    ///
    /// This moves `point` into the local space of this [`GlobalTransform`].
    #[inline]
    pub fn transform_point(&self, point: DVec3) -> DVec3 {
        self.0.transform_point3(point)
    }

    /// Transforms the given `point`, applying shear, scale, rotation and translation.
    ///
    /// This moves `point` into the local space of this [`GlobalTransform`].
    #[inline]
    pub fn transform_point_vec3(&self, point: Vec3) -> Vec3 {
        self.affine().transform_point3( point )
    }

    /// Multiplies `self` with `transform` component by component, returning the
    /// resulting [`GlobalTransform`]
    #[inline]
    pub fn mul_transform(&self, transform: Transform) -> Self {
        Self(self.0 * transform.compute_affine())
    }
}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Transform> for GlobalTransform {
    fn from(transform: Transform) -> Self {
        Self(transform.compute_affine())
    }
}

impl From<DAffine3> for GlobalTransform {
    fn from(affine: DAffine3) -> Self {
        Self(affine)
    }
}

impl From<DMat4> for GlobalTransform {
    fn from(matrix: DMat4) -> Self {
        Self(DAffine3::from_mat4(matrix))
    }
}

impl Mul<GlobalTransform> for GlobalTransform {
    type Output = GlobalTransform;

    #[inline]
    fn mul(self, global_transform: GlobalTransform) -> Self::Output {
        GlobalTransform(self.0 * global_transform.0)
    }
}

impl Mul<Transform> for GlobalTransform {
    type Output = GlobalTransform;

    #[inline]
    fn mul(self, transform: Transform) -> Self::Output {
        self.mul_transform(transform)
    }
}

impl Mul<DVec3> for GlobalTransform {
    type Output = DVec3;

    #[inline]
    fn mul(self, value: DVec3) -> Self::Output {
        self.transform_point(value)
    }
}
