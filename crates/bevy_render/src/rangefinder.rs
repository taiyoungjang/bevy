use bevy_math::{DMat4, DVec4, Mat4};

/// A distance calculator for the draw order of [`PhaseItem`](crate::render_phase::PhaseItem)s.
pub struct ViewRangefinder3d {
    inverse_view_row_2: DVec4,
}

impl ViewRangefinder3d {
    /// Creates a 3D rangefinder for a view matrix
    pub fn from_view_matrix(view_matrix: &DMat4) -> ViewRangefinder3d {
        let inverse_view_matrix = view_matrix.inverse();
        ViewRangefinder3d {
            inverse_view_row_2: inverse_view_matrix.row(2),
        }
    }

    /// Calculates the distance, or view-space `Z` value, for a transform
    #[inline]
    pub fn distance(&self, transform: &DMat4) -> f64 {
        // NOTE: row 2 of the inverse view matrix dotted with column 3 of the model matrix
        // gives the z component of translation of the mesh in view-space
        self.inverse_view_row_2.dot(transform.col(3))
    }
    #[inline]
    pub fn distance_mat4(&self, transform: &Mat4) -> f64 {
        // NOTE: row 2 of the inverse view matrix dotted with column 3 of the model matrix
        // gives the z component of translation of the mesh in view-space
        let v = transform.col(3);
        self.inverse_view_row_2.dot( DVec4::new(v.x as f64, v.y as f64, v.z as f64, v.w as f64) )
    }
}

#[cfg(test)]
mod tests {
    use super::ViewRangefinder3d;
    use bevy_math::{DMat4, DVec3};

    #[test]
    fn distance() {
        let view_matrix = DMat4::from_translation(DVec3::new(0.0, 0.0, -1.0));
        let rangefinder = ViewRangefinder3d::from_view_matrix(&view_matrix);
        assert_eq!(rangefinder.distance(&DMat4::IDENTITY), 1.0);
        assert_eq!(
            rangefinder.distance(&DMat4::from_translation(DVec3::new(0.0, 0.0, 1.0))),
            2.0
        );
    }
}
