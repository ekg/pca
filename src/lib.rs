//! Principal component analysis (PCA)

#![doc = include_str!("../README.md")]

use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::svd::SVD;
use std::error::Error;

/// Principal component analysis (PCA) structure
pub struct PCA {
    /// the rotation matrix
    rotation: Option<Array2<f64>>,
    /// mean of input data
    mean: Option<Array1<f64>>,
    /// scale of input data
    scale: Option<Array1<f64>>,
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

impl PCA {
    /// Create a new PCA struct with default values
    ///
    /// # Examples
    ///
    /// ```
    /// use pca::PCA;
    /// let pca = PCA::new();
    /// ```
    pub fn new() -> Self {
        Self {
            rotation: None,
            mean: None,
            scale: None,
        }
    }

    /// Fit the PCA rotation to the data
    ///
    /// This computes the mean, scaling and rotation to apply PCA 
    /// to the input data matrix.
    ///
    /// * `x` - Input data as a 2D array
    /// * `tol` - Tolerance for excluding low variance components.
    ///           If None, all components are kept.
    ///
    /// # Errors
    ///
    /// Returns an error if the input matrix has fewer than 2 rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use pca::PCA;
    ///
    /// let x = array![[1.0, 2.0], [3.0, 4.0]];
    /// let mut pca = PCA::new();
    /// pca.fit(x, None).unwrap();
    /// ```
    pub fn fit(&mut self, mut x: Array2<f64>, tol: Option<f64>) -> Result<(), Box<dyn Error>> {
        let n = x.nrows();
        if n < 2 {
            return Err("Input matrix must have at least 2 rows.".into());
        }

        // Compute mean for centering
        let mean = x.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;
        self.mean = Some(mean.clone());
        x -= &mean;

        // Compute scale
        let std_dev = x.map_axis(Axis(0), |v| v.std(1.0));
        self.scale = Some(std_dev.clone());
        x /= &std_dev.mapv(|v| if v != 0. { v } else { 1. });

        let k = std::cmp::min(n, x.ncols());

        // Compute SVD
        let (_u, mut s, vt) = x.svd(true, true).map_err(|_| "Failed to compute SVD")?;

        // Normalize singular values
        s.mapv_inplace(|v| v / ((n as f64 - 1.0).max(1.0)).sqrt());

        // Compute Rotation
        let rotation;
        if let Some(t) = tol {
            let threshold = s[0] * t;
            let rank = s.iter().take_while(|&si| *si > threshold).count();
            rotation = vt.unwrap().slice_move(s![..std::cmp::min(rank, k), ..]).reversed_axes();
        } else {
            rotation = vt.unwrap().slice_move(s![..k, ..]).reversed_axes();
        }
        self.rotation = Some(rotation);

        Ok(())
    }

    /// Apply the PCA rotation to the data
    ///
    /// This projects the data into the PCA space using the 
    /// previously computed rotation, mean and scale.
    ///
    /// * `x` - Input data to transform
    ///
    /// # Errors
    ///
    /// Returns an error if PCA has not been fitted yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use pca::PCA;
    ///
    /// let x = array![[1.0, 2.0]]; 
    /// let pca = PCA::new();
    /// pca.transform(x).unwrap();
    /// ```
    pub fn transform(&self, mut x: Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        match (&self.rotation, &self.mean, &self.scale) {
            (Some(rotation), Some(mean), Some(scale)) => {
                x -= mean;
                x /= scale;
                Ok(x.dot(rotation))
            }
            _ => Err("PCA not fitted yet.".into()),
        }
    }
}


#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::*;
    use float_cmp::approx_eq;

    fn test_pca(input: Array2<f64>, expected_output: Array2<f64>, tol: Option<f64>) {
        let mut pca = PCA::new();
        pca.fit(input.clone(), tol).unwrap();
        let output = pca.transform(input).unwrap();

        eprintln!("output: {:?}", output);
        eprintln!("expected_output: {:?}", expected_output);
        
        // Calculate absolute values for arrays
        let output_abs = output.mapv_into(f64::abs);
        let expected_output_abs = expected_output.mapv_into(f64::abs);

        // Compare arrays
        let equal = output_abs.shape() == expected_output_abs.shape() &&
            output_abs.iter().zip(expected_output_abs.iter())
                      .all(|(a, b)| approx_eq!(f64, *a, *b, epsilon = 1.0e-6));
        assert!(equal);
    }

    #[test]
    fn test_pca_2x2() {
        let input = array![[0.5855288, -0.1093033], 
                           [0.7094660, -0.4534972]];
        let expected = array![[-1.0, 4.109492e-16],
                              [1.0, 2.647088e-16]];

        test_pca(input, expected, None);
    }
    
    #[test]
    fn test_pca_3x5() {
        let input = array![[0.5855288, -0.4534972, 0.6300986, -0.9193220, 0.3706279],
                           [0.7094660, 0.6058875, -0.2761841, -0.1162478, 0.5202165],
                           [-0.1093033, -1.8179560, -0.2841597, 1.8173120, -0.7505320]];
        
        let expected = array![[-1.197063, 1.00708746, -2.503967e-17],
                              [-1.098771, -1.03624808, 1.566910e-16],
                              [2.295834, 0.02916061, 3.427952e-17]];

        test_pca(input, expected, None);
    }

    #[test]  
    fn test_pca_5x5() {
        let input = array![[0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219],
                           [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851],
                           [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284],
                           [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374],
                           [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095]];

        let expected = array![[0.7982358, 1.5873362, -0.81051708, 0.1302931, -1.372149e-16],
                              [-2.8355417, 0.1286742, -0.04173705, -0.2169158, -1.150065e-16],
                              [-0.1933991, -0.7382029, 0.34873461, 0.6235378, 8.618531e-17],
                              [1.0944225, -1.5788366, -0.57193457, -0.2771233, 4.276631e-16],
                              [1.1362824, 0.6010291, 1.07545409, -0.2597919, -6.793015e-17]];

        test_pca(input, expected, None);
    }

    #[test]
    fn test_pca_5x7() {
        let input = array![[0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
                           [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
                           [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
                           [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
                           [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]];
        
        let expected = array![[1.7585642, -1.3627442, -1.26793991, 0.050491148, -2.437707e-16],
                              [-3.0789816, -0.8169344, 0.05331594, 0.277390220, 3.028250e-15],
                              [-0.5847649, 0.8009802, -0.25561556, -0.836326771, -5.830345e-15],
                              [0.5487077, 1.8825193, -0.30961285, 0.511856077, 3.404246e-15],
                              [1.3564746, -0.5038210, 1.77985239, -0.003410674, -3.187086e-16]];

        test_pca(input, expected, None);
    }

    #[test]
    #[should_panic]
    fn test_pca_fit_insufficient_samples() {
        let x = array![[1.0]];
        
        let mut pca = PCA::new();
        pca.fit(x, None).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_pca_transform_not_fitted() {
        let x = array![[1.0, 2.0]];
        
        let pca = PCA::new();
        let _ = pca.transform(x).unwrap();
    }
}
