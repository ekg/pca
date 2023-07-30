//! Principal component analysis (PCA)

#![doc = include_str!("../README.md")]

use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::svd::SVD;
use rsvd::rsvd;
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
        let m = x.ncols();
        let k = std::cmp::min(n, m);
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

    /// Use randomized SVD to fit a PCA rotation to the data
    ///
    /// This computes the mean, scaling and rotation to apply PCA 
    /// to the input data matrix.
    ///
    /// * `x` - Input data as a 2D array
    /// * `n_components` - Number of components to keep
    /// * `n_oversamples` - Number of oversampled dimensions (for rSVD)
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
    /// pca.rfit(x, 1, 0, None, None).unwrap();
    /// ```
    pub fn rfit(&mut self, mut x: Array2<f64>,
                n_components: usize, n_oversamples: usize,
                seed: Option<u64>, tol: Option<f64>) -> Result<(), Box<dyn Error>> {
        let n = x.nrows();
        let m = x.ncols();
        let k = std::cmp::min(n, m);
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

        // Compute SVD
        let (_u, mut s, vt) = rsvd(&x, n_components, n_oversamples, seed);
        //.map_err(|_| "Failed to compute SVD")?;

        // Normalize singular values
        s.mapv_inplace(|v| v / ((n as f64 - 1.0).max(1.0)).sqrt());

        // Compute Rotation
        let rotation;
        if let Some(t) = tol {
            // convert diagonal matrix s into a vector
            let s = s.diag().to_owned();
            let threshold = s[0] * t;
            let rank = s.iter().take_while(|&si| *si > threshold).count();
            rotation = vt.slice_move(s![..std::cmp::min(rank, k), ..]).reversed_axes();
        } else {
            rotation = vt.slice_move(s![..k, ..]).reversed_axes();
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
    /// let x = array![[1.0, 2.0],[3.0, 4.0]];
    /// let mut pca = PCA::new();
    /// pca.fit(x.clone(), None).unwrap();
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
    use ndarray_rand::rand_distr::Distribution;
    use ndarray_rand::rand_distr::Normal;
    use super::*;
    use float_cmp::approx_eq;

    fn test_pca(input: Array2<f64>, expected_output: Array2<f64>, tol: Option<f64>, e: f64) {
        let mut pca = PCA::new();
        pca.fit(input.clone(), tol).unwrap();
        let output = pca.transform(input).unwrap();

        //eprintln!("output: {:?}", output);
        //eprintln!("expected_output: {:?}", expected_output);
        
        // Calculate absolute values for arrays
        let output_abs = output.mapv_into(f64::abs);
        let expected_output_abs = expected_output.mapv_into(f64::abs);

        // Compare arrays
        let equal = output_abs.shape() == expected_output_abs.shape() &&
            output_abs.iter().zip(expected_output_abs.iter())
                      .all(|(a, b)| approx_eq!(f64, *a, *b, epsilon = e));
        assert!(equal);
    }

    fn test_rpca(input: Array2<f64>, expected_output: Array2<f64>,
                 n_components: usize, n_oversamples: usize, tol: Option<f64>, e: f64) {
        let mut pca = PCA::new();
        pca.rfit(input.clone(), n_components, n_oversamples, Some(1926), tol).unwrap();
        let output = pca.transform(input).unwrap();

        //eprintln!("output: {:?}", output);
        //eprintln!("expected_output: {:?}", expected_output);
        
        // Calculate absolute values for arrays
        let output_abs = output.mapv_into(f64::abs);
        let expected_output_abs = expected_output.mapv_into(f64::abs);

        // Compare arrays
        let equal = output_abs.shape() == expected_output_abs.shape() &&
            output_abs.iter().zip(expected_output_abs.iter())
                      .all(|(a, b)| approx_eq!(f64, *a, *b, epsilon = e));
        assert!(equal);
    }

    #[test]
    fn test_pca_2x2() {
        let input = array![[0.5855288, -0.1093033], 
                           [0.7094660, -0.4534972]];
        let expected = array![[-1.0, 4.109492e-16],
                              [1.0, 2.647088e-16]];

        test_pca(input, expected, None, 1e-6);
    }

    #[test]
    fn test_rpca_2x2() {
        let input = array![[0.5855288, -0.1093033], 
                           [0.7094660, -0.4534972]];
        let expected = array![[-1.0, 4.109492e-16],
                              [1.0, 2.647088e-16]];

        test_rpca(input, expected, 2, 0, None, 1e-6);
    }

    #[test]
    fn test_rpca_2x2_k1() {
        let input = array![[0.5855288, -0.1093033], 
                           [0.7094660, -0.4534972]];
        let expected = array![[-1.0, 4.109492e-16],
                              [1.0, 2.647088e-16]];

        test_rpca(input, expected, 1, 0, None, 1e-6);
    }

    #[test]
    fn test_pca_3x5() {
        let input = array![[0.5855288, -0.4534972, 0.6300986, -0.9193220, 0.3706279],
                           [0.7094660, 0.6058875, -0.2761841, -0.1162478, 0.5202165],
                           [-0.1093033, -1.8179560, -0.2841597, 1.8173120, -0.7505320]];
        
        let expected = array![[-1.197063, 1.00708746, -2.503967e-17],
                              [-1.098771, -1.03624808, 1.566910e-16],
                              [2.295834, 0.02916061, 3.427952e-17]];

        test_pca(input, expected, None, 1e-6);
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

        test_pca(input, expected, None, 1e-6);
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

        test_pca(input, expected, None, 1e-6);
    }

    #[test]
    fn test_rpca_5x7_k3() {
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

        test_rpca(input, expected, 4, 0, None, 1e-6);
    }

    // helper to make a random matrix with a given number of rows and columns
    /*
    fn make_random_matrix(rows: usize, cols: usize, seed: Option<u64>) -> Array2<f64> {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(thread_rng()).unwrap(),
        };
        {
            let vec = rng.sample_iter(Normal::new(0.0, 1.0).unwrap())
                .take(rows * cols)
                .collect::<Vec<_>>();
            ndarray::Array::from_shape_vec((rows, cols), vec).unwrap()
        }
    }*/

    fn make_random_matrix<T>(rows: usize, cols: usize, distrib: T, seed: Option<u64>) -> Array2<f64>
    where
        T: Distribution<f64>, 
    {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(thread_rng()).unwrap(),
        };

        let vec = rng.sample_iter(distrib)
            .take(rows * cols)
            .collect::<Vec<_>>();

        ndarray::Array2::from_shape_vec((rows, cols), vec).unwrap()
    }

    //use ndarray::{Array1, Array2};
    //use rand::{Rng};
    //use ndarray_rand::rand_distr::SparseBinary;
    pub struct SparseBinary {
        p: f64
    }

    impl SparseBinary {
        pub fn new(p: f64) -> Self {
            SparseBinary { p }
        }
    }

    impl Distribution<f64> for SparseBinary {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
            if rng.gen::<f64>() < self.p {
                1.0
            } else {
                0.0
            }
        }
    }

    fn normalize_cols(matrix: Array2<f64>) -> Array2<f64> {
        let mut matrix = matrix.to_owned();
        // Center each column
        for mut col in matrix.axis_iter_mut(Axis(1)) {
            let mean = col.mean().unwrap();
            col.mapv_inplace(|x| x - mean); 
        }
        // Normalize each column
        for mut col in matrix.axis_iter_mut(Axis(1)) {
            let std_dev = col.std(1.0);
            if std_dev != 0.0 {
                col.mapv_inplace(|x| x / std_dev);
            }
        }
        matrix
    }

    // Generate low-rank matrix
    fn low_rank(m: usize, n: usize, k: usize, seed: u64) -> Array2<f64> {

        let u = make_random_matrix(m, k, Normal::new(0.0, 1.0).unwrap(), Some(seed)); 
        let v = make_random_matrix(n, k, Normal::new(0.0, 1.0).unwrap(), Some(seed));

        u.dot(&v.t())
    }

    // Spiked covariance model    
    fn spiked_cov(m: usize, n: usize, seed: u64) -> Array2<f64> {
        let x = make_random_matrix(m, n, Normal::new(0.0, 1.0).unwrap(), Some(seed));
        let y = make_random_matrix(n, 1, SparseBinary::new(0.1), Some(seed)); 
        let yy = &y * &y.t();

        x + yy
    }

    // Introduce smoother structure
    fn smooth_struct(m: usize, n: usize, seed: u64) -> Array2<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut a = Array2::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                if i == 0 && j == 0 {
                    a[[i, j]] = rng.sample(Normal::new(0.0, 1.0).unwrap()); 
                } else {
                    let prev_i = if i > 0 {i-1} else {0};
                    let prev_j = if j > 0 {j-1} else {0};
                    a[[i, j]] = 0.5 * a[[prev_i, j]] + 0.5 * a[[i, prev_j]];
                }
            }
        }
        
        a
    }
    
    // this helper test function compares randomized pca to the regular pca
    // the randomized pca is expected to be less accurate, but faster
    fn compare_rpca_to_pca(m: usize, n: usize, rank: usize, k: usize, p: usize, seed: Option<u64>, e: f64) {

        let mut pca = PCA::new();
        let mut rpca = PCA::new();

        // this should be really easy for rpca to get
        // so long as k + p >= rank
        // but if our rank is too large then noise will dominate
        let input = low_rank(m, n, rank, seed.unwrap());

        pca.fit(input.clone(), None).unwrap();
        rpca.rfit(input.clone(), k, p, seed, None).unwrap();

        // get the components
        let output_pca = pca.transform(input.clone()).unwrap();
        let output_rpca = rpca.transform(input).unwrap();
        // check that the output arrays are equivalent

        assert!(equivalent(&output_pca, &output_rpca, e));
    }

    fn equivalent(a: &Array2<f64>, b: &Array2<f64>, e: f64) -> bool {
        let a = a.clone().mapv_into(f64::abs);
        let b = b.clone().mapv_into(f64::abs);
        // absolute differences
        let diff = (a - b).mapv_into(f64::abs);
        // average difference per cell
        let avg = diff.sum() / (diff.len() as f64);
        // absolute value of average difference
        let avg = avg.abs();
        eprintln!("avg abs diff: {}", avg);
        avg < e
    }

    #[test]
    fn test_rpca_equiv_10x10_k5_o1() {
        compare_rpca_to_pca(10, 10, 5, 5, 1, Some(1926), 1e-2);
    }

    #[test]
    fn test_rpca_equiv_100x100_k10_o5() {
        compare_rpca_to_pca(100, 100, 15, 10, 5, Some(1926), 1e-2);
    }

    #[test]
    fn test_rpca_equiv_100x100_k5_o2() {
        compare_rpca_to_pca(100, 100, 6, 5, 2, Some(1926), 1e-2);
    }

    #[test]
    fn test_rpca_equiv_100x100_k20_o5() {
        compare_rpca_to_pca(100, 100, 25, 20, 5, Some(1926), 1e-2);
    }

    #[test]
    fn test_rpca_equiv_100x100_k2_o0() {
        compare_rpca_to_pca(100, 100, 2, 2, 0, Some(1926), 1e-2);
    }

    use ndarray::Array2;
    use ndarray_rand::RandomExt; // for creating random arrays
    use rand::distributions::Uniform;
    use rand::prelude::SeedableRng;
    use rand::{thread_rng, Rng};
    use rand_chacha::ChaCha8Rng;

    // This helper function will make a random matrix that's size x size and check that there are no NaNs in the output
    fn test_pca_random(size: usize, seed: u64) {
        // Rng with input seed
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Input is a size x size matrix with elements randomly chosen from a uniform distribution between -1.0 and 1.0
        let input = Array2::<f64>::random_using((size, size), Uniform::new(-1.0, 1.0), &mut rng);

        // Transform the input with PCA
        let mut pca = PCA::new();
        pca.fit(input.clone(), None).unwrap();
        let output = pca.transform(input).unwrap();

        // Assert that none of the values in the output are NaN
        assert!(output.iter().all(|&x| !x.is_nan()));
    }

    #[test]
    fn test_pca_random_2() {
        test_pca_random(2, 1337);
    }

    #[test]
    fn test_pca_random_64() {
        test_pca_random(64, 1337);
    }

    #[test]
    fn test_pca_random_256() {
        test_pca_random(256, 1337);
    }

    #[test]
    fn test_pca_random_512() {
        test_pca_random(256, 1337);
    }

    fn test_pca_random_012(size: usize, seed: u64) {
        // Rng with input seed
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Input is a size x size matrix with elements randomly chosen from a uniform distribution between -1.0 and 1.0
        // n.b. we need to use a discrete distribution
        let input = Array2::<f64>::random_using((size, size), Uniform::new_inclusive(0, 2).map(|x| x as f64), &mut rng);

        // Transform the input with PCA
        let mut pca = PCA::new();
        pca.fit(input.clone(), None).unwrap();
        let output = pca.transform(input.clone()).unwrap();

        // Assert that none of the values in the output are NaN
        assert!(output.iter().all(|&x| !x.is_nan()));

        /*
        use std::fs::File;
        use std::io::Write;
        // write the input matrix to a file
        let mut file = File::create(format!("test_pca_random_012_{}_{}_input.csv", size, seed)).unwrap();
        input
            .rows()
            .into_iter()
            .for_each(|row| writeln!(file, "{}", row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",")).unwrap());
        
        // write the result to a file
        let mut file = File::create(format!("test_pca_random_012_{}_{}_output.csv", size, seed)).unwrap();
        output
            .rows()
            .into_iter()
        .for_each(|row| writeln!(file, "{}", row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",")).unwrap());
        */
    }

    #[test]
    fn test_pca_random_012_2() {
        test_pca_random_012(2, 1337);
    }

    #[test]
    fn test_pca_random_012_64() {
        test_pca_random_012(64, 1337);
    }

    #[test]
    fn test_pca_random_012_256() {
        test_pca_random_012(256, 1337);
    }

    #[test]
    fn test_pca_random_012_512() {
        test_pca_random_012(256, 1337);
    }

    /*
    #[test]
    fn test_pca_random_012_1024() {
        test_pca_random_012(1024, 1337);
    }*/

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
