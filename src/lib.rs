use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Eigh;
use std::error::Error;

pub struct PCA {
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    scale: bool,
}

impl PCA {
    pub fn new(scale: bool) -> Self {
        Self {
            components: None,
            mean: None,
            scale,
        }
    }

    pub fn fit(&mut self, mut x: Array2<f64>, n_components: usize) -> Result<(), Box<dyn Error>> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err("Input matrix must have at least 2 rows.".into());
        }

        let mean = x.mean_axis(Axis(0)).unwrap();
        self.mean = Some(mean.clone());
        x -= &mean;

        if self.scale {
            let std_dev = x.map_axis(Axis(0), |v| v.std(1.0));
            x /= &std_dev.mapv(|v| if v != 0. { v } else { 1. });
        }

        let covariance_matrix = x.t().dot(&x) / (n_samples - 1) as f64;
        let (eigenvalues, eigenvectors) = covariance_matrix
            .eigh(ndarray_linalg::UPLO::Upper)
            .unwrap();

        let mut eigen_pairs: Vec<(_, _)> = eigenvalues
            .iter()
            .cloned()
            .zip(eigenvectors.axis_iter(Axis(1)))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let (_, n_features) = x.dim();
        let mut w = Array2::zeros((n_features, n_components));
        for (i, pair) in eigen_pairs.iter().take(n_components).enumerate() {
            let (_, ev) = pair;
            w.column_mut(i).assign(ev);
        }

        self.components = Some(w);
        Ok(())
    }

    pub fn transform(&self, mut x: Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        match (&self.components, &self.mean) {
            (Some(components), Some(mean)) => {
                x -= mean;
                if self.scale {
                    let std_dev = x.map_axis(Axis(0), |v| v.std(1.0));
                    x /= &std_dev.mapv(|v| if v != 0. { v } else { 1. });
                }
                Ok(x.dot(components))
            }
            _ => Err("PCA not fitted yet.".into()),
        }
    }
}


#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::*;

    #[test]
    fn test_pca_2x2() {
        let x = array![[0.5855288, -0.1093033], 
                       [0.7094660, -0.4534972]];
        
        let expected_pcs = array![[-1.0, 4.109492e-16],
                                  [1.0, 2.647088e-16]];

        let mut pca = PCA::new(true);
        pca.fit(x.clone(), 2).unwrap();

        let pcs = pca.transform(x).unwrap();

        assert_eq!(pcs, expected_pcs);
    }

    /*
    #[test]
    fn test_pca_3x5() {
        let x = array![[0.5855288, -0.4534972, 0.6300986, -0.9193220, 0.3706279],
                       [0.7094660, 0.6058875, -0.2761841, -0.1162478, 0.5202165],
                       [-0.1093033, -1.8179560, -0.2841597, 1.8173120, -0.7505320]];
        
        let expected_pcs = array![[-1.197063, 1.00708746, -2.503967e-17],
                                  [-1.098771, -1.03624808, 1.566910e-16],
                                  [2.295834, 0.02916061, 3.427952e-17]];

        let mut pca = PCA::new(true);
        pca.fit(x.clone(), 3).unwrap();
        
        let pcs = pca.transform(x).unwrap();
        
        assert_eq!(pcs, expected_pcs);
    }

    #[test]  
    fn test_pca_5x5() {
        let x = array![[0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219],
                       [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851],
                       [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284],
                       [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374],
                       [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095]];

        let expected_pcs = array![[0.7982358, 1.5873362, -0.81051708, 0.1302931, -1.372149e-16],
                                  [-2.8355417, 0.1286742, -0.04173705, -0.2169158, -1.150065e-16],
                                  [-0.1933991, -0.7382029, 0.34873461, 0.6235378, 8.618531e-17],
                                  [1.0944225, -1.5788366, -0.57193457, -0.2771233, 4.276631e-16],
                                  [1.1362824, 0.6010291, 1.07545409, -0.2597919, -6.793015e-17]];
        
        let mut pca = PCA::new(true);
        pca.fit(x.clone(), 5).unwrap();  
        
        let pcs = pca.transform(x).unwrap();
        
        assert_eq!(pcs, expected_pcs);
    }

    #[test]
    fn test_pca_5x7() {
        let x = array![[0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
                       [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
                       [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
                       [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
                       [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]];
        
        let expected_pcs = array![[1.7585642, -1.3627442, -1.26793991, 0.050491148, -2.437707e-16],
                                  [-3.0789816, -0.8169344, 0.05331594, 0.277390220, 3.028250e-15],
                                  [-0.5847649, 0.8009802, -0.25561556, -0.836326771, -5.830345e-15],
                                  [0.5487077, 1.8825193, -0.30961285, 0.511856077, 3.404246e-15],
                                  [1.3564746, -0.5038210, 1.77985239, -0.003410674, -3.187086e-16]];

        let mut pca = PCA::new(true);
        pca.fit(x.clone(), 5).unwrap();
        
        let pcs = pca.transform(x).unwrap();

        assert_eq!(pcs, expected_pcs);  
}
    */

    #[test]
    #[should_panic]
    fn test_pca_fit_insufficient_samples() {
        let x = array![[1.0]];
        
        let mut pca = PCA::new(true);
        pca.fit(x, 1).unwrap(); 
    }

    #[test]
    #[should_panic]
    fn test_pca_transform_not_fitted() {
        let x = array![[1.0, 2.0]];
        
        let pca = PCA::new(true);
        let _ = pca.transform(x).unwrap();
    }
}
