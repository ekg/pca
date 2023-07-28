# Principal component analysis (PCA)

This is a rust library for performing principal component analysis (PCA). It supports:

- Fitting a PCA model on a data matrix 
- Projecting data into the PCA space
- Specifying variance explained tolerance to reduce dimensionality

The implementation follows R's prcomp, and should provide equivalent results with minor differences due to numerical stability and the ambiguity of component sign.
Tests confirm the correspondence.
[The PCA is obtained via SVD](https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca).

## Usage

```rust
use pca::PCA;
use ndarray::array;

// Create PCA instance
let mut pca = PCA::new(); 

// Input data 
let x = array![[1.0, 2.0], 
               [3.0, 4.0]];

// Fit PCA model                
pca.fit(x.clone(), None).unwrap();

// Project data
let transformed = pca.transform(x).unwrap();
```

The `fit()` method computes the PCA rotation matrix, mean and scaling factors. It takes the input data and an optional variance explained tolerance. 

The `transform()` method applies the PCA rotation to project new data into the PCA space.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
pca = "0.1.2"
```

Or just `cargo add pca` to get the latest version.

## Authors

Erik Garrison <erik.garrison@gmail.com>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
