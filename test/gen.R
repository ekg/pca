#!/usr/bin/env Rscript

# Importing required library
library(stats)

options(width = 160)

# Define a function to generate a matrix and perform PCA
perform_pca <- function(n, m) {
  # Set the seed for reproducibility
  set.seed(12345)
  
  # Generate a nxm random matrix
  matrix_data <- matrix(rnorm(n*m), nrow=n, ncol=m)
  
  # Print the generated matrix
  print(paste("Generated matrix of size", n, "x", m, ":"))
  print(matrix_data)
  
  # Perform PCA
  pca_results <- prcomp(matrix_data, scale. = TRUE)
  
  # Print the principal components
  print("Principal Components:")
  print(pca_results$x)
}

# Test the function with different matrix sizes
#perform_pca(3, 5)
#perform_pca(5, 5)
#perform_pca(5, 7)
perform_pca(2, 2)