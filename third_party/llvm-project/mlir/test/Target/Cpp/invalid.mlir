// RUN: mlir-translate -split-input-file -mlir-to-cpp -verify-diagnostics %s

// expected-error@+1 {{'func.func' op with multiple blocks needs variables declared at top}}
func.func @multiple_blocks() {
^bb1:
    cf.br ^bb2
^bb2:
    return
}

// -----

func.func @unsupported_std_op(%arg0: f64) -> f64 {
  // expected-error@+1 {{'math.abs' op unable to find printer for op}}
  %0 = math.abs %arg0 : f64
  return %0 : f64
}

// -----

// expected-error@+1 {{cannot emit integer type 'i80'}}
func.func @unsupported_integer_type(%arg0 : i80) {
  return
}

// -----

// expected-error@+1 {{cannot emit float type 'f80'}}
func.func @unsupported_float_type(%arg0 : f80) {
  return
}

// -----

// expected-error@+1 {{cannot emit type 'memref<100xf32>'}}
func.func @memref_type(%arg0 : memref<100xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit type 'vector<100xf32>'}}
func.func @vector_type(%arg0 : vector<100xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit tensor type with non static shape}}
func.func @non_static_shape(%arg0 : tensor<?xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit unranked tensor type}}
func.func @unranked_tensor(%arg0 : tensor<*xf32>) {
  return
}
