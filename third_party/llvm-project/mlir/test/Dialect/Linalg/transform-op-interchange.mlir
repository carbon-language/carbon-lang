// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

//       CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @interchange_generic
func.func @interchange_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //      CHECK:   linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP]]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = math.exp %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_generic : benefit(1) {
    %args = operands
    %results = types
    %0 = pdl.operation "linalg.generic"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_generic in %arg1
    transform.structured.interchange %0 { iterator_interchange = [1, 0]}
  }
}

// -----

func.func @interchange_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-note @below {{attempted to apply to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_generic : benefit(1) {
    %args = operands
    %results = types
    %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_generic in %arg1
    // expected-error @below {{applies to linalg.generic ops}}
    transform.structured.interchange %0 { iterator_interchange = [1, 0]}
  }
}
