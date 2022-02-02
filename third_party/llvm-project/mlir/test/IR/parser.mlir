// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

// CHECK-DAG: #map{{[0-9]+}} = affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0, d1, d2, d4, d3)>
#map0 = affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0, d1, d2, d4, d3)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<()[s0] -> (0, s0 - 1)>
#inline_map_minmax_loop1 = affine_map<()[s0] -> (0, s0 - 1)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<()[s0] -> (100, s0 + 1)>
#inline_map_minmax_loop2 = affine_map<()[s0] -> (100, s0 + 1)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0)>
#bound_map1 = affine_map<(i, j)[s] -> (i + j + s)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 + d1)>
#inline_map_loop_bounds2 = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-DAG: #map{{[0-9]+}} = affine_map<(d0)[s0] -> (d0 + s0, d0 - s0)>
#bound_map2 = affine_map<(i)[s] -> (i + s, i - s)>

// All maps appear in arbitrary order before all sets, in arbitrary order.
// CHECK-NOT: Placeholder

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0)[s0, s1] : (d0 >= 0, -d0 + s0 >= 0, s0 - 5 == 0, -d0 + s1 + 1 >= 0)>
#set0 = affine_set<(i)[N, M] : (i >= 0, -i + N >= 0, N - 5 == 0, -i + M + 1 >= 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0] : (d0 >= 0, d1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 >= 0, d1 >= 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0) : (d0 - 1 == 0)>
#set2 = affine_set<(d0) : (d0 - 1 == 0)>

// CHECK-DAG: [[$SET_TRUE:#set[0-9]+]] = affine_set<() : (0 == 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0)[s0] : (d0 - 2 >= 0, -d0 + 4 >= 0)>

// CHECK: func private @foo(i32, i64) -> f32
func private @foo(i32, i64) -> f32

// CHECK: func private @bar()
func private @bar() -> ()

// CHECK: func private @baz() -> (i1, index, f32)
func private @baz() -> (i1, index, f32)

// CHECK: func private @missingReturn()
func private @missingReturn()

// CHECK: func private @int_types(i0, i1, i2, i4, i7, i87) -> (i1, index, i19)
func private @int_types(i0, i1, i2, i4, i7, i87) -> (i1, index, i19)

// CHECK: func private @sint_types(si2, si4) -> (si7, si1023)
func private @sint_types(si2, si4) -> (si7, si1023)

// CHECK: func private @uint_types(ui2, ui4) -> (ui7, ui1023)
func private @uint_types(ui2, ui4) -> (ui7, ui1023)

// CHECK: func private @float_types(f80, f128)
func private @float_types(f80, f128)

// CHECK: func private @vectors(vector<f32>, vector<1xf32>, vector<2x4xf32>)
func private @vectors(vector<f32>, vector<1 x f32>, vector<2x4xf32>)

// CHECK: func private @tensors(tensor<*xf32>, tensor<*xvector<2x4xf32>>, tensor<1x?x4x?x?xi32>, tensor<i8>)
func private @tensors(tensor<* x f32>, tensor<* x vector<2x4xf32>>,
                 tensor<1x?x4x?x?xi32>, tensor<i8>)

// CHECK: func private @tensor_encoding(tensor<16x32xf64, "sparse">)
func private @tensor_encoding(tensor<16x32xf64, "sparse">)

// CHECK: func private @large_shape_dimension(tensor<9223372036854775807xf32>)
func private @large_shape_dimension(tensor<9223372036854775807xf32>)

// CHECK: func private @functions((memref<1x?x4x?x?xi32, #map0>, memref<8xi8>) -> (), () -> ())
func private @functions((memref<1x?x4x?x?xi32, #map0, 0>, memref<8xi8, #map1, 0>) -> (), ()->())

// CHECK: func private @memrefs2(memref<2x4x8xi8, 1>)
func private @memrefs2(memref<2x4x8xi8, #map2, 1>)

// CHECK: func private @memrefs3(memref<2x4x8xi8>)
func private @memrefs3(memref<2x4x8xi8, affine_map<(d0, d1, d2) -> (d0, d1, d2)>>)

// CHECK: func private @memrefs_drop_triv_id_inline(memref<2xi8>)
func private @memrefs_drop_triv_id_inline(memref<2xi8, affine_map<(d0) -> (d0)>>)

// CHECK: func private @memrefs_drop_triv_id_inline0(memref<2xi8>)
func private @memrefs_drop_triv_id_inline0(memref<2xi8, affine_map<(d0) -> (d0)>, 0>)

// CHECK: func private @memrefs_drop_triv_id_inline1(memref<2xi8, 1>)
func private @memrefs_drop_triv_id_inline1(memref<2xi8, affine_map<(d0) -> (d0)>, 1>)

// Test memref with custom memory space

// CHECK: func private @memrefs_nomap_nospace(memref<5x6x7xf32>)
func private @memrefs_nomap_nospace(memref<5x6x7xf32>)

// CHECK: func private @memrefs_map_nospace(memref<5x6x7xf32, #map{{[0-9]+}}>)
func private @memrefs_map_nospace(memref<5x6x7xf32, #map3>)

// CHECK: func private @memrefs_nomap_intspace(memref<5x6x7xf32, 3>)
func private @memrefs_nomap_intspace(memref<5x6x7xf32, 3>)

// CHECK: func private @memrefs_map_intspace(memref<5x6x7xf32, #map{{[0-9]+}}, 5>)
func private @memrefs_map_intspace(memref<5x6x7xf32, #map3, 5>)

// CHECK: func private @memrefs_nomap_strspace(memref<5x6x7xf32, "local">)
func private @memrefs_nomap_strspace(memref<5x6x7xf32, "local">)

// CHECK: func private @memrefs_map_strspace(memref<5x6x7xf32, #map{{[0-9]+}}, "private">)
func private @memrefs_map_strspace(memref<5x6x7xf32, #map3, "private">)

// CHECK: func private @memrefs_nomap_dictspace(memref<5x6x7xf32, {memSpace = "special", subIndex = 1 : i64}>)
func private @memrefs_nomap_dictspace(memref<5x6x7xf32, {memSpace = "special", subIndex = 1}>)

// CHECK: func private @memrefs_map_dictspace(memref<5x6x7xf32, #map{{[0-9]+}}, {memSpace = "special", subIndex = 3 : i64}>)
func private @memrefs_map_dictspace(memref<5x6x7xf32, #map3, {memSpace = "special", subIndex = 3}>)

// CHECK: func private @complex_types(complex<i1>) -> complex<f32>
func private @complex_types(complex<i1>) -> complex<f32>

// CHECK: func private @memref_with_index_elems(memref<1x?xindex>)
func private @memref_with_index_elems(memref<1x?xindex>)

// CHECK: func private @memref_with_complex_elems(memref<1x?xcomplex<f32>>)
func private @memref_with_complex_elems(memref<1x?xcomplex<f32>>)

// CHECK: func private @memref_with_vector_elems(memref<1x?xvector<10xf32>>)
func private @memref_with_vector_elems(memref<1x?xvector<10xf32>>)

// CHECK: func private @memref_with_custom_elem(memref<1x?x!test.memref_element>)
func private @memref_with_custom_elem(memref<1x?x!test.memref_element>)

// CHECK: func private @memref_of_memref(memref<1xmemref<1xf64>>)
func private @memref_of_memref(memref<1xmemref<1xf64>>)

// CHECK: func private @memref_of_unranked_memref(memref<1xmemref<*xf32>>)
func private @memref_of_unranked_memref(memref<1xmemref<*xf32>>)

// CHECK: func private @unranked_memref_of_memref(memref<*xmemref<1xf32>>)
func private @unranked_memref_of_memref(memref<*xmemref<1xf32>>)

// CHECK: func private @unranked_memref_of_unranked_memref(memref<*xmemref<*xi32>>)
func private @unranked_memref_of_unranked_memref(memref<*xmemref<*xi32>>)

// CHECK: func private @unranked_memref_with_complex_elems(memref<*xcomplex<f32>>)
func private @unranked_memref_with_complex_elems(memref<*xcomplex<f32>>)

// CHECK: func private @unranked_memref_with_index_elems(memref<*xindex>)
func private @unranked_memref_with_index_elems(memref<*xindex>)

// CHECK: func private @unranked_memref_with_vector_elems(memref<*xvector<10xf32>>)
func private @unranked_memref_with_vector_elems(memref<*xvector<10xf32>>)

// CHECK-LABEL: func @simpleCFG(%{{.*}}: i32, %{{.*}}: f32) -> i1 {
func @simpleCFG(%arg0: i32, %f: f32) -> i1 {
  // CHECK: %{{.*}} = "foo"() : () -> i64
  %1 = "foo"() : ()->i64
  // CHECK: "bar"(%{{.*}}) : (i64) -> (i1, i1, i1)
  %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
  // CHECK: return %{{.*}}#1
  return %2#1 : i1
// CHECK: }
}

// CHECK-LABEL: func @simpleCFGUsingBBArgs(%{{.*}}: i32, %{{.*}}: i64) {
func @simpleCFGUsingBBArgs(i32, i64) {
^bb42 (%arg0: i32, %f: i64):
  // CHECK: "bar"(%{{.*}}) : (i64) -> (i1, i1, i1)
  %2:3 = "bar"(%f) : (i64) -> (i1,i1,i1)
  // CHECK: return{{$}}
  return
// CHECK: }
}

// CHECK-LABEL: func @multiblock() {
func @multiblock() {
  return     // CHECK:   return
^bb1:         // CHECK: ^bb1:   // no predecessors
  br ^bb4     // CHECK:   br ^bb3
^bb2:         // CHECK: ^bb2:   // pred: ^bb2
  br ^bb2     // CHECK:   br ^bb2
^bb4:         // CHECK: ^bb3:   // pred: ^bb1
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: func @emptyMLF() {
func @emptyMLF() {
  return     // CHECK:  return
}            // CHECK: }

// CHECK-LABEL: func @func_with_one_arg(%{{.*}}: i1) -> i2 {
func @func_with_one_arg(%c : i1) -> i2 {
  // CHECK: %{{.*}} = "foo"(%{{.*}}) : (i1) -> i2
  %b = "foo"(%c) : (i1) -> (i2)
  return %b : i2   // CHECK: return %{{.*}} : i2
} // CHECK: }

// CHECK-LABEL: func @func_with_two_args(%{{.*}}: f16, %{{.*}}: i8) -> (i1, i32) {
func @func_with_two_args(%a : f16, %b : i8) -> (i1, i32) {
  // CHECK: %{{.*}}:2 = "foo"(%{{.*}}, %{{.*}}) : (f16, i8) -> (i1, i32)
  %c:2 = "foo"(%a, %b) : (f16, i8)->(i1, i32)
  return %c#0, %c#1 : i1, i32  // CHECK: return %{{.*}}#0, %{{.*}}#1 : i1, i32
} // CHECK: }

// CHECK-LABEL: func @second_order_func() -> (() -> ()) {
func @second_order_func() -> (() -> ()) {
// CHECK-NEXT: %{{.*}} = constant @emptyMLF : () -> ()
  %c = constant @emptyMLF : () -> ()
// CHECK-NEXT: return %{{.*}} : () -> ()
  return %c : () -> ()
}

// CHECK-LABEL: func @third_order_func() -> (() -> (() -> ())) {
func @third_order_func() -> (() -> (() -> ())) {
// CHECK-NEXT:  %{{.*}} = constant @second_order_func : () -> (() -> ())
  %c = constant @second_order_func : () -> (() -> ())
// CHECK-NEXT:  return %{{.*}} : () -> (() -> ())
  return %c : () -> (() -> ())
}

// CHECK-LABEL: func @identity_functor(%{{.*}}: () -> ()) -> (() -> ())  {
func @identity_functor(%a : () -> ()) -> (() -> ())  {
// CHECK-NEXT: return %{{.*}} : () -> ()
  return %a : () -> ()
}

// CHECK-LABEL: func @func_ops_in_loop() {
func @func_ops_in_loop() {
  // CHECK: %{{.*}} = "foo"() : () -> i64
  %a = "foo"() : ()->i64
  // CHECK: affine.for %{{.*}} = 1 to 10 {
  affine.for %i = 1 to 10 {
    // CHECK: %{{.*}} = "doo"() : () -> f32
    %b = "doo"() : ()->f32
    // CHECK: "bar"(%{{.*}}, %{{.*}}) : (i64, f32) -> ()
    "bar"(%a, %b) : (i64, f32) -> ()
  // CHECK: }
  }
  // CHECK: return
  return
  // CHECK: }
}


// CHECK-LABEL: func @loops() {
func @loops() {
  // CHECK: affine.for %{{.*}} = 1 to 100 step 2 {
  affine.for %i = 1 to 100 step 2 {
    // CHECK: affine.for %{{.*}} = 1 to 200 {
    affine.for %j = 1 to 200 {
    }        // CHECK:     }
  }          // CHECK:   }
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: func @complex_loops() {
func @complex_loops() {
  affine.for %i1 = 1 to 100 {      // CHECK:   affine.for %{{.*}} = 1 to 100 {
    affine.for %j1 = 1 to 100 {    // CHECK:     affine.for %{{.*}} = 1 to 100 {
       // CHECK: "foo"(%{{.*}}, %{{.*}}) : (index, index) -> ()
       "foo"(%i1, %j1) : (index,index) -> ()
    }                       // CHECK:     }
    "boo"() : () -> ()      // CHECK:     "boo"() : () -> ()
    affine.for %j2 = 1 to 10 {     // CHECK:     affine.for %{{.*}} = 1 to 10 {
      affine.for %k2 = 1 to 10 {   // CHECK:       affine.for %{{.*}} = 1 to 10 {
        "goo"() : () -> ()  // CHECK:         "goo"() : () -> ()
      }                     // CHECK:       }
    }                       // CHECK:     }
  }                         // CHECK:   }
  return                    // CHECK:   return
}                           // CHECK: }

// CHECK: func @triang_loop(%{{.*}}: index, %{{.*}}: memref<?x?xi32>) {
func @triang_loop(%arg0: index, %arg1: memref<?x?xi32>) {
  %c = arith.constant 0 : i32       // CHECK: %{{.*}} = arith.constant 0 : i32
  affine.for %i0 = 1 to %arg0 {      // CHECK: affine.for %{{.*}} = 1 to %{{.*}} {
    affine.for %i1 = affine_map<(d0)[]->(d0)>(%i0)[] to %arg0 {  // CHECK:   affine.for %{{.*}} = #map{{[0-9]+}}(%{{.*}}) to %{{.*}} {
      memref.store %c, %arg1[%i0, %i1] : memref<?x?xi32>  // CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
    }          // CHECK:     }
  }            // CHECK:   }
  return       // CHECK:   return
}              // CHECK: }

// CHECK: func @minmax_loop(%{{.*}}: index, %{{.*}}: index, %{{.*}}: memref<100xf32>) {
func @minmax_loop(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
  // CHECK: affine.for %{{.*}} = max #map{{.*}}()[%{{.*}}] to min #map{{.*}}()[%{{.*}}] {
  affine.for %i0 = max affine_map<()[s]->(0,s-1)>()[%arg0] to min affine_map<()[s]->(100,s+1)>()[%arg1] {
    // CHECK: "foo"(%{{.*}}, %{{.*}}) : (memref<100xf32>, index) -> ()
    "foo"(%arg2, %i0) : (memref<100xf32>, index) -> ()
  }      // CHECK:   }
  return // CHECK:   return
}        // CHECK: }

// CHECK-LABEL: func @loop_bounds(%{{.*}}: index) {
func @loop_bounds(%N : index) {
  // CHECK: %{{.*}} = "foo"(%{{.*}}) : (index) -> index
  %s = "foo"(%N) : (index) -> index
  // CHECK: affine.for %{{.*}} = %{{.*}} to %{{.*}}
  affine.for %i = %s to %N {
    // CHECK: affine.for %{{.*}} = #map{{[0-9]+}}(%{{.*}}) to 0
    affine.for %j = affine_map<(d0)[]->(d0)>(%i)[] to 0 step 1 {
       // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]
       %w1 = affine.apply affine_map<(d0, d1)[s0] -> (d0+d1)> (%i, %j) [%s]
       // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]
       %w2 = affine.apply affine_map<(d0, d1)[s0] -> (s0+1)> (%i, %j) [%s]
       // CHECK: affine.for %{{.*}} = #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}] to #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}] {
       affine.for %k = #bound_map1 (%w1, %i)[%N] to affine_map<(i, j)[s] -> (i + j + s)> (%w2, %j)[%s] {
          // CHECK: "foo"(%{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index) -> ()
          "foo"(%i, %j, %k) : (index, index, index)->()
          // CHECK: %{{.*}} = arith.constant 30 : index
          %c = arith.constant 30 : index
          // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})
          %u = affine.apply affine_map<(d0, d1)->(d0+d1)> (%N, %c)
          // CHECK: affine.for %{{.*}} = max #map{{.*}}(%{{.*}})[%{{.*}}] to min #map{{.*}}(%{{.*}})[%{{.*}}] {
          affine.for %l = max #bound_map2(%i)[%u] to min #bound_map2(%k)[%c] {
            // CHECK: "bar"(%{{.*}}) : (index) -> ()
            "bar"(%l) : (index) -> ()
          } // CHECK:           }
       }    // CHECK:         }
     }      // CHECK:       }
  }         // CHECK:     }
  return    // CHECK:   return
}           // CHECK: }

// CHECK-LABEL: func @ifinst(%{{.*}}: index) {
func @ifinst(%N: index) {
  %c = arith.constant 200 : index // CHECK   %{{.*}} = arith.constant 200
  affine.for %i = 1 to 10 {           // CHECK   affine.for %{{.*}} = 1 to 10 {
    affine.if #set0(%i)[%N, %c] {     // CHECK     affine.if #set0(%{{.*}})[%{{.*}}, %{{.*}}] {
      %x = arith.constant 1 : i32
       // CHECK: %{{.*}} = arith.constant 1 : i32
      %y = "add"(%x, %i) : (i32, index) -> i32 // CHECK: %{{.*}} = "add"(%{{.*}}, %{{.*}}) : (i32, index) -> i32
      %z = "mul"(%y, %y) : (i32, i32) -> i32 // CHECK: %{{.*}} = "mul"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    } else { // CHECK } else {
      affine.if affine_set<(i)[N] : (i - 2 >= 0, 4 - i >= 0)>(%i)[%N]  {      // CHECK  affine.if (#set1(%{{.*}})[%{{.*}}]) {
        // CHECK: %{{.*}} = arith.constant 1 : index
        %u = arith.constant 1 : index
        // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]
        %w = affine.apply affine_map<(d0,d1)[s0] -> (d0+d1+s0)> (%i, %i) [%u]
      } else {            // CHECK     } else {
        %v = arith.constant 3 : i32 // %c3_i32 = arith.constant 3 : i32
      }
    }       // CHECK     }
  }         // CHECK   }
  return    // CHECK   return
}           // CHECK }

// CHECK-LABEL: func @simple_ifinst(%{{.*}}: index) {
func @simple_ifinst(%N: index) {
  %c = arith.constant 200 : index // CHECK   %{{.*}} = arith.constant 200
  affine.for %i = 1 to 10 {           // CHECK   affine.for %{{.*}} = 1 to 10 {
    affine.if #set0(%i)[%N, %c] {     // CHECK     affine.if #set0(%{{.*}})[%{{.*}}, %{{.*}}] {
      %x = arith.constant 1 : i32
       // CHECK: %{{.*}} = arith.constant 1 : i32
      %y = "add"(%x, %i) : (i32, index) -> i32 // CHECK: %{{.*}} = "add"(%{{.*}}, %{{.*}}) : (i32, index) -> i32
      %z = "mul"(%y, %y) : (i32, i32) -> i32 // CHECK: %{{.*}} = "mul"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    }       // CHECK     }
  }         // CHECK   }
  return    // CHECK   return
}           // CHECK }

// CHECK-LABEL: func @attributes() {
func @attributes() {
  // CHECK: "foo"()
  "foo"(){} : ()->()

  // CHECK: "foo"() {a = 1 : i64, b = -423 : i64, c = [true, false], d = 1.600000e+01 : f64}  : () -> ()
  "foo"() {a = 1, b = -423, c = [true, false], d = 16.0 } : () -> ()

  // CHECK: "foo"() {map1 = #map{{[0-9]+}}}
  "foo"() {map1 = #map1} : () -> ()

  // CHECK: "foo"() {map2 = #map{{[0-9]+}}}
  "foo"() {map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : () -> ()

  // CHECK: "foo"() {map12 = [#map{{[0-9]+}}, #map{{[0-9]+}}]}
  "foo"() {map12 = [#map1, #map2]} : () -> ()

  // CHECK: "foo"() {set1 = #set{{[0-9]+}}}
  "foo"() {set1 = #set1} : () -> ()

  // CHECK: "foo"() {set2 = #set{{[0-9]+}}}
  "foo"() {set2 = affine_set<(d0, d1, d2) : (d0 >= 0, d1 >= 0, d2 - d1 == 0)>} : () -> ()

  // CHECK: "foo"() {set12 = [#set{{[0-9]+}}, #set{{[0-9]+}}]}
  "foo"() {set12 = [#set1, #set2]} : () -> ()

  // CHECK: "foo"() {dictionary = {bool = true, fn = @ifinst}}
  "foo"() {dictionary = {bool = true, fn = @ifinst}} : () -> ()

  // Check that the dictionary attribute elements are sorted.
  // CHECK: "foo"() {dictionary = {bar = false, bool = true, fn = @ifinst}}
  "foo"() {dictionary = {fn = @ifinst, bar = false, bool = true}} : () -> ()

  // CHECK: "foo"() {d = 1.000000e-09 : f64, func = [], i123 = 7 : i64, if = "foo"} : () -> ()
  "foo"() {if = "foo", func = [], i123 = 7, d = 1.e-9} : () -> ()

  // CHECK: "foo"() {fn = @attributes, if = @ifinst} : () -> ()
  "foo"() {fn = @attributes, if = @ifinst} : () -> ()

  // CHECK: "foo"() {int = 0 : i42} : () -> ()
  "foo"() {int = 0 : i42} : () -> ()
  return
}

// CHECK-LABEL: func @ssa_values() -> (i16, i8) {
func @ssa_values() -> (i16, i8) {
  // CHECK: %{{.*}}:2 = "foo"() : () -> (i1, i17)
  %0:2 = "foo"() : () -> (i1, i17)
  br ^bb2

^bb1:       // CHECK: ^bb1: // pred: ^bb2
  // CHECK: %{{.*}}:2 = "baz"(%{{.*}}#1, %{{.*}}#0, %{{.*}}#1) : (f32, i11, i17) -> (i16, i8)
  %1:2 = "baz"(%2#1, %2#0, %0#1) : (f32, i11, i17) -> (i16, i8)

  // CHECK: return %{{.*}}#0, %{{.*}}#1 : i16, i8
  return %1#0, %1#1 : i16, i8

^bb2:       // CHECK: ^bb2:  // pred: ^bb0
  // CHECK: %{{.*}}:2 = "bar"(%{{.*}}#0, %{{.*}}#1) : (i1, i17) -> (i11, f32)
  %2:2 = "bar"(%0#0, %0#1) : (i1, i17) -> (i11, f32)
  br ^bb1
}

// CHECK-LABEL: func @bbargs() -> (i16, i8) {
func @bbargs() -> (i16, i8) {
  // CHECK: %{{.*}}:2 = "foo"() : () -> (i1, i17)
  %0:2 = "foo"() : () -> (i1, i17)
  br ^bb1(%0#1, %0#0 : i17, i1)

^bb1(%x: i17, %y: i1):       // CHECK: ^bb1(%{{.*}}: i17, %{{.*}}: i1):
  // CHECK: %{{.*}}:2 = "baz"(%{{.*}}, %{{.*}}, %{{.*}}#1) : (i17, i1, i17) -> (i16, i8)
  %1:2 = "baz"(%x, %y, %0#1) : (i17, i1, i17) -> (i16, i8)
  return %1#0, %1#1 : i16, i8
}

// CHECK-LABEL: func @verbose_terminators() -> (i1, i17)
func @verbose_terminators() -> (i1, i17) {
  %0:2 = "foo"() : () -> (i1, i17)
// CHECK:  br ^bb1(%{{.*}}#0, %{{.*}}#1 : i1, i17)
  "std.br"(%0#0, %0#1)[^bb1] : (i1, i17) -> ()

^bb1(%x : i1, %y : i17):
// CHECK:  cond_br %{{.*}}, ^bb2(%{{.*}} : i17), ^bb3(%{{.*}}, %{{.*}} : i1, i17)
  "std.cond_br"(%x, %y, %x, %y) [^bb2, ^bb3] {operand_segment_sizes = dense<[1, 1, 2]>: vector<3xi32>} : (i1, i17, i1, i17) -> ()

^bb2(%a : i17):
  %true = arith.constant true
// CHECK:  return %{{.*}}, %{{.*}} : i1, i17
  "std.return"(%true, %a) : (i1, i17) -> ()

^bb3(%b : i1, %c : i17):
// CHECK:  return %{{.*}}, %{{.*}} : i1, i17
  "std.return"(%b, %c) : (i1, i17) -> ()
}

// CHECK-LABEL: func @condbr_simple
func @condbr_simple() -> (i32) {
  %cond = "foo"() : () -> i1
  %a = "bar"() : () -> i32
  %b = "bar"() : () -> i64
  // CHECK: cond_br %{{.*}}, ^bb1(%{{.*}} : i32), ^bb2(%{{.*}} : i64)
  cond_br %cond, ^bb1(%a : i32), ^bb2(%b : i64)

// CHECK: ^bb1({{.*}}: i32): // pred: ^bb0
^bb1(%x : i32):
  br ^bb2(%b: i64)

// CHECK: ^bb2({{.*}}: i64): // 2 preds: ^bb0, ^bb1
^bb2(%y : i64):
  %z = "foo"() : () -> i32
  return %z : i32
}

// CHECK-LABEL: func @condbr_moarargs
func @condbr_moarargs() -> (i32) {
  %cond = "foo"() : () -> i1
  %a = "bar"() : () -> i32
  %b = "bar"() : () -> i64
  // CHECK: cond_br %{{.*}}, ^bb1(%{{.*}}, %{{.*}} : i32, i64), ^bb2(%{{.*}}, %{{.*}}, %{{.*}} : i64, i32, i32)
  cond_br %cond, ^bb1(%a, %b : i32, i64), ^bb2(%b, %a, %a : i64, i32, i32)

^bb1(%x : i32, %y : i64):
  return %x : i32

^bb2(%x2 : i64, %y2 : i32, %z2 : i32):
  %z = "foo"() : () -> i32
  return %z : i32
}


// Test pretty printing of constant names.
// CHECK-LABEL: func @constants
func @constants() -> (i32, i23, i23, i1, i1) {
  // CHECK: %{{.*}} = arith.constant 42 : i32
  %x = arith.constant 42 : i32
  // CHECK: %{{.*}} = arith.constant 17 : i23
  %y = arith.constant 17 : i23

  // This is a redundant definition of 17, the asmprinter gives it a unique name
  // CHECK: %{{.*}} = arith.constant 17 : i23
  %z = arith.constant 17 : i23

  // CHECK: %{{.*}} = arith.constant true
  %t = arith.constant true
  // CHECK: %{{.*}} = arith.constant false
  %f = arith.constant false

  // The trick to parse type declarations should not interfere with hex
  // literals.
  // CHECK: %{{.*}} = arith.constant 3890 : i32
  %h = arith.constant 0xf32 : i32

  // CHECK: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  return %x, %y, %z, %t, %f : i32, i23, i23, i1, i1
}

// CHECK-LABEL: func @typeattr
func @typeattr() -> () {
^bb0:
// CHECK: "foo"() {bar = tensor<*xf32>} : () -> ()
  "foo"(){bar = tensor<*xf32>} : () -> ()
  return
}

// CHECK-LABEL: func @stringquote
func @stringquote() -> () {
^bb0:
  // CHECK: "foo"() {bar = "a\22quoted\22string"} : () -> ()
  "foo"(){bar = "a\"quoted\"string"} : () -> ()

  // CHECK-NEXT: "typed_string" : !foo.string
  "foo"(){bar = "typed_string" : !foo.string} : () -> ()
  return
}

// CHECK-LABEL: func @unitAttrs
func @unitAttrs() -> () {
  // CHECK-NEXT: "foo"() {unitAttr}
  "foo"() {unitAttr = unit} : () -> ()

  // CHECK-NEXT: "foo"() {unitAttr}
  "foo"() {unitAttr} : () -> ()

  // CHECK-NEXT: "foo"() {nested = {unitAttr}}
  "foo"() {nested = {unitAttr}} : () -> ()
  return
}

// CHECK-LABEL: func @floatAttrs
func @floatAttrs() -> () {
^bb0:
  // CHECK: "foo"() {a = 4.000000e+00 : f64, b = 2.000000e+00 : f64, c = 7.100000e+00 : f64, d = -0.000000e+00 : f64} : () -> ()
  "foo"(){a = 4.0, b = 2.0, c = 7.1, d = -0.0} : () -> ()
  return
}

// CHECK-LABEL: func private @externalfuncattr
func private @externalfuncattr() -> ()
  // CHECK: attributes {dialect.a = "a\22quoted\22string", dialect.b = 4.000000e+00 : f64, dialect.c = tensor<*xf32>}
  attributes {dialect.a = "a\"quoted\"string", dialect.b = 4.0, dialect.c = tensor<*xf32>}

// CHECK-LABEL: func private @funcattrempty
func private @funcattrempty() -> ()
  attributes {}

// CHECK-LABEL: func private @funcattr
func private @funcattr() -> ()
  // CHECK: attributes {dialect.a = "a\22quoted\22string", dialect.b = 4.000000e+00 : f64, dialect.c = tensor<*xf32>}
  attributes {dialect.a = "a\"quoted\"string", dialect.b = 4.0, dialect.c = tensor<*xf32>} {
^bb0:
  return
}

// CHECK-LABEL: func @funcattrwithblock
func @funcattrwithblock() -> ()
  attributes {} {
^bb0:
  return
}

// CHECK-label func @funcsimplemap
#map_simple0 = affine_map<()[] -> (10)>
#map_simple1 = affine_map<()[s0] -> (s0)>
#map_non_simple0 = affine_map<(d0)[] -> (d0)>
#map_non_simple1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map_non_simple2 = affine_map<()[s0, s1] -> (s0 + s1)>
#map_non_simple3 = affine_map<()[s0] -> (s0 + 3)>
func @funcsimplemap(%arg0: index, %arg1: index) -> () {
  affine.for %i0 = 0 to #map_simple0()[] {
  // CHECK: affine.for %{{.*}} = 0 to 10 {
    affine.for %i1 = 0 to #map_simple1()[%arg1] {
    // CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
      affine.for %i2 = 0 to #map_non_simple0(%i0)[] {
      // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}(%{{.*}}) {
        affine.for %i3 = 0 to #map_non_simple1(%i0)[%arg1] {
        // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}(%{{.*}})[%{{.*}}] {
          affine.for %i4 = 0 to #map_non_simple2()[%arg1, %arg0] {
          // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}()[%{{.*}}, %{{.*}}] {
            affine.for %i5 = 0 to #map_non_simple3()[%arg0] {
            // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}()[%{{.*}}] {
              %c42_i32 = arith.constant 42 : i32
            }
          }
        }
      }
    }
  }
  return
}

// CHECK-LABEL: func @splattensorattr
func @splattensorattr() -> () {
^bb0:
  // CHECK: "splatBoolTensor"() {bar = dense<false> : tensor<i1>} : () -> ()
  "splatBoolTensor"(){bar = dense<false> : tensor<i1>} : () -> ()

  // CHECK: "splatUIntTensor"() {bar = dense<222> : tensor<2x1x4xui8>} : () -> ()
  "splatUIntTensor"(){bar = dense<222> : tensor<2x1x4xui8>} : () -> ()

  // CHECK: "splatIntTensor"() {bar = dense<5> : tensor<2x1x4xi32>} : () -> ()
  "splatIntTensor"(){bar = dense<5> : tensor<2x1x4xi32>} : () -> ()

  // CHECK: "splatFloatTensor"() {bar = dense<-5.000000e+00> : tensor<2x1x4xf32>} : () -> ()
  "splatFloatTensor"(){bar = dense<-5.0> : tensor<2x1x4xf32>} : () -> ()

  // CHECK: "splatIntVector"() {bar = dense<5> : vector<2x1x4xi64>} : () -> ()
  "splatIntVector"(){bar = dense<5> : vector<2x1x4xi64>} : () -> ()

  // CHECK: "splatFloatVector"() {bar = dense<-5.000000e+00> : vector<2x1x4xf16>} : () -> ()
  "splatFloatVector"(){bar = dense<-5.0> : vector<2x1x4xf16>} : () -> ()

  // CHECK: "splatIntScalar"() {bar = dense<5> : tensor<i9>} : () -> ()
  "splatIntScalar"() {bar = dense<5> : tensor<i9>} : () -> ()
  // CHECK: "splatFloatScalar"() {bar = dense<-5.000000e+00> : tensor<f16>} : () -> ()
  "splatFloatScalar"() {bar = dense<-5.0> : tensor<f16>} : () -> ()
  return
}

// CHECK-LABEL: func @densetensorattr
func @densetensorattr() -> () {
^bb0:

// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi3"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi3>} : () -> ()
  "fooi3"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi3>} : () -> ()
// CHECK: "fooi6"() {bar = dense<{{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]> : tensor<2x1x4xi6>} : () -> ()
  "fooi6"(){bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : tensor<2x1x4xi6>} : () -> ()
// CHECK: "fooi8"() {bar = dense<5> : tensor<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = dense<[[[5]]]> : tensor<1x1x1xi8>} : () -> ()
// CHECK: "fooi13"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi13>} : () -> ()
  "fooi13"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi13>} : () -> ()
// CHECK: "fooi16"() {bar = dense<-5> : tensor<1x1x1xi16>} : () -> ()
  "fooi16"(){bar = dense<[[[-5]]]> : tensor<1x1x1xi16>} : () -> ()
// CHECK: "fooi23"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi23>} : () -> ()
  "fooi23"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi23>} : () -> ()
// CHECK: "fooi32"() {bar = dense<5> : tensor<1x1x1xi32>} : () -> ()
  "fooi32"(){bar = dense<[[[5]]]> : tensor<1x1x1xi32>} : () -> ()
// CHECK: "fooi33"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi33>} : () -> ()
  "fooi33"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi33>} : () -> ()
// CHECK: "fooi43"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi43>} : () -> ()
  "fooi43"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi43>} : () -> ()
// CHECK: "fooi53"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi53>} : () -> ()
  "fooi53"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi53>} : () -> ()
// CHECK: "fooi64"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 3, -1, 2]]]> : tensor<2x1x4xi64>} : () -> ()
  "fooi64"(){bar = dense<[[[1, -2, 1, 2]], [[0, 3, -1, 2]]]> : tensor<2x1x4xi64>} : () -> ()
// CHECK: "fooi64"() {bar = dense<-5> : tensor<1x1x1xi64>} : () -> ()
  "fooi64"(){bar = dense<[[[-5]]]> : tensor<1x1x1xi64>} : () -> ()
// CHECK: "fooi67"() {bar = dense<{{\[\[\[}}-5, 4, 6, 2]]]> : vector<1x1x4xi67>} : () -> ()
  "fooi67"(){bar = dense<[[[-5, 4, 6, 2]]]> : vector<1x1x4xi67>} : () -> ()

// CHECK: "foo2"() {bar = dense<> : tensor<0xi32>} : () -> ()
  "foo2"(){bar = dense<> : tensor<0xi32>} : () -> ()
// CHECK: "foo2"() {bar = dense<> : tensor<1x0xi32>} : () -> ()
  "foo2"(){bar = dense<> : tensor<1x0xi32>} : () -> ()
// CHECK: dense<> : tensor<0x512x512xi32>
  "foo2"(){bar = dense<> : tensor<0x512x512xi32>} : () -> ()
// CHECK: "foo3"() {bar = dense<{{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]> : tensor<2x1x4xi32>} : () -> ()
  "foo3"(){bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : tensor<2x1x4xi32>} : () -> ()

// CHECK: "float1"() {bar = dense<5.000000e+00> : tensor<1x1x1xf32>} : () -> ()
  "float1"(){bar = dense<[[[5.0]]]> : tensor<1x1x1xf32>} : () -> ()
// CHECK: "float2"() {bar = dense<> : tensor<0xf32>} : () -> ()
  "float2"(){bar = dense<> : tensor<0xf32>} : () -> ()
// CHECK: "float2"() {bar = dense<> : tensor<1x0xf32>} : () -> ()
  "float2"(){bar = dense<> : tensor<1x0xf32>} : () -> ()

// CHECK: "bfloat16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xbf16>} : () -> ()
  "bfloat16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xbf16>} : () -> ()
// CHECK: "float16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xf16>} : () -> ()
  "float16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xf16>} : () -> ()
// CHECK: "float32"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xf32>} : () -> ()
  "float32"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xf32>} : () -> ()
// CHECK: "float64"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xf64>} : () -> ()
  "float64"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xf64>} : () -> ()

// CHECK: "intscalar"() {bar = dense<1> : tensor<i32>} : () -> ()
  "intscalar"(){bar = dense<1> : tensor<i32>} : () -> ()
// CHECK: "floatscalar"() {bar = dense<5.000000e+00> : tensor<f32>} : () -> ()
  "floatscalar"(){bar = dense<5.0> : tensor<f32>} : () -> ()

// CHECK: "index"() {bar = dense<1> : tensor<index>} : () -> ()
  "index"(){bar = dense<1> : tensor<index>} : () -> ()
// CHECK: "index"() {bar = dense<[1, 2]> : tensor<2xindex>} : () -> ()
  "index"(){bar = dense<[1, 2]> : tensor<2xindex>} : () -> ()

  // CHECK: dense<(1,1)> : tensor<complex<i64>>
  "complex_attr"(){bar = dense<(1,1)> : tensor<complex<i64>>} : () -> ()
  // CHECK: dense<[(1,1), (2,2)]> : tensor<2xcomplex<i64>>
  "complex_attr"(){bar = dense<[(1,1), (2,2)]> : tensor<2xcomplex<i64>>} : () -> ()
  // CHECK: dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
  "complex_attr"(){bar = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>} : () -> ()
  // CHECK: dense<[(1.000000e+00,0.000000e+00), (2.000000e+00,2.000000e+00)]> : tensor<2xcomplex<f32>>
  "complex_attr"(){bar = dense<[(1.000000e+00,0.000000e+00), (2.000000e+00,2.000000e+00)]> : tensor<2xcomplex<f32>>} : () -> ()
  return
}

// CHECK-LABEL: func @densevectorattr
func @densevectorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar = dense<5> : vector<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = dense<[[[5]]]> : vector<1x1x1xi8>} : () -> ()
// CHECK: "fooi16"() {bar = dense<-5> : vector<1x1x1xi16>} : () -> ()
  "fooi16"(){bar = dense<[[[-5]]]> : vector<1x1x1xi16>} : () -> ()
// CHECK: "foo32"() {bar = dense<5> : vector<1x1x1xi32>} : () -> ()
  "foo32"(){bar = dense<[[[5]]]> : vector<1x1x1xi32>} : () -> ()
// CHECK: "fooi64"() {bar = dense<-5> : vector<1x1x1xi64>} : () -> ()
  "fooi64"(){bar = dense<[[[-5]]]> : vector<1x1x1xi64>} : () -> ()

// CHECK: "foo3"() {bar = dense<{{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]> : vector<2x1x4xi32>} : () -> ()
  "foo3"(){bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : vector<2x1x4xi32>} : () -> ()

// CHECK: "float1"() {bar = dense<5.000000e+00> : vector<1x1x1xf32>} : () -> ()
  "float1"(){bar = dense<[[[5.0]]]> : vector<1x1x1xf32>} : () -> ()

// CHECK: "bfloat16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xbf16>} : () -> ()
  "bfloat16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xbf16>} : () -> ()
// CHECK: "float16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xf16>} : () -> ()
  "float16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xf16>} : () -> ()
// CHECK: "float32"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xf32>} : () -> ()
  "float32"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xf32>} : () -> ()
// CHECK: "float64"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xf64>} : () -> ()
  "float64"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xf64>} : () -> ()
  return
}

// CHECK-LABEL: func @sparsetensorattr
func @sparsetensorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar = sparse<0, -2> : tensor<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = sparse<0, -2> : tensor<1x1x1xi8>} : () -> ()
// CHECK: "fooi16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2, -1, 5]> : tensor<2x2x2xi16>} : () -> ()
  "fooi16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]> : tensor<2x2x2xi16>} : () -> ()
// CHECK: "fooi32"() {bar = sparse<> : tensor<1x1xi32>} : () -> ()
  "fooi32"(){bar = sparse<> : tensor<1x1xi32>} : () -> ()
// CHECK: "fooi64"() {bar = sparse<0, -1> : tensor<1xi64>} : () -> ()
  "fooi64"(){bar = sparse<[0], [-1]> : tensor<1xi64>} : () -> ()
// CHECK: "foo2"() {bar = sparse<> : tensor<0xi32>} : () -> ()
  "foo2"(){bar = sparse<> : tensor<0xi32>} : () -> ()
// CHECK: "foo3"() {bar = sparse<> : tensor<i32>} : () -> ()
  "foo3"(){bar = sparse<> : tensor<i32>} : () -> ()

// CHECK: "foof16"() {bar = sparse<0, -2.000000e+00> : tensor<1x1x1xf16>} : () -> ()
  "foof16"(){bar = sparse<0, -2.0> : tensor<1x1x1xf16>} : () -> ()
// CHECK: "foobf16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2.000000e+00, -1.000000e+00, 5.000000e+00]> : tensor<2x2x2xbf16>} : () -> ()
  "foobf16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.0, -1.0, 5.0]> : tensor<2x2x2xbf16>} : () -> ()
// CHECK: "foof32"() {bar = sparse<> : tensor<1x0x1xf32>} : () -> ()
  "foof32"(){bar = sparse<> : tensor<1x0x1xf32>} : () -> ()
// CHECK:  "foof64"() {bar = sparse<0, -1.000000e+00> : tensor<1xf64>} : () -> ()
  "foof64"(){bar = sparse<[[0]], [-1.0]> : tensor<1xf64>} : () -> ()
// CHECK: "foof320"() {bar = sparse<> : tensor<0xf32>} : () -> ()
  "foof320"(){bar = sparse<> : tensor<0xf32>} : () -> ()
// CHECK: "foof321"() {bar = sparse<> : tensor<f32>} : () -> ()
  "foof321"(){bar = sparse<> : tensor<f32>} : () -> ()

// CHECK: "foostr"() {bar = sparse<0, "foo"> : tensor<1x1x1x!unknown<"">>} : () -> ()
  "foostr"(){bar = sparse<0, "foo"> : tensor<1x1x1x!unknown<"">>} : () -> ()
// CHECK: "foostr"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}"a", "b", "c"]> : tensor<2x2x2x!unknown<"">>} : () -> ()
  "foostr"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], ["a", "b", "c"]> : tensor<2x2x2x!unknown<"">>} : () -> ()
  return
}

// CHECK-LABEL: func @sparsevectorattr
func @sparsevectorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar = sparse<0, -2> : vector<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = sparse<0, -2> : vector<1x1x1xi8>} : () -> ()
// CHECK: "fooi16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2, -1, 5]> : vector<2x2x2xi16>} : () -> ()
  "fooi16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]> : vector<2x2x2xi16>} : () -> ()
// CHECK: "fooi32"() {bar = sparse<> : vector<1x1xi32>} : () -> ()
  "fooi32"(){bar = sparse<> : vector<1x1xi32>} : () -> ()
// CHECK: "fooi64"() {bar = sparse<0, -1> : vector<1xi64>} : () -> ()
  "fooi64"(){bar = sparse<[[0]], [-1]> : vector<1xi64>} : () -> ()

// CHECK: "foof16"() {bar = sparse<0, -2.000000e+00> : vector<1x1x1xf16>} : () -> ()
  "foof16"(){bar = sparse<0, -2.0> : vector<1x1x1xf16>} : () -> ()
// CHECK: "foobf16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2.000000e+00, -1.000000e+00, 5.000000e+00]> : vector<2x2x2xbf16>} : () -> ()
  "foobf16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.0, -1.0, 5.0]> : vector<2x2x2xbf16>} : () -> ()
// CHECK:  "foof64"() {bar = sparse<0, -1.000000e+00> : vector<1xf64>} : () -> ()
  "foof64"(){bar = sparse<0, [-1.0]> : vector<1xf64>} : () -> ()
  return
}

// CHECK-LABEL: func @unknown_dialect_type() -> !bar<""> {
func @unknown_dialect_type() -> !bar<""> {
  // Unregistered dialect 'bar'.
  // CHECK: "foo"() : () -> !bar<"">
  %0 = "foo"() : () -> !bar<"">

  // CHECK: "foo"() : () -> !bar.baz
  %1 = "foo"() : () -> !bar<"baz">

  return %0 : !bar<"">
}

// CHECK-LABEL: func @type_alias() -> i32 {
!i32_type_alias = type i32
func @type_alias() -> !i32_type_alias {

  // Return a non-aliased i32 type.
  %0 = "foo"() : () -> i32
  return %0 : i32
}

// CHECK-LABEL: func @no_integer_set_constraints(
func @no_integer_set_constraints() {
  // CHECK: affine.if [[$SET_TRUE]]() {
  affine.if affine_set<() : ()> () {
  }
  return
}

// CHECK-LABEL: func @verbose_if(
func @verbose_if(%N: index) {
  %c = arith.constant 200 : index

  // CHECK: affine.if #set{{.*}}(%{{.*}})[%{{.*}}, %{{.*}}] {
  "affine.if"(%c, %N, %c) ({
    // CHECK-NEXT: "add"
    %y = "add"(%c, %N) : (index, index) -> index
    "affine.yield"() : () -> ()
    // CHECK-NEXT: } else {
  }, { // The else region.
    // CHECK-NEXT: "add"
    %z = "add"(%c, %c) : (index, index) -> index
    "affine.yield"() : () -> ()
  })
  { condition = #set0 } : (index, index, index) -> ()
  return
}

// CHECK-LABEL: func @terminator_with_regions
func @terminator_with_regions() {
  // Combine successors and regions in the same operation.
  // CHECK: "region"()[^bb1] ( {
  // CHECK: }) : () -> ()
  "region"()[^bb2] ({}) : () -> ()
^bb2:
  return
}

// CHECK-LABEL: func @unregistered_term
func @unregistered_term(%arg0 : i1) -> i1 {
  // CHECK-NEXT: "unregistered_br"(%{{.*}})[^bb1] : (i1) -> ()
  "unregistered_br"(%arg0)[^bb1] : (i1) -> ()

^bb1(%arg1 : i1):
  return %arg1 : i1
}

// CHECK-LABEL: func @dialect_attrs
func @dialect_attrs()
    // CHECK: attributes  {dialect.attr = 10
    attributes {dialect.attr = 10} {
  return
}

// CHECK-LABEL: func private @_valid.function$name
func private @_valid.function$name()

// CHECK-LABEL: func private @external_func_arg_attrs(i32, i1 {dialect.attr = 10 : i64}, i32)
func private @external_func_arg_attrs(i32, i1 {dialect.attr = 10 : i64}, i32)

// CHECK-LABEL: func @func_arg_attrs(%{{.*}}: i1 {dialect.attr = 10 : i64})
func @func_arg_attrs(%arg0: i1 {dialect.attr = 10 : i64}) {
  return
}

// CHECK-LABEL: func @func_result_attrs({{.*}}) -> (f32 {dialect.attr = 1 : i64})
func @func_result_attrs(%arg0: f32) -> (f32 {dialect.attr = 1}) {
  return %arg0 : f32
}

// CHECK-LABEL: func private @empty_tuple(tuple<>)
func private @empty_tuple(tuple<>)

// CHECK-LABEL: func private @tuple_single_element(tuple<i32>)
func private @tuple_single_element(tuple<i32>)

// CHECK-LABEL: func private @tuple_multi_element(tuple<i32, i16, f32>)
func private @tuple_multi_element(tuple<i32, i16, f32>)

// CHECK-LABEL: func private @tuple_nested(tuple<tuple<tuple<i32>>>)
func private @tuple_nested(tuple<tuple<tuple<i32>>>)

// CHECK-LABEL: func @pretty_form_multi_result
func @pretty_form_multi_result() -> (i16, i16) {
  // CHECK: %{{.*}}:2 = "foo_div"() : () -> (i16, i16)
  %quot, %rem = "foo_div"() : () -> (i16, i16)
  return %quot, %rem : i16, i16
}

// CHECK-LABEL: func @pretty_form_multi_result_groups
func @pretty_form_multi_result_groups() -> (i16, i16, i16, i16, i16) {
  // CHECK: %[[RES:.*]]:5 =
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2, %[[RES]]#3, %[[RES]]#4
  %group_1:2, %group_2, %group_3:2 = "foo_test"() : () -> (i16, i16, i16, i16, i16)
  return %group_1#0, %group_1#1, %group_2, %group_3#0, %group_3#1 : i16, i16, i16, i16, i16
}

// CHECK-LABEL: func @pretty_dialect_attribute()
func @pretty_dialect_attribute() {
  // CHECK: "foo.unknown_op"() {foo = #foo.simple_attr} : () -> ()
  "foo.unknown_op"() {foo = #foo.simple_attr} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.complexattr<abcd>} : () -> ()
  "foo.unknown_op"() {foo = #foo.complexattr<abcd>} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.complexattr<abcd<f32>>} : () -> ()
  "foo.unknown_op"() {foo = #foo.complexattr<abcd<f32>>} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.complexattr<abcd<[f]$$[32]>>} : () -> ()
  "foo.unknown_op"() {foo = #foo.complexattr<abcd<[f]$$[32]>>} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.dialect<!x@#!@#>} : () -> ()
  "foo.unknown_op"() {foo = #foo.dialect<!x@#!@#>} : () -> ()

  // Extraneous extra > character can't use the pretty syntax.
  // CHECK: "foo.unknown_op"() {foo = #foo<"dialect<!x@#!@#>>">} : () -> ()
  "foo.unknown_op"() {foo = #foo<"dialect<!x@#!@#>>">} : () -> ()

  return
}

// CHECK-LABEL: func @pretty_dialect_type()
func @pretty_dialect_type() {

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.simpletype
  %0 = "foo.unknown_op"() : () -> !foo.simpletype

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.complextype<abcd>
  %1 = "foo.unknown_op"() : () -> !foo.complextype<abcd>

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.complextype<abcd<f32>>
  %2 = "foo.unknown_op"() : () -> !foo.complextype<abcd<f32>>

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.complextype<abcd<[f]$$[32]>>
  %3 = "foo.unknown_op"() : () -> !foo.complextype<abcd<[f]$$[32]>>

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.dialect<!x@#!@#>
  %4 = "foo.unknown_op"() : () -> !foo.dialect<!x@#!@#>

  // Extraneous extra > character can't use the pretty syntax.
  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo<"dialect<!x@#!@#>>">
  %5 = "foo.unknown_op"() : () -> !foo<"dialect<!x@#!@#>>">

  return
}

// CHECK-LABEL: func @none_type
func @none_type() {
  // CHECK: "foo.unknown_op"() : () -> none
  %none_val = "foo.unknown_op"() : () -> none
  return
}

// CHECK-LABEL: func @scoped_names
func @scoped_names() {
  // CHECK-NEXT: "foo.region_op"
  "foo.region_op"() ({
    // CHECK-NEXT: "foo.unknown_op"
    %scoped_name = "foo.unknown_op"() : () -> none
    "foo.terminator"() : () -> ()
  }, {
    // CHECK: "foo.unknown_op"
    %scoped_name = "foo.unknown_op"() : () -> none
    "foo.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @dialect_attribute_with_type
func @dialect_attribute_with_type() {
  // CHECK-NEXT: foo = #foo.attr : i32
  "foo.unknown_op"() {foo = #foo.attr : i32} : () -> ()
}

// CHECK-LABEL: @f16_special_values
func @f16_special_values() {
  // F16 NaNs.
  // CHECK: arith.constant 0x7C01 : f16
  %0 = arith.constant 0x7C01 : f16
  // CHECK: arith.constant 0x7FFF : f16
  %1 = arith.constant 0x7FFF : f16
  // CHECK: arith.constant 0xFFFF : f16
  %2 = arith.constant 0xFFFF : f16

  // F16 positive infinity.
  // CHECK: arith.constant 0x7C00 : f16
  %3 = arith.constant 0x7C00 : f16
  // F16 negative infinity.
  // CHECK: arith.constant 0xFC00 : f16
  %4 = arith.constant 0xFC00 : f16

  return
}

// CHECK-LABEL: @f32_special_values
func @f32_special_values() {
  // F32 signaling NaNs.
  // CHECK: arith.constant 0x7F800001 : f32
  %0 = arith.constant 0x7F800001 : f32
  // CHECK: arith.constant 0x7FBFFFFF : f32
  %1 = arith.constant 0x7FBFFFFF : f32

  // F32 quiet NaNs.
  // CHECK: arith.constant 0x7FC00000 : f32
  %2 = arith.constant 0x7FC00000 : f32
  // CHECK: arith.constant 0xFFFFFFFF : f32
  %3 = arith.constant 0xFFFFFFFF : f32

  // F32 positive infinity.
  // CHECK: arith.constant 0x7F800000 : f32
  %4 = arith.constant 0x7F800000 : f32
  // F32 negative infinity.
  // CHECK: arith.constant 0xFF800000 : f32
  %5 = arith.constant 0xFF800000 : f32

  return
}

// CHECK-LABEL: @f64_special_values
func @f64_special_values() {
  // F64 signaling NaNs.
  // CHECK: arith.constant 0x7FF0000000000001 : f64
  %0 = arith.constant 0x7FF0000000000001 : f64
  // CHECK: arith.constant 0x7FF8000000000000 : f64
  %1 = arith.constant 0x7FF8000000000000 : f64

  // F64 quiet NaNs.
  // CHECK: arith.constant 0x7FF0000001000000 : f64
  %2 = arith.constant 0x7FF0000001000000 : f64
  // CHECK: arith.constant 0xFFF0000001000000 : f64
  %3 = arith.constant 0xFFF0000001000000 : f64

  // F64 positive infinity.
  // CHECK: arith.constant 0x7FF0000000000000 : f64
  %4 = arith.constant 0x7FF0000000000000 : f64
  // F64 negative infinity.
  // CHECK: arith.constant 0xFFF0000000000000 : f64
  %5 = arith.constant 0xFFF0000000000000 : f64

  // Check that values that can't be represented with the default format, use
  // hex instead.
  // CHECK: arith.constant 0xC1CDC00000000000 : f64
  %6 = arith.constant 0xC1CDC00000000000 : f64

  return
}

// CHECK-LABEL: @bfloat16_special_values
func @bfloat16_special_values() {
  // bfloat16 signaling NaNs.
  // CHECK: arith.constant 0x7F81 : bf16
  %0 = arith.constant 0x7F81 : bf16
  // CHECK: arith.constant 0xFF81 : bf16
  %1 = arith.constant 0xFF81 : bf16

  // bfloat16 quiet NaNs.
  // CHECK: arith.constant 0x7FC0 : bf16
  %2 = arith.constant 0x7FC0 : bf16
  // CHECK: arith.constant 0xFFC0 : bf16
  %3 = arith.constant 0xFFC0 : bf16

  // bfloat16 positive infinity.
  // CHECK: arith.constant 0x7F80 : bf16
  %4 = arith.constant 0x7F80 : bf16
  // bfloat16 negative infinity.
  // CHECK: arith.constant 0xFF80 : bf16
  %5 = arith.constant 0xFF80 : bf16

  return
}

// We want to print floats in exponential notation with 6 significant digits,
// but it may lead to precision loss when parsing back, in which case we print
// the decimal form instead.
// CHECK-LABEL: @f32_potential_precision_loss()
func @f32_potential_precision_loss() {
  // CHECK: arith.constant -1.23697901 : f32
  %0 = arith.constant -1.23697901 : f32
  return
}

// CHECK-LABEL: @special_float_values_in_tensors
func @special_float_values_in_tensors() {
  // CHECK: dense<0xFFFFFFFF> : tensor<4x4xf32>
  "foo"(){bar = dense<0xFFFFFFFF> : tensor<4x4xf32>} : () -> ()
  // CHECK: dense<[{{\[}}0xFFFFFFFF, 0x7F800000], [0x7FBFFFFF, 0x7F800001]]> : tensor<2x2xf32>
  "foo"(){bar = dense<[[0xFFFFFFFF, 0x7F800000], [0x7FBFFFFF, 0x7F800001]]> : tensor<2x2xf32>} : () -> ()
  // CHECK: dense<[0xFFFFFFFF, 0.000000e+00]> : tensor<2xf32>
  "foo"(){bar = dense<[0xFFFFFFFF, 0.0]> : tensor<2xf32>} : () -> ()

  // CHECK: sparse<[{{\[}}1, 1, 0], [0, 1, 1]], [0xFFFFFFFF, 0x7F800001]>
  "foo"(){bar = sparse<[[1,1,0],[0,1,1]], [0xFFFFFFFF, 0x7F800001]> : tensor<2x2x2xf32>} : () -> ()
}

// Test parsing of an op with multiple region arguments, and without a
// delimiter.

// CHECK-LABEL: func @op_with_region_args
func @op_with_region_args() {
  // CHECK: "test.polyfor"() ( {
  // CHECK-NEXT: ^bb{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
  test.polyfor %i, %j, %k {
    "foo"() : () -> ()
  }
  return
}

// Test allowing different name scopes for regions isolated from above.

// CHECK-LABEL: func @op_with_passthrough_region_args
func @op_with_passthrough_region_args() {
  // CHECK: [[VAL:%.*]] = arith.constant
  %0 = arith.constant 10 : index

  // CHECK: test.isolated_region [[VAL]] {
  // CHECK-NEXT: "foo.consumer"([[VAL]]) : (index)
  // CHECK-NEXT: }
  test.isolated_region %0 {
    "foo.consumer"(%0) : (index) -> ()
  }

  // CHECK: [[VAL:%.*]]:2 = "foo.op"
  %result:2 = "foo.op"() : () -> (index, index)

  // CHECK: test.isolated_region [[VAL]]#1 {
  // CHECK-NEXT: "foo.consumer"([[VAL]]#1) : (index)
  // CHECK-NEXT: }
  test.isolated_region %result#1 {
    "foo.consumer"(%result#1) : (index) -> ()
  }

  return
}

// CHECK-LABEL: func private @ptr_to_function() -> !unreg.ptr<() -> ()>
func private @ptr_to_function() -> !unreg.ptr<() -> ()>

// CHECK-LABEL: func private @escaped_string_char(i1 {foo.value = "\0A"})
func private @escaped_string_char(i1 {foo.value = "\n"})

// CHECK-LABEL: func @parse_integer_literal_test
func @parse_integer_literal_test() {
  // CHECK: test.parse_integer_literal : 5
  test.parse_integer_literal : 5
  return
}

// CHECK-LABEL: func @parse_wrapped_keyword_test
func @parse_wrapped_keyword_test() {
  // CHECK: test.parse_wrapped_keyword foo.keyword
  test.parse_wrapped_keyword foo.keyword
  return
}

// CHECK-LABEL: func @"\22_string_symbol_reference\22"
func @"\"_string_symbol_reference\""() {
  // CHECK: ref = @"\22_string_symbol_reference\22"
  "foo.symbol_reference"() {ref = @"\"_string_symbol_reference\""} : () -> ()
  return
}

// CHECK-LABEL: func private @parse_opaque_attr_escape
func private @parse_opaque_attr_escape() {
    // CHECK: value = #foo<"\22escaped\\\0A\22">
    "foo.constant"() {value = #foo<"\"escaped\\\n\"">} : () -> ()
}

// CHECK-LABEL: func private @string_attr_name
// CHECK-SAME: {"0 . 0", nested = {"0 . 0"}}
func private @string_attr_name() attributes {"0 . 0", nested = {"0 . 0"}}

// CHECK-LABEL: func private @nested_reference
// CHECK: ref = @some_symbol::@some_nested_symbol
func private @nested_reference() attributes {test.ref = @some_symbol::@some_nested_symbol }

// CHECK-LABEL: func @custom_asm_names
func @custom_asm_names() -> (i32, i32, i32, i32, i32, i32, i32) {
  // CHECK: %[[FIRST:first.*]], %[[MIDDLE:middle_results.*]]:2, %[[LAST:[0-9]+]]
  %0, %1:2, %2 = "test.asm_interface_op"() : () -> (i32, i32, i32, i32)

  // CHECK: %[[FIRST_2:first.*]], %[[LAST_2:[0-9]+]]
  %3, %4 = "test.asm_interface_op"() : () -> (i32, i32)

  // CHECK: %[[RESULT:result.*]]
  %5 = "test.asm_dialect_interface_op"() : () -> (i32)

  // CHECK: return %[[FIRST]], %[[MIDDLE]]#0, %[[MIDDLE]]#1, %[[LAST]], %[[FIRST_2]], %[[LAST_2]]
  return %0, %1#0, %1#1, %2, %3, %4, %5 : i32, i32, i32, i32, i32, i32, i32
}


// CHECK-LABEL: func @pretty_names

// This tests the behavior
func @pretty_names() {
  // Simple case, should parse and print as %x being an implied 'name'
  // attribute.
  %x = test.string_attr_pretty_name
  // CHECK: %x = test.string_attr_pretty_name
  // CHECK-NOT: attributes

  // This specifies an explicit name, which should override the result.
  %YY = test.string_attr_pretty_name attributes { names = ["y"] }
  // CHECK: %y = test.string_attr_pretty_name
  // CHECK-NOT: attributes

  // Conflicts with the 'y' name, so need an explicit attribute.
  %0 = "test.string_attr_pretty_name"() { names = ["y"]} : () -> i32
  // CHECK: %y_0 = test.string_attr_pretty_name attributes {names = ["y"]}

  // Name contains a space.
  %1 = "test.string_attr_pretty_name"() { names = ["space name"]} : () -> i32
  // CHECK: %space_name = test.string_attr_pretty_name attributes {names = ["space name"]}

  "unknown.use"(%x, %YY, %0, %1) : (i32, i32, i32, i32) -> ()

  // Multi-result support.

  %a, %b, %c = test.string_attr_pretty_name
  // CHECK: %a, %b, %c = test.string_attr_pretty_name
  // CHECK-NOT: attributes

  %q:3, %r = test.string_attr_pretty_name
  // CHECK: %q, %q_1, %q_2, %r = test.string_attr_pretty_name attributes {names = ["q", "q", "q", "r"]}

  // CHECK: return
  return
}


// This tests the behavior of "default dialect":
// operations like `test.default_dialect` can define a default dialect
// used in nested region.
// CHECK-LABEL: func @default_dialect
func @default_dialect(%bool : i1) {
  test.default_dialect {
    // The test dialect is the default in this region, the following two
    // operations are parsed identically.
    // CHECK-NOT: test.parse_integer_literal
    parse_integer_literal : 5
    // CHECK: parse_integer_literal : 6
    test.parse_integer_literal : 6
    // Verify that only an op prefix is stripped, not an attribute value for
    // example.
    // CHECK:  "test.op_with_attr"() {test.attr = "test.value"} : () -> ()
    "test.op_with_attr"() {test.attr = "test.value"} : () -> ()

    // TODO: remove this after removing the special casing for std in the printer.
    // Verify that operations in the standard dialect keep the `std.` prefix.
    // CHECK: std.assert
    assert %bool, "Assertion"
    "test.terminator"() : ()->()
  }
  // The same operation outside of the region does not have an std. prefix.
  // CHECK-NOT: std.assert
  // CHECK: assert
  assert %bool, "Assertion"
  return
}

// CHECK-LABEL: func @unreachable_dominance_violation_ok
func @unreachable_dominance_violation_ok() -> i1 {
// CHECK:   [[VAL:%.*]] = arith.constant false
// CHECK:   return [[VAL]] : i1
// CHECK: ^bb1:   // no predecessors
// CHECK:   [[VAL2:%.*]]:3 = "bar"([[VAL3:%.*]]) : (i64) -> (i1, i1, i1)
// CHECK:   br ^bb3
// CHECK: ^bb2:   // pred: ^bb2
// CHECK:   br ^bb2
// CHECK: ^bb3:   // pred: ^bb1
// CHECK:   [[VAL3]] = "foo"() : () -> i64
// CHECK:   return [[VAL2]]#1 : i1
// CHECK: }
  %c = arith.constant false
  return %c : i1
^bb1:
  // %1 is not dominated by it's definition, but block is not reachable.
  %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
  br ^bb3
^bb2:
  br ^bb2
^bb3:
  %1 = "foo"() : ()->i64
  return %2#1 : i1
}

// CHECK-LABEL: func @graph_region_in_hierarchy_ok
func @graph_region_in_hierarchy_ok() -> i64 {
// CHECK:   br ^bb2
// CHECK: ^bb1:
// CHECK:   test.graph_region {
// CHECK:     [[VAL2:%.*]]:3 = "bar"([[VAL3:%.*]]) : (i64) -> (i1, i1, i1)
// CHECK:   }
// CHECK:   br ^bb3
// CHECK: ^bb2:   // pred: ^bb0
// CHECK:   [[VAL3]] = "foo"() : () -> i64
// CHECK:   br ^bb1
// CHECK: ^bb3:   // pred: ^bb1
// CHECK:   return [[VAL3]] : i64
// CHECK: }
  br ^bb2
^bb1:
  test.graph_region {
    // %1 is well-defined here, since bb2 dominates bb1.
    %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
  }
  br ^bb4
^bb2:
  %1 = "foo"() : ()->i64
  br ^bb1
^bb4:
  return %1 : i64
}

// CHECK-LABEL: func @graph_region_kind
func @graph_region_kind() -> () {
// CHECK: [[VAL2:%.*]]:3 = "bar"([[VAL3:%.*]]) : (i64) -> (i1, i1, i1)
// CHECK: [[VAL3]] = "baz"([[VAL2]]#0) : (i1) -> i64
  test.graph_region {
    // %1 OK here in in graph region.
    %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
    %1 = "baz"(%2#0) : (i1) -> (i64)
  }
  return
}

// CHECK-LABEL: func @graph_region_inside_ssacfg_region
func @graph_region_inside_ssacfg_region() -> () {
// CHECK: "test.ssacfg_region"
// CHECK:   [[VAL3:%.*]] = "baz"() : () -> i64
// CHECK:   test.graph_region {
// CHECK:     [[VAL2:%.*]]:3 = "bar"([[VAL3]]) : (i64) -> (i1, i1, i1)
// CHECK:   }
// CHECK:   [[VAL4:.*]] = "baz"() : () -> i64
  "test.ssacfg_region"() ({
    %1 = "baz"() : () -> (i64)
    test.graph_region {
      %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
    }
    %3 = "baz"() : () -> (i64)
  }) : () -> ()
  return
}

// CHECK-LABEL: func @graph_region_in_graph_region_ok
func @graph_region_in_graph_region_ok() -> () {
// CHECK: test.graph_region {
// CHECK:   test.graph_region {
// CHECK:     [[VAL2:%.*]]:3 = "bar"([[VAL3:%.*]]) : (i64) -> (i1, i1, i1)
// CHECK:   }
// CHECK:   [[VAL3]] = "foo"() : () -> i64
// CHECK: }
test.graph_region {
    test.graph_region {
    // %1 is well-defined here since defined in graph region
      %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
    }
    %1 = "foo"() : ()->i64
    "test.terminator"() : ()->()
  }
  return
}

// CHECK: test.graph_region {
test.graph_region {
// CHECK:   [[VAL1:%.*]] = "op1"([[VAL3:%.*]]) : (i32) -> i32
// CHECK:   [[VAL2:%.*]] = "test.ssacfg_region"([[VAL1]], [[VAL2]], [[VAL3]], [[VAL4:%.*]]) ( {
// CHECK:     [[VAL5:%.*]] = "op2"([[VAL1]], [[VAL2]], [[VAL3]], [[VAL4]]) : (i32, i32, i32, i32) -> i32
// CHECK:   }) : (i32, i32, i32, i32) -> i32
// CHECK:   [[VAL3]] = "op2"([[VAL1]], [[VAL4]]) : (i32, i32) -> i32
// CHECK:   [[VAL4]] = "op3"([[VAL1]]) : (i32) -> i32
  %1 = "op1"(%3) : (i32) -> (i32)
  %2 = "test.ssacfg_region"(%1, %2, %3, %4) ({
    %5 = "op2"(%1, %2, %3, %4) :
	 (i32, i32, i32, i32) -> (i32)
  }) : (i32, i32, i32, i32) -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)
  %4 = "op3"(%1) : (i32) -> (i32)
}

// CHECK: "unregistered_func_might_have_graph_region"() ( {
// CHECK: [[VAL1:%.*]] = "foo"([[VAL1]], [[VAL2:%.*]]) : (i64, i64) -> i64
// CHECK: [[VAL2]] = "bar"([[VAL1]])
"unregistered_func_might_have_graph_region"() ( {
  %1 = "foo"(%1, %2) : (i64, i64) -> i64
  %2 = "bar"(%1) : (i64) -> i64
  "unregistered_terminator"() : () -> ()
}) {sym_name = "unregistered_op_dominance_violation_ok", type = () -> i1} : () -> ()

// This is an unregister operation, the printing/parsing is handled by the dialect.
// CHECK: test.dialect_custom_printer custom_format
test.dialect_custom_printer custom_format

// This is a registered operation with no custom parser and printer, and should
// be handled by the dialect.
// CHECK: test.dialect_custom_format_fallback custom_format_fallback
test.dialect_custom_format_fallback custom_format_fallback
