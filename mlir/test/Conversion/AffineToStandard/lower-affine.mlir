// RUN: mlir-opt -lower-affine %s | FileCheck %s

// CHECK-LABEL: func @empty() {
func @empty() {
  return     // CHECK:  return
}            // CHECK: }

func private @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: func @simple_loop
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1_0:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c1]] to %[[c42]] step %[[c1_0]] {
// CHECK-NEXT:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @simple_loop() {
  affine.for %i = 1 to 42 {
    call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @for_with_yield(%buffer: memref<1024xf32>) -> (f32) {
  %sum_0 = constant 0.0 : f32
  %sum = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = addf %sum_iter, %t : f32
    affine.yield %sum_next : f32
  }
  return %sum : f32
}

// CHECK-LABEL: func @for_with_yield
// CHECK:         %[[INIT_SUM:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[LOWER:.*]] = constant 0 : index
// CHECK-NEXT:    %[[UPPER:.*]] = constant 10 : index
// CHECK-NEXT:    %[[STEP:.*]] = constant 2 : index
// CHECK-NEXT:    %[[SUM:.*]] = scf.for %[[IV:.*]] = %[[LOWER]] to %[[UPPER]] step %[[STEP]] iter_args(%[[SUM_ITER:.*]] = %[[INIT_SUM]]) -> (f32) {
// CHECK-NEXT:      memref.load
// CHECK-NEXT:      %[[SUM_NEXT:.*]] = addf
// CHECK-NEXT:      scf.yield %[[SUM_NEXT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[SUM]] : f32

/////////////////////////////////////////////////////////////////////

func private @pre(index) -> ()
func private @body2(index, index) -> ()
func private @post(index) -> ()

// CHECK-LABEL: func @imperfectly_nested_loops
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:     %[[c56:.*]] = constant 56 : index
// CHECK-NEXT:     %[[c2:.*]] = constant 2 : index
// CHECK-NEXT:     for %{{.*}} = %[[c7]] to %[[c56]] step %[[c2]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
      call @body2(%i, %j) : (index, index) -> ()
    }
    call @post(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func private @mid(index) -> ()
func private @body3(index, index) -> ()

// CHECK-LABEL: func @more_imperfectly_nested_loops
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:     %[[c56:.*]] = constant 56 : index
// CHECK-NEXT:     %[[c2:.*]] = constant 2 : index
// CHECK-NEXT:     for %{{.*}} = %[[c7]] to %[[c56]] step %[[c2]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @mid(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c18:.*]] = constant 18 : index
// CHECK-NEXT:     %[[c37:.*]] = constant 37 : index
// CHECK-NEXT:     %[[c3:.*]] = constant 3 : index
// CHECK-NEXT:     for %{{.*}} = %[[c18]] to %[[c37]] step %[[c3]] {
// CHECK-NEXT:       call @body3(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @more_imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
      call @body2(%i, %j) : (index, index) -> ()
    }
    call @mid(%i) : (index) -> ()
    affine.for %k = 18 to 37 step 3 {
      call @body3(%i, %k) : (index, index) -> ()
    }
    call @post(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @affine_apply_loops_shorthand
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %{{.*}} step %[[c1]] {
// CHECK-NEXT:     %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:     %[[c1_0:.*]] = constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %{{.*}} to %[[c42]] step %[[c1_0]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @affine_apply_loops_shorthand(%N : index) {
  affine.for %i = 0 to %N {
    affine.for %j = affine_map<(d0)[]->(d0)>(%i)[] to 42 {
      call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

/////////////////////////////////////////////////////////////////////

func private @get_idx() -> (index)

#set1 = affine_set<(d0) : (20 - d0 >= 0)>
#set2 = affine_set<(d0) : (d0 - 10 >= 0)>

// CHECK-LABEL: func @if_only
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_only() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    call @body(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_else
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   } else {
// CHECK-NEXT:     call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_else() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @nested_ifs
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     %[[c0_0:.*]] = constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]] = constant -10 : index
// CHECK-NEXT:     %[[v4:.*]] = addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:     %[[v5:.*]] = cmpi sge, %[[v4]], %[[c0_0]] : index
// CHECK-NEXT:     if %[[v5]] {
// CHECK-NEXT:       call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[c0_0:.*]] = constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]] = constant -10 : index
// CHECK-NEXT:     %{{.*}} = addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:     %{{.*}} = cmpi sge, %{{.*}}, %[[c0_0]] : index
// CHECK-NEXT:     if %{{.*}} {
// CHECK-NEXT:       call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @nested_ifs() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    affine.if #set2(%i) {
      call @body(%i) : (index) -> ()
    }
  } else {
    affine.if #set2(%i) {
      call @mid(%i) : (index) -> ()
    }
  }
  return
}

// CHECK-LABEL: func @if_with_yield
// CHECK-NEXT:   %[[c0_i64:.*]] = constant 0 : i64
// CHECK-NEXT:   %[[c1_i64:.*]] = constant 1 : i64
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm10:.*]] = constant -10 : index
// CHECK-NEXT:   %[[v1:.*]] = addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:   %[[v2:.*]] = cmpi sge, %[[v1]], %[[c0]] : index
// CHECK-NEXT:   %[[v3:.*]] = scf.if %[[v2]] -> (i64) {
// CHECK-NEXT:     scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %[[c1_i64]] : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[v3]] : i64
// CHECK-NEXT: }
func @if_with_yield() -> (i64) {
  %cst0 = constant 0 : i64
  %cst1 = constant 1 : i64
  %i = call @get_idx() : () -> (index)
  %1 = affine.if #set2(%i) -> (i64) {
      affine.yield %cst0 : i64
  } else {
      affine.yield %cst1 : i64
  }
  return %1 : i64
}

#setN = affine_set<(d0)[N,M,K,L] : (N - d0 + 1 >= 0, N - 1 >= 0, M - 1 >= 0, K - 1 >= 0, L - 42 == 0)>

// CHECK-LABEL: func @multi_cond
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %{{.*}} : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[v3:.*]] = addi %[[v2]], %[[c1]] : index
// CHECK-NEXT:   %[[v4:.*]] = cmpi sge, %[[v3]], %[[c0]] : index
// CHECK-NEXT:   %[[cm1_0:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v5:.*]] = addi %{{.*}}, %[[cm1_0]] : index
// CHECK-NEXT:   %[[v6:.*]] = cmpi sge, %[[v5]], %[[c0]] : index
// CHECK-NEXT:   %[[v7:.*]] = and %[[v4]], %[[v6]] : i1
// CHECK-NEXT:   %[[cm1_1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v8:.*]] = addi %{{.*}}, %[[cm1_1]] : index
// CHECK-NEXT:   %[[v9:.*]] = cmpi sge, %[[v8]], %[[c0]] : index
// CHECK-NEXT:   %[[v10:.*]] = and %[[v7]], %[[v9]] : i1
// CHECK-NEXT:   %[[cm1_2:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v11:.*]] = addi %{{.*}}, %[[cm1_2]] : index
// CHECK-NEXT:   %[[v12:.*]] = cmpi sge, %[[v11]], %[[c0]] : index
// CHECK-NEXT:   %[[v13:.*]] = and %[[v10]], %[[v12]] : i1
// CHECK-NEXT:   %[[cm42:.*]] = constant -42 : index
// CHECK-NEXT:   %[[v14:.*]] = addi %{{.*}}, %[[cm42]] : index
// CHECK-NEXT:   %[[v15:.*]] = cmpi eq, %[[v14]], %[[c0]] : index
// CHECK-NEXT:   %[[v16:.*]] = and %[[v13]], %[[v15]] : i1
// CHECK-NEXT:   if %[[v16]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   } else {
// CHECK-NEXT:     call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @multi_cond(%N : index, %M : index, %K : index, %L : index) {
  %i = call @get_idx() : () -> (index)
  affine.if #setN(%i)[%N,%M,%K,%L] {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_for
func @if_for() {
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
  %i = call @get_idx() : () -> (index)
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi sge, %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     %[[c0:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:     %[[c42:.*]]{{.*}} = constant 42 : index
// CHECK-NEXT:     %[[c1:.*]]{{.*}} = constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %[[c0:.*]]{{.*}} to %[[c42:.*]]{{.*}} step %[[c1:.*]]{{.*}} {
// CHECK-NEXT:       %[[c0_:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:       %[[cm10:.*]] = constant -10 : index
// CHECK-NEXT:       %[[v4:.*]] = addi %{{.*}}, %[[cm10]] : index
// CHECK-NEXT:       %[[v5:.*]] = cmpi sge, %[[v4]], %[[c0_:.*]]{{.*}} : index
// CHECK-NEXT:       if %[[v5]] {
// CHECK-NEXT:         call @body2(%[[v0]], %{{.*}}) : (index, index) -> ()
  affine.if #set1(%i) {
    affine.for %j = 0 to 42 {
      affine.if #set2(%j) {
        call @body2(%i, %j) : (index, index) -> ()
      }
    }
  }
//      CHECK:   %[[c0:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]]{{.*}} = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]]{{.*}} = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0:.*]]{{.*}} to %[[c42:.*]]{{.*}} step %[[c1:.*]]{{.*}} {
// CHECK-NEXT:     %[[c0:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]]{{.*}} = constant -10 : index
// CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %[[cm10:.*]]{{.*}} : index
// CHECK-NEXT:     %{{.*}} = cmpi sge, %{{.*}}, %[[c0:.*]]{{.*}} : index
// CHECK-NEXT:     if %{{.*}} {
// CHECK-NEXT:       %[[c0_:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:       %[[c42_:.*]]{{.*}} = constant 42 : index
// CHECK-NEXT:       %[[c1_:.*]]{{.*}} = constant 1 : index
// CHECK-NEXT:       for %{{.*}} = %[[c0_:.*]]{{.*}} to %[[c42_:.*]]{{.*}} step %[[c1_:.*]]{{.*}} {
  affine.for %k = 0 to 42 {
    affine.if #set2(%k) {
      affine.for %l = 0 to 42 {
        call @body3(%k, %l) : (index, index) -> ()
      }
    }
  }
//      CHECK:   return
  return
}

#lbMultiMap = affine_map<(d0)[s0] -> (d0, s0 - d0)>
#ubMultiMap = affine_map<(d0)[s0] -> (s0, d0 + 10)>

// CHECK-LABEL: func @loop_min_max
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:     %[[a:.*]] = muli %{{.*}}, %[[cm1]] : index
// CHECK-NEXT:     %[[b:.*]] = addi %[[a]], %{{.*}} : index
// CHECK-NEXT:     %[[c:.*]] = cmpi sgt, %{{.*}}, %[[b]] : index
// CHECK-NEXT:     %[[d:.*]] = select %[[c]], %{{.*}}, %[[b]] : index
// CHECK-NEXT:     %[[c10:.*]] = constant 10 : index
// CHECK-NEXT:     %[[e:.*]] = addi %{{.*}}, %[[c10]] : index
// CHECK-NEXT:     %[[f:.*]] = cmpi slt, %{{.*}}, %[[e]] : index
// CHECK-NEXT:     %[[g:.*]] = select %[[f]], %{{.*}}, %[[e]] : index
// CHECK-NEXT:     %[[c1_0:.*]] = constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %[[d]] to %[[g]] step %[[c1_0]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @loop_min_max(%N : index) {
  affine.for %i = 0 to 42 {
    affine.for %j = max #lbMultiMap(%i)[%N] to min #ubMultiMap(%i)[%N] {
      call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

#map_7_values = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>

// Check that the "min" (cmpi slt + select) reduction sequence is emitted
// correctly for an affine map with 7 results.

// CHECK-LABEL: func @min_reduction_tree
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c01:.+]] = cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[r01:.+]] = select %[[c01]], %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[c012:.+]] = cmpi slt, %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[r012:.+]] = select %[[c012]], %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123:.+]] = cmpi slt, %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123:.+]] = select %[[c0123]], %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[c01234:.+]] = cmpi slt, %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[r01234:.+]] = select %[[c01234]], %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[c012345:.+]] = cmpi slt, %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[r012345:.+]] = select %[[c012345]], %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123456:.+]] = cmpi slt, %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123456:.+]] = select %[[c0123456]], %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[r0123456]] step %[[c1]] {
// CHECK-NEXT:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @min_reduction_tree(%v1 : index, %v2 : index, %v3 : index, %v4 : index, %v5 : index, %v6 : index, %v7 : index) {
  affine.for %i = 0 to min #map_7_values(%v1, %v2, %v3, %v4, %v5, %v6, %v7)[] {
    call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0 + 1)>
#map4 = affine_map<(d0,d1,d2,d3)[s0,s1,s2] -> (d0 + 2*d1 + 3*d2 + 4*d3 + 5*s0 + 6*s1 + 7*s2)>
#map5 = affine_map<(d0,d1,d2) -> (d0,d1,d2)>
#map6 = affine_map<(d0,d1,d2) -> (d0 + d1 + d2)>

// CHECK-LABEL: func @affine_applies(
func @affine_applies(%arg0 : index) {
// CHECK: %[[c0:.*]] = constant 0 : index
  %zero = affine.apply #map0()

// Identity maps are just discarded.
// CHECK-NEXT: %[[c101:.*]] = constant 101 : index
  %101 = constant 101 : index
  %symbZero = affine.apply #map1()[%zero]
// CHECK-NEXT: %[[c102:.*]] = constant 102 : index
  %102 = constant 102 : index
  %copy = affine.apply #map2(%zero)

// CHECK-NEXT: %[[v0:.*]] = addi %[[c0]], %[[c0]] : index
// CHECK-NEXT: %[[c1:.*]] = constant 1 : index
// CHECK-NEXT: %[[v1:.*]] = addi %[[v0]], %[[c1]] : index
  %one = affine.apply #map3(%symbZero)[%zero]

// CHECK-NEXT: %[[c2:.*]] = constant 2 : index
// CHECK-NEXT: %[[v2:.*]] = muli %arg0, %[[c2]] : index
// CHECK-NEXT: %[[v3:.*]] = addi %arg0, %[[v2]] : index
// CHECK-NEXT: %[[c3:.*]] = constant 3 : index
// CHECK-NEXT: %[[v4:.*]] = muli %arg0, %[[c3]] : index
// CHECK-NEXT: %[[v5:.*]] = addi %[[v3]], %[[v4]] : index
// CHECK-NEXT: %[[c4:.*]] = constant 4 : index
// CHECK-NEXT: %[[v6:.*]] = muli %arg0, %[[c4]] : index
// CHECK-NEXT: %[[v7:.*]] = addi %[[v5]], %[[v6]] : index
// CHECK-NEXT: %[[c5:.*]] = constant 5 : index
// CHECK-NEXT: %[[v8:.*]] = muli %arg0, %[[c5]] : index
// CHECK-NEXT: %[[v9:.*]] = addi %[[v7]], %[[v8]] : index
// CHECK-NEXT: %[[c6:.*]] = constant 6 : index
// CHECK-NEXT: %[[v10:.*]] = muli %arg0, %[[c6]] : index
// CHECK-NEXT: %[[v11:.*]] = addi %[[v9]], %[[v10]] : index
// CHECK-NEXT: %[[c7:.*]] = constant 7 : index
// CHECK-NEXT: %[[v12:.*]] = muli %arg0, %[[c7]] : index
// CHECK-NEXT: %[[v13:.*]] = addi %[[v11]], %[[v12]] : index
  %four = affine.apply #map4(%arg0, %arg0, %arg0, %arg0)[%arg0, %arg0, %arg0]
  return
}

// CHECK-LABEL: func @args_ret_affine_apply(
func @args_ret_affine_apply(index, index) -> (index, index) {
^bb0(%0 : index, %1 : index):
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, index
  %00 = affine.apply #map2 (%0)
  %11 = affine.apply #map1 ()[%1]
  return %00, %11 : index, index
}

//===---------------------------------------------------------------------===//
// Test lowering of Euclidean (floor) division, ceil division and modulo
// operation used in affine expressions.  In addition to testing the
// operation-level output, check that the obtained results are correct by
// applying constant folding transformation after affine lowering.
//===---------------------------------------------------------------------===//

#mapmod = affine_map<(i) -> (i mod 42)>

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_mod
func @affine_apply_mod(%arg0 : index) -> (index) {
// CHECK-NEXT: %[[c42:.*]] = constant 42 : index
// CHECK-NEXT: %[[v0:.*]] = remi_signed %{{.*}}, %[[c42]] : index
// CHECK-NEXT: %[[c0:.*]] = constant 0 : index
// CHECK-NEXT: %[[v1:.*]] = cmpi slt, %[[v0]], %[[c0]] : index
// CHECK-NEXT: %[[v2:.*]] = addi %[[v0]], %[[c42]] : index
// CHECK-NEXT: %[[v3:.*]] = select %[[v1]], %[[v2]], %[[v0]] : index
  %0 = affine.apply #mapmod (%arg0)
  return %0 : index
}

#mapfloordiv = affine_map<(i) -> (i floordiv 42)>

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_floordiv
func @affine_apply_floordiv(%arg0 : index) -> (index) {
// CHECK-NEXT: %[[c42:.*]] = constant 42 : index
// CHECK-NEXT: %[[c0:.*]] = constant 0 : index
// CHECK-NEXT: %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT: %[[v0:.*]] = cmpi slt, %{{.*}}, %[[c0]] : index
// CHECK-NEXT: %[[v1:.*]] = subi %[[cm1]], %{{.*}} : index
// CHECK-NEXT: %[[v2:.*]] = select %[[v0]], %[[v1]], %{{.*}} : index
// CHECK-NEXT: %[[v3:.*]] = divi_signed %[[v2]], %[[c42]] : index
// CHECK-NEXT: %[[v4:.*]] = subi %[[cm1]], %[[v3]] : index
// CHECK-NEXT: %[[v5:.*]] = select %[[v0]], %[[v4]], %[[v3]] : index
  %0 = affine.apply #mapfloordiv (%arg0)
  return %0 : index
}

#mapceildiv = affine_map<(i) -> (i ceildiv 42)>

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_ceildiv
func @affine_apply_ceildiv(%arg0 : index) -> (index) {
// CHECK-NEXT:  %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:  %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:  %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:  %[[v0:.*]] = cmpi sle, %{{.*}}, %[[c0]] : index
// CHECK-NEXT:  %[[v1:.*]] = subi %[[c0]], %{{.*}} : index
// CHECK-NEXT:  %[[v2:.*]] = subi %{{.*}}, %[[c1]] : index
// CHECK-NEXT:  %[[v3:.*]] = select %[[v0]], %[[v1]], %[[v2]] : index
// CHECK-NEXT:  %[[v4:.*]] = divi_signed %[[v3]], %[[c42]] : index
// CHECK-NEXT:  %[[v5:.*]] = subi %[[c0]], %[[v4]] : index
// CHECK-NEXT:  %[[v6:.*]] = addi %[[v4]], %[[c1]] : index
// CHECK-NEXT:  %[[v7:.*]] = select %[[v0]], %[[v5]], %[[v6]] : index
  %0 = affine.apply #mapceildiv (%arg0)
  return %0 : index
}

// CHECK-LABEL: func @affine_load
func @affine_load(%arg0 : index) {
  %0 = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %1 = affine.load %0[%i0 + symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %[[a:.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:  %[[b:.*]] = addi %[[a]], %[[c7]] : index
// CHECK-NEXT:  %{{.*}} = memref.load %[[v0:.*]][%[[b]]] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_store
func @affine_store(%arg0 : index) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = constant 11.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %1, %0[%i0 - symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %c-1 = constant -1 : index
// CHECK-NEXT:  %[[a:.*]] = muli %arg0, %c-1 : index
// CHECK-NEXT:  %[[b:.*]] = addi %{{.*}}, %[[a]] : index
// CHECK-NEXT:  %c7 = constant 7 : index
// CHECK-NEXT:  %[[c:.*]] = addi %[[b]], %c7 : index
// CHECK-NEXT:  store %cst, %0[%[[c]]] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_load_store_zero_dim
func @affine_load_store_zero_dim(%arg0 : memref<i32>, %arg1 : memref<i32>) {
  %0 = affine.load %arg0[] : memref<i32>
  affine.store %0, %arg1[] : memref<i32>
// CHECK: %[[x:.*]] = memref.load %arg0[] : memref<i32>
// CHECK: store %[[x]], %arg1[] : memref<i32>
  return
}

// CHECK-LABEL: func @affine_prefetch
func @affine_prefetch(%arg0 : index) {
  %0 = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.prefetch %0[%i0 + symbol(%arg0) + 7], read, locality<3>, data : memref<10xf32>
  }
// CHECK:       %[[a:.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:  %[[b:.*]] = addi %[[a]], %[[c7]] : index
// CHECK-NEXT:  memref.prefetch %[[v0:.*]][%[[b]]], read, locality<3>, data : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_dma_start
func @affine_dma_start(%arg0 : index) {
  %0 = memref.alloc() : memref<100xf32>
  %1 = memref.alloc() : memref<100xf32, 2>
  %2 = memref.alloc() : memref<1xi32>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_start %0[%i0 + 7], %1[%arg0 + 11], %2[%c0], %c64
        : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  }
// CHECK:       %c7 = constant 7 : index
// CHECK-NEXT:  %[[a:.*]] = addi %{{.*}}, %c7 : index
// CHECK-NEXT:  %c11 = constant 11 : index
// CHECK-NEXT:  %[[b:.*]] = addi %arg0, %c11 : index
// CHECK-NEXT:  dma_start %0[%[[a]]], %1[%[[b]]], %c64, %2[%c0] : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  return
}

// CHECK-LABEL: func @affine_dma_wait
func @affine_dma_wait(%arg0 : index) {
  %2 = memref.alloc() : memref<1xi32>
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_wait %2[%i0 + %arg0 + 17], %c64 : memref<1xi32>
  }
// CHECK:       %[[a:.*]] = addi %{{.*}}, %arg0 : index
// CHECK-NEXT:  %c17 = constant 17 : index
// CHECK-NEXT:  %[[b:.*]] = addi %[[a]], %c17 : index
// CHECK-NEXT:  dma_wait %0[%[[b]]], %c64 : memref<1xi32>
  return
}

// CHECK-LABEL: func @affine_min
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func @affine_min(%arg0: index, %arg1: index) -> index{
  // CHECK: %[[Cm1:.*]] = constant -1
  // CHECK: %[[neg1:.*]] = muli %[[ARG1]], %[[Cm1:.*]]
  // CHECK: %[[first:.*]] = addi %[[ARG0]], %[[neg1]]
  // CHECK: %[[Cm2:.*]] = constant -1
  // CHECK: %[[neg2:.*]] = muli %[[ARG0]], %[[Cm2:.*]]
  // CHECK: %[[second:.*]] = addi %[[ARG1]], %[[neg2]]
  // CHECK: %[[cmp:.*]] = cmpi slt, %[[first]], %[[second]]
  // CHECK: select %[[cmp]], %[[first]], %[[second]]
  %0 = affine.min affine_map<(d0,d1) -> (d0 - d1, d1 - d0)>(%arg0, %arg1)
  return %0 : index
}

// CHECK-LABEL: func @affine_max
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func @affine_max(%arg0: index, %arg1: index) -> index{
  // CHECK: %[[Cm1:.*]] = constant -1
  // CHECK: %[[neg1:.*]] = muli %[[ARG1]], %[[Cm1:.*]]
  // CHECK: %[[first:.*]] = addi %[[ARG0]], %[[neg1]]
  // CHECK: %[[Cm2:.*]] = constant -1
  // CHECK: %[[neg2:.*]] = muli %[[ARG0]], %[[Cm2:.*]]
  // CHECK: %[[second:.*]] = addi %[[ARG1]], %[[neg2]]
  // CHECK: %[[cmp:.*]] = cmpi sgt, %[[first]], %[[second]]
  // CHECK: select %[[cmp]], %[[first]], %[[second]]
  %0 = affine.max affine_map<(d0,d1) -> (d0 - d1, d1 - d0)>(%arg0, %arg1)
  return %0 : index
}

// CHECK-LABEL: func @affine_parallel(
// CHECK-SAME: %[[ARG0:.*]]: memref<100x100xf32>, %[[ARG1:.*]]: memref<100x100xf32>) {
func @affine_parallel(%o: memref<100x100xf32>, %a: memref<100x100xf32>) {
  affine.parallel (%i, %j) = (0, 0) to (100, 100) {
  }
  return
}

// CHECK-DAG:    %[[C100:.*]] = constant 100
// CHECK-DAG:    %[[C100_1:.*]] = constant 100
// CHECK-DAG:    %[[C0:.*]] = constant 0
// CHECK-DAG:    %[[C0_1:.*]] = constant 0
// CHECK-DAG:    %[[C1:.*]] = constant 1
// CHECK-DAG:    %[[C1_1:.*]] = constant 1
// CHECK-DAG:    scf.parallel (%arg2, %arg3) = (%[[C0]], %[[C0_1]]) to (%[[C100]], %[[C100_1]]) step (%[[C1]], %[[C1_1]]) {

// CHECK-LABEL: func @affine_parallel_tiled(
// CHECK-SAME: %[[ARG0:.*]]: memref<100x100xf32>, %[[ARG1:.*]]: memref<100x100xf32>, %[[ARG2:.*]]: memref<100x100xf32>) {
func @affine_parallel_tiled(%o: memref<100x100xf32>, %a: memref<100x100xf32>, %b: memref<100x100xf32>) {
  affine.parallel (%i0, %j0, %k0) = (0, 0, 0) to (100, 100, 100) step (10, 10, 10) {
    affine.parallel (%i1, %j1, %k1) = (%i0, %j0, %k0) to (%i0 + 10, %j0 + 10, %k0 + 10) {
      %0 = affine.load %a[%i1, %k1] : memref<100x100xf32>
      %1 = affine.load %b[%k1, %j1] : memref<100x100xf32>
      %2 = mulf %0, %1 : f32
    }
  }
  return
}

// CHECK-DAG:     %[[C100:.*]] = constant 100
// CHECK-DAG:     %[[C100_0:.*]] = constant 100
// CHECK-DAG:     %[[C100_1:.*]] = constant 100
// CHECK-DAG:     %[[C0:.*]] = constant 0
// CHECK-DAG:     %[[C0_2:.*]] = constant 0
// CHECK-DAG:     %[[C0_3:.*]] = constant 0
// CHECK-DAG:     %[[C10:.*]] = constant 10
// CHECK-DAG:     %[[C10_4:.*]] = constant 10
// CHECK-DAG:     %[[C10_5:.*]] = constant 10
// CHECK:         scf.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]]) = (%[[C0]], %[[C0_2]], %[[C0_3]]) to (%[[C100]], %[[C100_0]], %[[C100_1]]) step (%[[C10]], %[[C10_4]], %[[C10_5]]) {
// CHECK-DAG:       %[[C10_6:.*]] = constant 10
// CHECK-DAG:       %[[A0:.*]] = addi %[[arg3]], %[[C10_6]]
// CHECK-DAG:       %[[C10_7:.*]] = constant 10
// CHECK-DAG:       %[[A1:.*]] = addi %[[arg4]], %[[C10_7]]
// CHECK-DAG:       %[[C10_8:.*]] = constant 10
// CHECK-DAG:       %[[A2:.*]] = addi %[[arg5]], %[[C10_8]]
// CHECK-DAG:       %[[C1:.*]] = constant 1
// CHECK-DAG:       %[[C1_9:.*]] = constant 1
// CHECK-DAG:       %[[C1_10:.*]] = constant 1
// CHECK:           scf.parallel (%[[arg6:.*]], %[[arg7:.*]], %[[arg8:.*]]) = (%[[arg3]], %[[arg4]], %[[arg5]]) to (%[[A0]], %[[A1]], %[[A2]]) step (%[[C1]], %[[C1_9]], %[[C1_10]]) {
// CHECK:             %[[A3:.*]] = memref.load %[[ARG1]][%[[arg6]], %[[arg8]]] : memref<100x100xf32>
// CHECK:             %[[A4:.*]] = memref.load %[[ARG2]][%[[arg8]], %[[arg7]]] : memref<100x100xf32>
// CHECK:             mulf %[[A3]], %[[A4]] : f32
// CHECK:             scf.yield

/////////////////////////////////////////////////////////////////////

func @affine_parallel_simple(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> (memref<3x3xf32>) {
  %O = memref.alloc() : memref<3x3xf32>
  affine.parallel (%kx, %ky) = (0, 0) to (2, 2) {
      %1 = affine.load %arg0[%kx, %ky] : memref<3x3xf32>
      %2 = affine.load %arg1[%kx, %ky] : memref<3x3xf32>
      %3 = mulf %1, %2 : f32
      affine.store %3, %O[%kx, %ky] : memref<3x3xf32>
  }
  return %O : memref<3x3xf32>
}
// CHECK-LABEL: func @affine_parallel_simple
// CHECK:         %[[LOWER_1:.*]] = constant 0 : index
// CHECK-NEXT:    %[[UPPER_1:.*]] = constant 2 : index
// CHECK-NEXT:    %[[LOWER_2:.*]] = constant 0 : index
// CHECK-NEXT:    %[[UPPER_2:.*]] = constant 2 : index
// CHECK-NEXT:    %[[STEP_1:.*]] = constant 1 : index
// CHECK-NEXT:    %[[STEP_2:.*]] = constant 1 : index
// CHECK-NEXT:    scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) {
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.load
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.load
// CHECK-NEXT:      %[[PRODUCT:.*]] = mulf
// CHECK-NEXT:      store
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////

func @affine_parallel_simple_dynamic_bounds(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %c_0 = constant 0 : index
  %output_dim = memref.dim %arg0, %c_0 : memref<?x?xf32>
  affine.parallel (%kx, %ky) = (%c_0, %c_0) to (%output_dim, %output_dim) {
      %1 = affine.load %arg0[%kx, %ky] : memref<?x?xf32>
      %2 = affine.load %arg1[%kx, %ky] : memref<?x?xf32>
      %3 = mulf %1, %2 : f32
      affine.store %3, %arg2[%kx, %ky] : memref<?x?xf32>
  }
  return
}
// CHECK-LABEL: func @affine_parallel_simple_dynamic_bounds
// CHECK-SAME:  %[[ARG_0:.*]]: memref<?x?xf32>, %[[ARG_1:.*]]: memref<?x?xf32>, %[[ARG_2:.*]]: memref<?x?xf32>
// CHECK:         %[[DIM_INDEX:.*]] = constant 0 : index
// CHECK-NEXT:    %[[UPPER:.*]] = memref.dim %[[ARG_0]], %[[DIM_INDEX]] : memref<?x?xf32>
// CHECK-NEXT:    %[[LOWER_1:.*]] = constant 0 : index
// CHECK-NEXT:    %[[LOWER_2:.*]] = constant 0 : index
// CHECK-NEXT:    %[[STEP_1:.*]] = constant 1 : index
// CHECK-NEXT:    %[[STEP_2:.*]] = constant 1 : index
// CHECK-NEXT:    scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER]], %[[UPPER]]) step (%[[STEP_1]], %[[STEP_2]]) {
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.load
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.load
// CHECK-NEXT:      %[[PRODUCT:.*]] = mulf
// CHECK-NEXT:      store
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////

func @affine_parallel_with_reductions(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> (f32, f32) {
  %0:2 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf", "mulf") -> (f32, f32) {
            %1 = affine.load %arg0[%kx, %ky] : memref<3x3xf32>
            %2 = affine.load %arg1[%kx, %ky] : memref<3x3xf32>
            %3 = mulf %1, %2 : f32
            %4 = addf %1, %2 : f32
            affine.yield %3, %4 : f32, f32
          }
  return %0#0, %0#1 : f32, f32
}
// CHECK-LABEL: func @affine_parallel_with_reductions
// CHECK:         %[[LOWER_1:.*]] = constant 0 : index
// CHECK-NEXT:    %[[UPPER_1:.*]] = constant 2 : index
// CHECK-NEXT:    %[[LOWER_2:.*]] = constant 0 : index
// CHECK-NEXT:    %[[UPPER_2:.*]] = constant 2 : index
// CHECK-NEXT:    %[[STEP_1:.*]] = constant 1 : index
// CHECK-NEXT:    %[[STEP_2:.*]] = constant 1 : index
// CHECK-NEXT:    %[[INIT_1:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[INIT_2:.*]] = constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[RES:.*]] = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) init (%[[INIT_1]], %[[INIT_2]]) -> (f32, f32) {
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.load
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.load
// CHECK-NEXT:      %[[PRODUCT:.*]] = mulf
// CHECK-NEXT:      %[[SUM:.*]] = addf
// CHECK-NEXT:      scf.reduce(%[[PRODUCT]]) : f32 {
// CHECK-NEXT:      ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
// CHECK-NEXT:        %[[RES:.*]] = addf
// CHECK-NEXT:        scf.reduce.return %[[RES]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.reduce(%[[SUM]]) : f32 {
// CHECK-NEXT:      ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
// CHECK-NEXT:        %[[RES:.*]] = mulf
// CHECK-NEXT:        scf.reduce.return %[[RES]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

/////////////////////////////////////////////////////////////////////

func @affine_parallel_with_reductions_f64(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>) -> (f64, f64) {
  %0:2 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf", "mulf") -> (f64, f64) {
            %1 = affine.load %arg0[%kx, %ky] : memref<3x3xf64>
            %2 = affine.load %arg1[%kx, %ky] : memref<3x3xf64>
            %3 = mulf %1, %2 : f64
            %4 = addf %1, %2 : f64
            affine.yield %3, %4 : f64, f64
          }
  return %0#0, %0#1 : f64, f64
}
// CHECK-LABEL: @affine_parallel_with_reductions_f64
// CHECK:  %[[LOWER_1:.*]] = constant 0 : index
// CHECK:  %[[UPPER_1:.*]] = constant 2 : index
// CHECK:  %[[LOWER_2:.*]] = constant 0 : index
// CHECK:  %[[UPPER_2:.*]] = constant 2 : index
// CHECK:  %[[STEP_1:.*]] = constant 1 : index
// CHECK:  %[[STEP_2:.*]] = constant 1 : index
// CHECK:  %[[INIT_1:.*]] = constant 0.000000e+00 : f64
// CHECK:  %[[INIT_2:.*]] = constant 1.000000e+00 : f64
// CHECK:  %[[RES:.*]] = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) init (%[[INIT_1]], %[[INIT_2]]) -> (f64, f64) {
// CHECK:    %[[VAL_1:.*]] = memref.load
// CHECK:    %[[VAL_2:.*]] = memref.load
// CHECK:    %[[PRODUCT:.*]] = mulf
// CHECK:    %[[SUM:.*]] = addf
// CHECK:    scf.reduce(%[[PRODUCT]]) : f64 {
// CHECK:    ^bb0(%[[LHS:.*]]: f64, %[[RHS:.*]]: f64):
// CHECK:      %[[RES:.*]] = addf
// CHECK:      scf.reduce.return %[[RES]] : f64
// CHECK:    }
// CHECK:    scf.reduce(%[[SUM]]) : f64 {
// CHECK:    ^bb0(%[[LHS:.*]]: f64, %[[RHS:.*]]: f64):
// CHECK:      %[[RES:.*]] = mulf
// CHECK:      scf.reduce.return %[[RES]] : f64
// CHECK:    }
// CHECK:    scf.yield
// CHECK:  }

/////////////////////////////////////////////////////////////////////

func @affine_parallel_with_reductions_i64(%arg0: memref<3x3xi64>, %arg1: memref<3x3xi64>) -> (i64, i64) {
  %0:2 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addi", "muli") -> (i64, i64) {
            %1 = affine.load %arg0[%kx, %ky] : memref<3x3xi64>
            %2 = affine.load %arg1[%kx, %ky] : memref<3x3xi64>
            %3 = muli %1, %2 : i64
            %4 = addi %1, %2 : i64
            affine.yield %3, %4 : i64, i64
          }
  return %0#0, %0#1 : i64, i64
}
// CHECK-LABEL: @affine_parallel_with_reductions_i64
// CHECK:  %[[LOWER_1:.*]] = constant 0 : index
// CHECK:  %[[UPPER_1:.*]] = constant 2 : index
// CHECK:  %[[LOWER_2:.*]] = constant 0 : index
// CHECK:  %[[UPPER_2:.*]] = constant 2 : index
// CHECK:  %[[STEP_1:.*]] = constant 1 : index
// CHECK:  %[[STEP_2:.*]] = constant 1 : index
// CHECK:  %[[INIT_1:.*]] = constant 0 : i64
// CHECK:  %[[INIT_2:.*]] = constant 1 : i64
// CHECK:  %[[RES:.*]] = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[LOWER_1]], %[[LOWER_2]]) to (%[[UPPER_1]], %[[UPPER_2]]) step (%[[STEP_1]], %[[STEP_2]]) init (%[[INIT_1]], %[[INIT_2]]) -> (i64, i64) {
// CHECK:    %[[VAL_1:.*]] = memref.load
// CHECK:    %[[VAL_2:.*]] = memref.load
// CHECK:    %[[PRODUCT:.*]] = muli
// CHECK:    %[[SUM:.*]] = addi
// CHECK:    scf.reduce(%[[PRODUCT]]) : i64 {
// CHECK:    ^bb0(%[[LHS:.*]]: i64, %[[RHS:.*]]: i64):
// CHECK:      %[[RES:.*]] = addi
// CHECK:      scf.reduce.return %[[RES]] : i64
// CHECK:    }
// CHECK:    scf.reduce(%[[SUM]]) : i64 {
// CHECK:    ^bb0(%[[LHS:.*]]: i64, %[[RHS:.*]]: i64):
// CHECK:      %[[RES:.*]] = muli
// CHECK:      scf.reduce.return %[[RES]] : i64
// CHECK:    }
// CHECK:    scf.yield
// CHECK:  }
