// RUN: mlir-opt -split-input-file -convert-memref-to-spirv -canonicalize -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @alloc_dealloc_workgroup_mem(%arg0 : index, %arg1 : index) {
    %0 = memref.alloc() : memref<4x5xf32, 3>
    %1 = memref.load %0[%arg0, %arg1] : memref<4x5xf32, 3>
    memref.store %1, %0[%arg0, %arg1] : memref<4x5xf32, 3>
    memref.dealloc %0 : memref<4x5xf32, 3>
    return
  }
}
//     CHECK: spv.GlobalVariable @[[VAR:.+]] : !spv.ptr<!spv.struct<(!spv.array<20 x f32, stride=4>)>, Workgroup>
//     CHECK: func @alloc_dealloc_workgroup_mem
// CHECK-NOT:   memref.alloc
//     CHECK:   %[[PTR:.+]] = spv.mlir.addressof @[[VAR]]
//     CHECK:   %[[LOADPTR:.+]] = spv.AccessChain %[[PTR]]
//     CHECK:   %[[VAL:.+]] = spv.Load "Workgroup" %[[LOADPTR]] : f32
//     CHECK:   %[[STOREPTR:.+]] = spv.AccessChain %[[PTR]]
//     CHECK:   spv.Store "Workgroup" %[[STOREPTR]], %[[VAL]] : f32
// CHECK-NOT:   memref.dealloc

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @alloc_dealloc_workgroup_mem(%arg0 : index, %arg1 : index) {
    %0 = memref.alloc() : memref<4x5xi16, 3>
    %1 = memref.load %0[%arg0, %arg1] : memref<4x5xi16, 3>
    memref.store %1, %0[%arg0, %arg1] : memref<4x5xi16, 3>
    memref.dealloc %0 : memref<4x5xi16, 3>
    return
  }
}

//       CHECK: spv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
//  CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<20 x i32, stride=4>)>, Workgroup>
// CHECK_LABEL: spv.func @alloc_dealloc_workgroup_mem
//       CHECK:   %[[VAR:.+]] = spv.mlir.addressof @__workgroup_mem__0
//       CHECK:   %[[LOC:.+]] = spv.SDiv
//       CHECK:   %[[PTR:.+]] = spv.AccessChain %[[VAR]][%{{.+}}, %[[LOC]]]
//       CHECK:   %{{.+}} = spv.Load "Workgroup" %[[PTR]] : i32
//       CHECK:   %[[LOC:.+]] = spv.SDiv
//       CHECK:   %[[PTR:.+]] = spv.AccessChain %[[VAR]][%{{.+}}, %[[LOC]]]
//       CHECK:   %{{.+}} = spv.AtomicAnd "Workgroup" "AcquireRelease" %[[PTR]], %{{.+}} : !spv.ptr<i32, Workgroup>
//       CHECK:   %{{.+}} = spv.AtomicOr "Workgroup" "AcquireRelease" %[[PTR]], %{{.+}} : !spv.ptr<i32, Workgroup>


// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @two_allocs() {
    %0 = memref.alloc() : memref<4x5xf32, 3>
    %1 = memref.alloc() : memref<2x3xi32, 3>
    return
  }
}

//  CHECK-DAG: spv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<6 x i32, stride=4>)>, Workgroup>
//  CHECK-DAG: spv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<20 x f32, stride=4>)>, Workgroup>
//      CHECK: func @two_allocs()

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @two_allocs_vector() {
    %0 = memref.alloc() : memref<4xvector<4xf32>, 3>
    %1 = memref.alloc() : memref<2xvector<2xi32>, 3>
    return
  }
}

//  CHECK-DAG: spv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<2 x vector<2xi32>, stride=8>)>, Workgroup>
//  CHECK-DAG: spv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16>)>, Workgroup>
//      CHECK: func @two_allocs_vector()


// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @alloc_dealloc_dynamic_workgroup_mem(%arg0 : index) {
    // expected-error @+1 {{unhandled allocation type}}
    %0 = memref.alloc(%arg0) : memref<4x?xf32, 3>
    return
  }
}

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @alloc_dealloc_mem() {
    // expected-error @+1 {{unhandled allocation type}}
    %0 = memref.alloc() : memref<4x5xf32>
    return
  }
}


// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @alloc_dealloc_dynamic_workgroup_mem(%arg0 : memref<4x?xf32, 3>) {
    // expected-error @+1 {{unhandled deallocation type}}
    memref.dealloc %arg0 : memref<4x?xf32, 3>
    return
  }
}

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
  }
{
  func @alloc_dealloc_mem(%arg0 : memref<4x5xf32>) {
    // expected-error @+1 {{unhandled deallocation type}}
    memref.dealloc %arg0 : memref<4x5xf32>
    return
  }
}
