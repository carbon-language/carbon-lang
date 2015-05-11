// RUN: %clang_cc1 -emit-llvm %s -fcuda-include-gpubinary %s -o - | FileCheck %s

#include "Inputs/cuda.h"

// Make sure that all parts of GPU code init/cleanup are there:
// * constant unnamed string with the kernel name
// CHECK: private unnamed_addr constant{{.*}}kernelfunc{{.*}}\00"
// * constant unnamed string with GPU binary
// CHECK: private unnamed_addr constant{{.*}}\00"
// * constant struct that wraps GPU binary
// CHECK: @__cuda_fatbin_wrapper = internal constant { i32, i32, i8*, i8* } 
// CHECK:       { i32 1180844977, i32 1, {{.*}}, i8* null }
// * variable to save GPU binary handle after initialization
// CHECK: @__cuda_gpubin_handle = internal global i8** null
// * Make sure our constructor/destructor was added to global ctor/dtor list.
// CHECK: @llvm.global_ctors = appending global {{.*}}@__cuda_module_ctor
// CHECK: @llvm.global_dtors = appending global {{.*}}@__cuda_module_dtor

// Test that we build the correct number of calls to cudaSetupArgument followed
// by a call to cudaLaunch.

// CHECK: define{{.*}}kernelfunc
// CHECK: call{{.*}}cudaSetupArgument
// CHECK: call{{.*}}cudaSetupArgument
// CHECK: call{{.*}}cudaSetupArgument
// CHECK: call{{.*}}cudaLaunch
__global__ void kernelfunc(int i, int j, int k) {}

// Test that we've built correct kernel launch sequence.
// CHECK: define{{.*}}hostfunc
// CHECK: call{{.*}}cudaConfigureCall
// CHECK: call{{.*}}kernelfunc
void hostfunc(void) { kernelfunc<<<1, 1>>>(1, 1, 1); }

// Test that we've built a function to register kernels
// CHECK: define internal void @__cuda_register_kernels
// CHECK: call{{.*}}cudaRegisterFunction(i8** %0, {{.*}}kernelfunc

// Test that we've built contructor..
// CHECK: define internal void @__cuda_module_ctor
//   .. that calls __cudaRegisterFatBinary(&__cuda_fatbin_wrapper)
// CHECK: call{{.*}}cudaRegisterFatBinary{{.*}}__cuda_fatbin_wrapper
//   .. stores return value in __cuda_gpubin_handle
// CHECK-NEXT: store{{.*}}__cuda_gpubin_handle
//   .. and then calls __cuda_register_kernels
// CHECK-NEXT: call void @__cuda_register_kernels

// Test that we've created destructor.
// CHECK: define internal void @__cuda_module_dtor
// CHECK: load{{.*}}__cuda_gpubin_handle
// CHECK-NEXT: call void @__cudaUnregisterFatBinary

