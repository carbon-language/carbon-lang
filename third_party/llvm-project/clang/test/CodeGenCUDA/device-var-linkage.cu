// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,NORDC %s
// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,RDC %s
// RUN: %clang_cc1 -triple nvptx \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,NORDC-H %s
// RUN: %clang_cc1 -triple nvptx \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,RDC-H %s

#include "Inputs/cuda.h"

// DEV-DAG: @v1 = addrspace(1) externally_initialized global i32 0
// NORDC-H-DAG: @v1 = internal global i32 undef
// RDC-H-DAG: @v1 = global i32 undef
__device__ int v1;
// DEV-DAG: @v2 = addrspace(4) externally_initialized global i32 0
// NORDC-H-DAG: @v2 = internal global i32 undef
// RDC-H-DAG: @v2 = global i32 undef
__constant__ int v2;
// DEV-DAG: @v3 = addrspace(1) externally_initialized global i32 addrspace(1)* null
// NORDC-H-DAG: @v3 = internal externally_initialized global i32* null
// RDC-H-DAG: @v3 = externally_initialized global i32* null
__managed__ int v3;

// DEV-DAG: @ev1 = external addrspace(1) global i32
// HOST-DAG: @ev1 = external global i32
extern __device__ int ev1;
// DEV-DAG: @ev2 = external addrspace(4) global i32
// HOST-DAG: @ev2 = external global i32
extern __constant__ int ev2;
// DEV-DAG: @ev3 = external addrspace(1) externally_initialized global i32 addrspace(1)*
// HOST-DAG: @ev3 = external externally_initialized global i32*
extern __managed__ int ev3;

// NORDC-DAG: @_ZL3sv1 = addrspace(1) externally_initialized global i32 0
// RDC-DAG: @_ZL3sv1__static__[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// HOST-DAG: @_ZL3sv1 = internal global i32 undef
static __device__ int sv1;
// NORDC-DAG: @_ZL3sv2 = addrspace(4) externally_initialized global i32 0
// RDC-DAG: @_ZL3sv2__static__[[HASH]] = addrspace(4) externally_initialized global i32 0
// HOST-DAG: @_ZL3sv2 = internal global i32 undef
static __constant__ int sv2;
// NORDC-DAG: @_ZL3sv3 = addrspace(1) externally_initialized global i32 addrspace(1)* null
// RDC-DAG: @_ZL3sv3__static__[[HASH]] = addrspace(1) externally_initialized global i32 addrspace(1)* null
// HOST-DAG: @_ZL3sv3 = internal externally_initialized global i32* null
static __managed__ int sv3;

__device__ __host__ int work(int *x);

__device__ __host__ int fun1() {
  return work(&ev1) + work(&ev2) + work(&ev3) + work(&sv1) + work(&sv2) + work(&sv3);
}

// HOST: hipRegisterVar({{.*}}@v1
// HOST: hipRegisterVar({{.*}}@v2
// HOST: hipRegisterManagedVar({{.*}}@v3
// HOST-NOT: hipRegisterVar({{.*}}@ev1
// HOST-NOT: hipRegisterVar({{.*}}@ev2
// HOST-NOT: hipRegisterManagedVar({{.*}}@ev3
// HOST: hipRegisterVar({{.*}}@_ZL3sv1
// HOST: hipRegisterVar({{.*}}@_ZL3sv2
// HOST: hipRegisterManagedVar({{.*}}@_ZL3sv3
