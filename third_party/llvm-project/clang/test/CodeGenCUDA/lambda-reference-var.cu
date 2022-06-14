// RUN: %clang_cc1 -no-opaque-pointers -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple x86_64-linux-gnu \
// RUN:   | FileCheck -check-prefix=HOST %s
// RUN: %clang_cc1 -no-opaque-pointers -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   | FileCheck -check-prefix=DEV %s

#include "Inputs/cuda.h"

// HOST: %[[T1:.*]] = type <{ i32*, i32, [4 x i8] }>
// HOST: %[[T2:.*]] = type { i32*, i32** }
// HOST: %[[T3:.*]] = type <{ i32*, i32, [4 x i8] }>
// DEV: %[[T1:.*]] = type { i32* }
// DEV: %[[T2:.*]] = type { i32** }
// DEV: %[[T3:.*]] = type <{ i32*, i32, [4 x i8] }>
int global_host_var;
__device__ int global_device_var;

template<class F>
__global__ void kern(F f) { f(); }

// DEV-LABEL: @_ZZ27dev_capture_dev_ref_by_copyPiENKUlvE_clEv(
// DEV: %[[VAL:.*]] = load i32, i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: store i32 %[[VAL]]
__device__ void dev_capture_dev_ref_by_copy(int *out) {
  int &ref = global_device_var;
  [=](){ *out = ref;}();
}

// DEV-LABEL: @_ZZ28dev_capture_dev_rval_by_copyPiENKUlvE_clEv(
// DEV: store i32 3
__device__ void dev_capture_dev_rval_by_copy(int *out) {
  constexpr int a = 1;
  constexpr int b = 2;
  constexpr int c = a + b;
  [=](){ *out = c;}();
}

// DEV-LABEL: @_ZZ26dev_capture_dev_ref_by_refPiENKUlvE_clEv(
// DEV: %[[VAL:.*]] = load i32, i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// DEV: store i32 %[[VAL2]], i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: %[[VAL:.*]] = load i32, i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: store i32 %[[VAL]]
__device__ void dev_capture_dev_ref_by_ref(int *out) {
  int &ref = global_device_var;
  [&](){ ref++; *out = ref;}();
}

// DEV-LABEL: define{{.*}} void @_Z7dev_refPi(
// DEV: %[[VAL:.*]] = load i32, i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// DEV: store i32 %[[VAL2]], i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: %[[VAL:.*]] = load i32, i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: store i32 %[[VAL]]
__device__ void dev_ref(int *out) {
  int &ref = global_device_var;
  ref++;
  *out = ref;
}

// DEV-LABEL: @_ZZ14dev_lambda_refPiENKUlvE_clEv(
// DEV: %[[VAL:.*]] = load i32, i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// DEV: store i32 %[[VAL2]], i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: %[[VAL:.*]] = load i32, i32* addrspacecast (i32 addrspace(1)* @global_device_var to i32*)
// DEV: store i32 %[[VAL]]
__device__ void dev_lambda_ref(int *out) {
  [=](){
    int &ref = global_device_var;
    ref++;
    *out = ref;
  }();
}

// HOST-LABEL: @_ZZ29host_capture_host_ref_by_copyPiENKUlvE_clEv(
// HOST: %[[VAL:.*]] = load i32, i32* @global_host_var
// HOST: store i32 %[[VAL]]
void host_capture_host_ref_by_copy(int *out) {
  int &ref = global_host_var;
  [=](){ *out = ref;}();
}

// HOST-LABEL: @_ZZ28host_capture_host_ref_by_refPiENKUlvE_clEv(
// HOST: %[[CAP:.*]] = getelementptr inbounds %[[T2]], %[[T2]]* %this1, i32 0, i32 0
// HOST: %[[REF:.*]] = load i32*, i32** %[[CAP]]
// HOST: %[[VAL:.*]] = load i32, i32* %[[REF]]
// HOST: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// HOST: store i32 %[[VAL2]], i32* %[[REF]]
// HOST: %[[VAL:.*]] = load i32, i32* @global_host_var
// HOST: store i32 %[[VAL]]
void host_capture_host_ref_by_ref(int *out) {
  int &ref = global_host_var;
  [&](){ ref++; *out = ref;}();
}

// HOST-LABEL: define{{.*}} void @_Z8host_refPi(
// HOST: %[[VAL:.*]] = load i32, i32* @global_host_var
// HOST: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// HOST: store i32 %[[VAL2]], i32* @global_host_var
// HOST: %[[VAL:.*]] = load i32, i32* @global_host_var
// HOST: store i32 %[[VAL]]
void host_ref(int *out) {
  int &ref = global_host_var;
  ref++;
  *out = ref;
}

// HOST-LABEL: @_ZZ15host_lambda_refPiENKUlvE_clEv(
// HOST: %[[VAL:.*]] = load i32, i32* @global_host_var
// HOST: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// HOST: store i32 %[[VAL2]], i32* @global_host_var
// HOST: %[[VAL:.*]] = load i32, i32* @global_host_var
// HOST: store i32 %[[VAL]]
void host_lambda_ref(int *out) {
  [=](){
    int &ref = global_host_var;
    ref++;
    *out = ref;
  }();
}

// HOST-LABEL: define{{.*}} void @_Z28dev_capture_host_ref_by_copyPi(
// HOST: %[[CAP:.*]] = getelementptr inbounds %[[T3]], %[[T3]]* %{{.*}}, i32 0, i32 1
// HOST: %[[VAL:.*]] = load i32, i32* @global_host_var
// HOST: store i32 %[[VAL]], i32* %[[CAP]]
// DEV-LABEL: define internal void @_ZZ28dev_capture_host_ref_by_copyPiENKUlvE_clEv(
// DEV: %[[CAP:.*]] = getelementptr inbounds %[[T3]], %[[T3]]* %this1, i32 0, i32 1
// DEV: %[[VAL:.*]] = load i32, i32* %[[CAP]]
// DEV: store i32 %[[VAL]]
void dev_capture_host_ref_by_copy(int *out) {
  int &ref = global_host_var;
  kern<<<1, 1>>>([=]__device__() { *out = ref;});
}

