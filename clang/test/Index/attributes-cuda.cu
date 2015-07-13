// RUN: c-index-test -test-load-source all -x cuda %s | FileCheck %s
// RUN: c-index-test -test-load-source all -x cuda --cuda-host-only %s | FileCheck %s
// RUN: c-index-test -test-load-source all -x cuda --cuda-device-only %s | FileCheck %s

__attribute__((device)) void f_device();
__attribute__((global)) void f_global();
__attribute__((constant)) int* g_constant;
__attribute__((shared)) float *g_shared;
__attribute__((host)) void f_host();

// CHECK:       attributes-cuda.cu:5:30: FunctionDecl=f_device:5:30
// CHECK-NEXT:  attributes-cuda.cu:5:16: attribute(device)
// CHECK:       attributes-cuda.cu:6:30: FunctionDecl=f_global:6:30
// CHECK-NEXT:  attributes-cuda.cu:6:16: attribute(global)
// CHECK:       attributes-cuda.cu:7:32: VarDecl=g_constant:7:32 (Definition)
// CHECK-NEXT:  attributes-cuda.cu:7:16: attribute(constant)
// CHECK:       attributes-cuda.cu:8:32: VarDecl=g_shared:8:32 (Definition)
// CHECK-NEXT:  attributes-cuda.cu:8:16: attribute(shared)
// CHECK:       attributes-cuda.cu:9:28: FunctionDecl=f_host:9:28
// CHECK-NEXT:  attributes-cuda.cu:9:16: attribute(host)
