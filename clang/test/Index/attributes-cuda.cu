// RUN: c-index-test -test-load-source all -x cuda %s | FileCheck %s

__attribute__((device)) void f_device();
__attribute__((global)) void f_global();
__attribute__((constant)) int* g_constant;
__attribute__((shared)) float *g_shared;
__attribute__((host)) void f_host();

// CHECK:       attributes-cuda.cu:3:30: FunctionDecl=f_device:3:30
// CHECK-NEXT:  attributes-cuda.cu:3:16: attribute(device)
// CHECK:       attributes-cuda.cu:4:30: FunctionDecl=f_global:4:30
// CHECK-NEXT:  attributes-cuda.cu:4:16: attribute(global)
// CHECK:       attributes-cuda.cu:5:32: VarDecl=g_constant:5:32 (Definition)
// CHECK-NEXT:  attributes-cuda.cu:5:16: attribute(constant)
// CHECK:       attributes-cuda.cu:6:32: VarDecl=g_shared:6:32 (Definition)
// CHECK-NEXT:  attributes-cuda.cu:6:16: attribute(shared)
// CHECK:       attributes-cuda.cu:7:28: FunctionDecl=f_host:7:28
// CHECK-NEXT:  attributes-cuda.cu:7:16: attribute(host)
