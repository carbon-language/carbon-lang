// RUN: %clang_cc1 %s -std=c++1y -triple=i686-pc-win32 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s  -std=c++1y -triple=i686-pc-win32 -ftls-model=local-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-LD

struct A {
  A();
  ~A();
};

// CHECK-DAG: $"??$a@X@@3UA@@A" = comdat any
// CHECK-DAG: @"??$a@X@@3UA@@A" = linkonce_odr dso_local thread_local global %struct.A zeroinitializer, comdat, align 1
// CHECK-DAG: @"??__E?$a@X@@YAXXZ$initializer$" = internal constant void ()* @"??__E?$a@X@@YAXXZ", section ".CRT$XDU", comdat($"??$a@X@@3UA@@A")
// CHECK-LD-DAG: $"??$a@X@@3UA@@A" = comdat any
// CHECK-LD-DAG: @"??$a@X@@3UA@@A" = linkonce_odr dso_local thread_local(localdynamic) global %struct.A zeroinitializer, comdat, align 1
// CHECK-LD-DAG: @"??__E?$a@X@@YAXXZ$initializer$" = internal constant void ()* @"??__E?$a@X@@YAXXZ", section ".CRT$XDU", comdat($"??$a@X@@3UA@@A")
template <typename T>
thread_local A a = A();

// CHECK-DAG: @"?b@@3UA@@A" = dso_local thread_local global %struct.A zeroinitializer, align 1
// CHECK-DAG: @"__tls_init$initializer$" = internal constant void ()* @__tls_init, section ".CRT$XDU"
// CHECK-LD-DAG: @"?b@@3UA@@A" = dso_local thread_local(localdynamic) global %struct.A zeroinitializer, align 1
// CHECK-LD-DAG: @"__tls_init$initializer$" = internal constant void ()* @__tls_init, section ".CRT$XDU"
thread_local A b;

// CHECK-LABEL: define internal void @__tls_init()
// CHECK: call void @"??__Eb@@YAXXZ"
// CHECK-LD-LABEL: define internal void @__tls_init()
// CHECK-LD: call void @"??__Eb@@YAXXZ"

thread_local A &c = b;
thread_local A &d = c;

A f() {
  (void)a<void>;
  (void)b;
  return c;
}

// CHECK: !llvm.linker.options = !{![[dyn_tls_init:[0-9]+]]}
// CHECK: ![[dyn_tls_init]] = !{!"/include:___dyn_tls_init@12"}
// CHECK-LD: !llvm.linker.options = !{![[dyn_tls_init:[0-9]+]]}
// CHECK-LD: ![[dyn_tls_init]] = !{!"/include:___dyn_tls_init@12"}
