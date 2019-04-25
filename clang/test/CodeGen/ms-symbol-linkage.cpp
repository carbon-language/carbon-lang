// RUN: %clangxx -target aarch64-windows \
// RUN: -fcxx-exceptions -c -o - %s \
// RUN: | llvm-objdump -syms - 2>&1 | FileCheck %s

void foo1() { throw 1; }
// CHECK-LABEL: foo1
// CHECK-NOT: __imp__CxxThrowException

void bar();
void foo2() noexcept(true) { bar(); }
// CHECK-LABEL: foo2
// CHECK-NOT: __imp___std_terminate

struct A {};
struct B { virtual void f(); };
struct C : A, virtual B {};
struct T {};
T *foo3() { return dynamic_cast<T *>((C *)0); }
// CHECK-LABEL: foo3
// CHECK-NOT: __imp___RTDynamicCast
