// Ensure compatiblity of UBSan unreachable with ASan in the presence of
// noreturn functions
// RUN: %clang -O2 -fsanitize=address,unreachable %s -emit-llvm -S -o - | FileCheck %s
// REQUIRES: ubsan-asan

void bar(void) __attribute__((noreturn));

void foo() {
  bar();
}
// CHECK-LABEL: define void @foo()
// CHECK:       call void @__asan_handle_no_return
// CHECK-NEXT:  call void @bar
// CHECK-NEXT:  call void @__asan_handle_no_return
// CHECK-NEXT:  call void @__ubsan_handle_builtin_unreachable
// CHECK-NEXT:  unreachable
