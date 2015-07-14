// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -fsanitize=memory -fsanitize-memory-use-after-dtor -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsanitize=memory -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s -check-prefix=NO_DTOR_CHECK

struct Simple {
  ~Simple() {}
};
Simple s;
// Simple internal member is poisoned by compiler-generated dtor
// CHECK-LABEL: @_ZN6SimpleD2Ev
// CHECK: call void @__sanitizer_dtor_callback
// CHECK: ret void

// Compiling without the flag does not generate member-poisoning dtor
// NO_DTOR_CHECK-LABEL: @_ZN6SimpleD2Ev
// NO_DTOR_CHECK-NOT: call void @sanitizer_dtor_callback
// NO_DTOR_CHECK: ret void
