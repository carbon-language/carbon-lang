// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

struct Simple {
  int x_;
  Simple() {
    x_ = 5;
  }
  ~Simple() {
    x_ += 1;
  }
};

Simple s;
// Simple internal member is poisoned by compiler-generated dtor
// CHECK-LABEL: define {{.*}}SimpleD2Ev
// CHECK: {{\s*}}call void @__sanitizer_dtor_callback
// CHECK-NOT: {{\s*}}call void @__sanitizer_dtor_callback
// CHECK-NOT: {{\s*}}tail call void @__sanitizer_dtor_callback
// CHECK: ret void
