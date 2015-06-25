// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s

class C {
  void present();
  void absent() __attribute__((nodebug));
};

C c;

// CHECK-NOT: name: "absent"
// CHECK:     name: "present"
// CHECK-NOT: name: "absent"
