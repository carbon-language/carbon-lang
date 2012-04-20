// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

class A {
 public:
  A() { }
  ~A() { }
};

void no_contstructor_destructor_infinite_recursion() {
  A a;

// Make sure that the constructor doesn't call itself:
// CHECK: define {{.*}} @"\01??0A@@QAE@XZ"
// CHECK-NOT: call void @"\01??0A@@QAE@XZ"
// CHECK: ret

// Make sure that the destructor doesn't call itself:
// CHECK: define {{.*}} @"\01??1A@@QAE@XZ"
// CHECK-NOT: call void @"\01??1A@@QAE@XZ"
// CHECK: ret
}
