// RUN: clang-cc -emit-llvm %s -o %t -triple=x86_64-apple-darwin9 && 
struct A {
  A();
  ~A();
  void f();
};

void f() {
  // RUN: grep "call void @_ZN1AC1Ev" %t | count 2 &&
  // RUN: grep "call void @_ZN1AD1Ev" %t | count 2
  A();
  A().f();
}
