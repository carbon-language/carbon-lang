// RUN: clang-cc -emit-llvm %s -o %t &&
struct C {
  void f();
};

// RUN: grep "define void @_ZN1C1fEv" %t | count 1 &&
void C::f() {
}

// RUN: grep "call void @_ZN1C1fEv" %t | count 1
void f() {
  C c;
  
  c.f();
}