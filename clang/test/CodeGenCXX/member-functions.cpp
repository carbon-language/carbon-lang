// RUN: clang-cc -emit-llvm %s -o %t &&
struct C {
  void f();
  void g(int, ...);
};

// RUN: grep "define void @_ZN1C1fEv" %t | count 1 &&
void C::f() {
}

void f() {
  C c;
  
// RUN: grep "call void @_ZN1C1fEv" %t | count 1 &&
  c.f();
  
// RUN: grep "call void (.struct.C\*, i32, ...)\* @_ZN1C1gEiz" %t | count 1
  c.g(1, 2, 3);
}
