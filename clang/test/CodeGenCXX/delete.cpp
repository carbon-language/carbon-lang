// RUN: clang-cc %s -emit-llvm -o %t &&

void t1(int *a) {
  delete a;
}

struct S {
  int a;
};

// POD types.
void t3(S *s) {
  delete s;
}

// Non-POD
struct T {
  ~T();
  int a;
};

void t4(T *t) {
  // RUN: grep "call void @_ZN1TD1Ev" %t | count 1
  delete t;
}

// PR5102
template <typename T>
class A {
  operator T *() const;
};

void f() {
  A<char*> a;
  
  delete a;
}
