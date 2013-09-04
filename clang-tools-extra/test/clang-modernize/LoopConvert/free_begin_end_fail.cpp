// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -loop-convert %t.cpp -- -I %S/Inputs -std=c++11
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *

struct MyArray {
  unsigned size();
};

template <typename T>
struct MyContainer {
};

int *begin(const MyArray &Arr);
int *end(const MyArray &Arr);

template <typename T>
T *begin(const MyContainer<T> &C);
template <typename T>
T *end(const MyContainer<T> &C);

// The Loop Convert Transform doesn't detect free functions begin()/end() and
// so fails to transform these cases which it should.
void f() {
  MyArray Arr;
  for (unsigned i = 0, e = Arr.size(); i < e; ++i) {}
  // CHECK: for (auto & elem : Arr) {}

  MyContainer<int> C;
  for (int *I = begin(C), *E = end(C); I != E; ++I) {}
  // CHECK: for (auto & elem : C) {}
}
