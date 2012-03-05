// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

template <int dimm> struct Patch {
  static const unsigned int no_neighbor = 1;
};
template <int dim>
const unsigned int Patch<dim>::no_neighbor;
void f(const unsigned int);
void g() {
  f(Patch<1>::no_neighbor);
}
template struct Patch<1>;

// CHECK: _ZN5PatchILi1EE11no_neighborE
