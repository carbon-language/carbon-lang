// RUN: %clang_cc1 -std=c++1z -verify %s

void use_from_own_init() {
  auto [a] = a; // expected-error {{binding 'a' cannot appear in the initializer of its own decomposition declaration}}
}

// As a Clang extension, _Complex can be decomposed.
float decompose_complex(_Complex float cf) {
  auto [re, im] = cf;
  //static_assert(&re == &__real cf);
  //static_assert(&im == &__imag cf);
  return re*re + im*im;
}

// As a Clang extension, vector types can be decomposed.
typedef float vf3 __attribute__((ext_vector_type(3)));
float decompose_vector(vf3 v) {
  auto [x, y, z] = v;
  auto *p = &x; // expected-error {{address of vector element requested}}
  return x + y + z;
}

// FIXME: by-value array copies
// FIXME: template instantiation
// FIXME: ast file support
// FIXME: code generation
// FIXME: constant expression evaluation
