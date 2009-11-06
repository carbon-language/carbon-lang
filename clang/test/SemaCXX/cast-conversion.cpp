// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct R {
  R(int);
};

struct A {
  A(R);
};

struct B {
  B(A);
};

int main () {
  B(10);	// expected-error {{functional-style cast from 'int' to 'struct B' is not allowed}}
  (B)10;	// expected-error {{C-style cast from 'int' to 'struct B' is not allowed}}
  static_cast<B>(10);	// expected-error {{static_cast from 'int' to 'struct B' is not allowed}} \\
			// expected-warning {{expression result unused}}
}

template<class T>
struct X0 {
  X0(const T &);
};

template<class T>
X0<T> make_X0(const T &Val) {
  return X0<T>(Val);
}

void test_X0() {
  const char array[2];
  make_X0(array);
}
