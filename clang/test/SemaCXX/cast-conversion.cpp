// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

struct R {
  R(int);
};

struct A {
  A(R);
};

struct B { // expected-note 3 {{candidate constructor (the implicit copy constructor) not viable}} \
              expected-note 3 {{candidate constructor (the implicit move constructor) not viable}}
  B(A); // expected-note 3 {{candidate constructor not viable}}
};

int main () {
  B(10);	// expected-error {{no matching conversion for functional-style cast from 'int' to 'B'}}
  (B)10;	// expected-error {{no matching conversion for C-style cast from 'int' to 'B'}}
  static_cast<B>(10);	// expected-error {{no matching conversion for static_cast from 'int' to 'B'}} \\
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
  const char array[2] = { 'a', 'b' };
  make_X0(array);
}

// PR5210 recovery
class C {
protected:
  template <int> float* &f0(); // expected-note{{candidate}}
  template <unsigned> float* &f0(); // expected-note{{candidate}}

  void f1() {
    static_cast<float*>(f0<0>()); // expected-error{{ambiguous}}
  }
};
