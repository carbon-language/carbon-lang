// RUN: %clang_cc1 -fsyntax-only -verify -Wvla %s
struct NonPOD {
  NonPOD();
};

struct NonPOD2 {
  NonPOD np;
};

struct POD {
  int x;
  int y;
};

// We allow VLAs of POD types, only.
void vla(int N) {
  int array1[N]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
  POD array2[N]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
  NonPOD array3[N]; // expected-error{{variable length array of non-POD element type 'NonPOD'}}
  NonPOD2 array4[N][3]; // expected-error{{variable length array of non-POD element type 'NonPOD2'}}
}

/// Warn about VLAs in templates.
template<typename T>
void vla_in_template(int N, T t) {
  int array1[N]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
}

struct HasConstantValue {
  static const unsigned int value = 2;
};

struct HasNonConstantValue {
  static unsigned int value;
};

template<typename T>
void vla_in_template(T t) {
  int array2[T::value]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
}

template void vla_in_template<HasConstantValue>(HasConstantValue);
template void vla_in_template<HasNonConstantValue>(HasNonConstantValue); // expected-note{{instantiation of}}

template<typename T> struct X0 { };

// Cannot use any variably-modified type with a template parameter or
// argument.
void inst_with_vla(int N) {
  int array[N]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
  X0<__typeof__(array)> x0a; // expected-error{{variably modified type 'typeof (array)' (aka 'int [N]') cannot be used as a template argument}}
}

template<typename T>
struct X1 {
  template<int (&Array)[T::value]> // expected-error{{non-type template parameter of variably modified type 'int (&)[HasNonConstantValue::value]'}}  \
  // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
  struct Inner {
    
  };
};

X1<HasConstantValue> x1a;
X1<HasNonConstantValue> x1b; // expected-note{{in instantiation of}}

// Template argument deduction does not allow deducing a size from a VLA.
template<typename T, unsigned N>
void accept_array(T (&array)[N]); // expected-note{{candidate template ignored: failed template argument deduction}}

void test_accept_array(int N) {
  int array[N]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
  accept_array(array); // expected-error{{no matching function for call to 'accept_array'}}
}

// Variably-modified types cannot be used in local classes.
void local_classes(int N) {
  struct X {
    int size;
    int array[N]; // expected-error{{fields must have a constant size: 'variable length array in structure' extension will never be supported}} \
                  // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
  };
}

namespace PR7206 {
  void f(int x) {
    struct edge_info {
      float left;
      float right;
    };
    struct edge_info edgeInfo[x]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
  }
}

namespace rdar8020206 {
  template<typename T>
  void f(int i) {
    const unsigned value = i;
    int array[value * i]; // expected-warning 2{{variable length arrays are a C99 feature, accepted as an extension}}
  }

  template void f<int>(int); // expected-note{{instantiation of}}
}

namespace rdar8021385 {
  typedef int my_int;
  struct A { typedef int my_int; };
  template<typename T>
  struct B {
    typedef typename T::my_int my_int;
    void f0() {
      int M = 4;
      my_int a[M]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
    }
  };
  B<A> a;
}

namespace PR8209 {
  void f(int n) {
    typedef int vla_type[n]; // expected-warning{{variable length arrays are a C99 feature, accepted as an extension}}
    (void)new vla_type; // expected-error{{variably}}
  }
}
