// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wvla-extension %s
struct StillPOD {
  StillPOD() = default;
};

struct StillPOD2 {
  StillPOD np;
};

struct NonPOD {
  NonPOD(int) {}
};

struct POD {
  int x;
  int y;
};

// We allow VLAs of POD types, only.
void vla(int N) {
  int array1[N]; // expected-warning{{variable length arrays are a C99 feature}}
  POD array2[N]; // expected-warning{{variable length arrays are a C99 feature}}
  StillPOD array3[N]; // expected-warning{{variable length arrays are a C99 feature}}
  StillPOD2 array4[N][3]; // expected-warning{{variable length arrays are a C99 feature}}
  NonPOD array5[N]; // expected-error{{variable length array of non-POD element type 'NonPOD'}}
}
