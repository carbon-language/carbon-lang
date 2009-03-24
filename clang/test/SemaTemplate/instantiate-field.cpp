// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct X {
  int x;
  T y; // expected-error{{data member instantiated with function type}}
  T* z;
  T bitfield : 12; // expected-error{{bit-field 'bitfield' has non-integral type 'float'}} \
                  // expected-error{{data member instantiated with function type}}

  mutable T x2; // expected-error{{data member instantiated with function type}}
};

void test1(const X<int> *xi) {
  int i1 = xi->x;
  const int &i2 = xi->y;
  int* ip1 = xi->z;
  int i3 = xi->bitfield;
  xi->x2 = 17;
}

void test2(const X<float> *xf) {
  (void)xf->x; // expected-note{{in instantiation of template class 'struct X<float>' requested here}}
}

void test3(const X<int(int)> *xf) {
  (void)xf->x; // expected-note{{in instantiation of template class 'struct X<int (int)>' requested here}}
}
