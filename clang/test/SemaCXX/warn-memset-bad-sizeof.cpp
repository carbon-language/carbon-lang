// RUN: %clang_cc1 -fsyntax-only -verify -Wno-sizeof-array-argument %s
//
extern "C" void *memset(void *, int, unsigned);
extern "C" void *memmove(void *s1, const void *s2, unsigned n);
extern "C" void *memcpy(void *s1, const void *s2, unsigned n);

struct S {int a, b, c, d;};
typedef S* PS;

struct Foo {};
typedef const Foo& CFooRef;
typedef const Foo CFoo;
typedef volatile Foo VFoo;
typedef const volatile Foo CVFoo;

typedef double Mat[4][4];

template <class Dest, class Source>
inline Dest bit_cast(const Source& source) {
  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

// http://www.lysator.liu.se/c/c-faq/c-2.html#2-6
void f(Mat m, const Foo& const_foo, char *buffer) {
  S s;
  S* ps = &s;
  PS ps2 = &s;
  char arr[5];
  char* parr[5];
  Foo foo;
  char* heap_buffer = new char[42];

  /* Should warn */
  memset(&s, 0, sizeof(&s));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same expression as the destination}}
  memset(ps, 0, sizeof(ps));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same expression as the destination}}
  memset(ps2, 0, sizeof(ps2));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same expression as the destination}}
  memset(ps2, 0, sizeof(typeof(ps2)));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same pointer type}}
  memset(ps2, 0, sizeof(PS));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same pointer type}}
  memset(heap_buffer, 0, sizeof(heap_buffer));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same expression as the destination}}

  memcpy(&s, 0, sizeof(&s));  // \
      // expected-warning {{argument to 'sizeof' in 'memcpy' call is the same expression as the destination}}
  memcpy(0, &s, sizeof(&s));  // \
      // expected-warning {{argument to 'sizeof' in 'memcpy' call is the same expression as the source}}

  /* Shouldn't warn */
  memset((void*)&s, 0, sizeof(&s));
  memset(&s, 0, sizeof(s));
  memset(&s, 0, sizeof(S));
  memset(&s, 0, sizeof(const S));
  memset(&s, 0, sizeof(volatile S));
  memset(&s, 0, sizeof(volatile const S));
  memset(&foo, 0, sizeof(CFoo));
  memset(&foo, 0, sizeof(VFoo));
  memset(&foo, 0, sizeof(CVFoo));
  memset(ps, 0, sizeof(*ps));
  memset(ps2, 0, sizeof(*ps2));
  memset(ps2, 0, sizeof(typeof(*ps2)));
  memset(arr, 0, sizeof(arr));
  memset(parr, 0, sizeof(parr));

  memcpy(&foo, &const_foo, sizeof(Foo));
  memcpy((void*)&s, 0, sizeof(&s));
  memcpy(0, (void*)&s, sizeof(&s));
  char *cptr;
  memcpy(&cptr, buffer, sizeof(cptr));
  memcpy((char*)&cptr, buffer, sizeof(cptr));

  CFooRef cfoo = foo;
  memcpy(&foo, &cfoo, sizeof(Foo));

  memcpy(0, &arr, sizeof(arr));
  typedef char Buff[8];
  memcpy(0, &arr, sizeof(Buff));

  unsigned char* puc;
  bit_cast<char*>(puc);

  float* pf;
  bit_cast<int*>(pf);

  int iarr[14];
  memset(&iarr[0], 0, sizeof iarr);

  int* iparr[14];
  memset(&iparr[0], 0, sizeof iparr);

  memset(m, 0, sizeof(Mat));

  // Copy to raw buffer shouldn't warn either
  memcpy(&foo, &arr, sizeof(Foo));
  memcpy(&arr, &foo, sizeof(Foo));
}
