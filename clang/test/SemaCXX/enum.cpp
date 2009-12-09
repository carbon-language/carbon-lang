// RUN: clang-cc -fsyntax-only -verify %s

enum E {
  Val1,
  Val2
};

int& enumerator_type(int);
float& enumerator_type(E);

void f() {
  E e = Val1;
  float& fr = enumerator_type(Val2);
}

// <rdar://problem/6502934>
typedef enum Foo {
  A = 0,
  B = 1
} Foo;

void bar() {
  Foo myvar = A;
  myvar = B;
}

/// PR3688
struct s1 {
  enum e1 (*bar)(void); // expected-error{{ISO C++ forbids forward references to 'enum' types}}
};

enum e1 { YES, NO };

static enum e1 badfunc(struct s1 *q) {
  return q->bar();
}

enum e2; // expected-error{{ISO C++ forbids forward references to 'enum' types}}

namespace test1 {
  template <class A, class B> struct is_same { static const int value = -1; };
  template <class A> struct is_same<A,A> { static const int value = 1; };

  enum enum0 { v0 };
  int test0[is_same<typeof(+v0), int>::value];

  enum enum1 { v1 = __INT_MAX__ };
  int test1[is_same<typeof(+v1), int>::value];

  enum enum2 { v2 = __INT_MAX__ * 2U };
  int test2[is_same<typeof(+v2), unsigned int>::value];

  // This kindof assumes that 'int' is smaller than 'long long'.
#if defined(__LP64__)
  enum enum3 { v3 = __LONG_LONG_MAX__ };
  int test3[is_same<typeof(+v3), long>::value];

  enum enum4 { v4 = __LONG_LONG_MAX__ * 2ULL };
  int test4[is_same<typeof(+v4), unsigned long>::value];
#else
  enum enum3 { v3 = __LONG_LONG_MAX__ };
  int test3[is_same<typeof(+v3), long long>::value];

  enum enum4 { v4 = __LONG_LONG_MAX__ * 2ULL };
  int test4[is_same<typeof(+v4), unsigned long long>::value];  
#endif
}
