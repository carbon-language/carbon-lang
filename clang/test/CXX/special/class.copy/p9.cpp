// RUN: %clang_cc1 -fsyntax-only -verify %s

struct ConstCopy {
  ConstCopy();
  ConstCopy(const ConstCopy&);
};

struct NonConstCopy {
  NonConstCopy();
  NonConstCopy(NonConstCopy&);
};

struct VirtualInheritsNonConstCopy : virtual NonConstCopy { 
  VirtualInheritsNonConstCopy();
  VirtualInheritsNonConstCopy(const VirtualInheritsNonConstCopy&);
};

struct ImplicitNonConstCopy1 : NonConstCopy { 
  ImplicitNonConstCopy1();
};

struct ImplicitNonConstCopy2 {
  ImplicitNonConstCopy2();
  NonConstCopy ncc;
};

struct ImplicitNonConstCopy3 { 
  ImplicitNonConstCopy3();
  NonConstCopy ncc_array[2][3];
};

struct ImplicitNonConstCopy4 : VirtualInheritsNonConstCopy { 
  ImplicitNonConstCopy4();
};

void test_non_const_copy(const ImplicitNonConstCopy1 &cincc1,
                         const ImplicitNonConstCopy2 &cincc2,
                         const ImplicitNonConstCopy3 &cincc3,
                         const ImplicitNonConstCopy4 &cincc4) {
  (void)sizeof(ImplicitNonConstCopy1(cincc1)); // expected-error{{functional-style cast from 'const ImplicitNonConstCopy1' to 'ImplicitNonConstCopy1' is not allowed}}
  (void)sizeof(ImplicitNonConstCopy2(cincc2)); // expected-error{{functional-style cast from 'const ImplicitNonConstCopy2' to 'ImplicitNonConstCopy2' is not allowed}}
  (void)sizeof(ImplicitNonConstCopy3(cincc3)); // expected-error{{functional-style cast from 'const ImplicitNonConstCopy3' to 'ImplicitNonConstCopy3' is not allowed}}
  (void)sizeof(ImplicitNonConstCopy4(cincc4)); // expected-error{{functional-style cast from 'const ImplicitNonConstCopy4' to 'ImplicitNonConstCopy4' is not allowed}}
}
