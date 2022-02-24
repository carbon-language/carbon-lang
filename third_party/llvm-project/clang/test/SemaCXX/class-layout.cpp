// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify -std=c++98 -Wno-inaccessible-base
// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify -std=c++11 -Wno-inaccessible-base
// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify -std=c++11 -Wno-inaccessible-base -fclang-abi-compat=6 -DCLANG_ABI_COMPAT=6
// expected-no-diagnostics

#define SA(n, p) int a##n[(p) ? 1 : -1]

struct A {
  int a;
  char b;
};

SA(0, sizeof(A) == 8);

struct B : A {
  char c;
};

SA(1, sizeof(B) == 12);

struct C {
// Make fields private so C won't be a POD type.
private:
  int a;
  char b;
};

SA(2, sizeof(C) == 8);

struct D : C {
  char c;
};

SA(3, sizeof(D) == 8);

struct __attribute__((packed)) E {
  char b;
  int a;
};

SA(4, sizeof(E) == 5);

struct __attribute__((packed)) F : E {
  char d;
};

SA(5, sizeof(F) == 6);

struct G { G(); };
struct H : G { };

SA(6, sizeof(H) == 1);

struct I {
  char b;
  int a;
} __attribute__((packed));

SA(6_1, sizeof(I) == 5);

// PR5580
namespace PR5580 {

class A { bool iv0 : 1; };
SA(7, sizeof(A) == 1);  

class B : A { bool iv0 : 1; };
SA(8, sizeof(B) == 2);

struct C { bool iv0 : 1; };
SA(9, sizeof(C) == 1);  

struct D : C { bool iv0 : 1; };
SA(10, sizeof(D) == 2);

}

namespace Test1 {

// Test that we don't assert on this hierarchy.
struct A { };
struct B : A { virtual void b(); };
class C : virtual A { int c; };
struct D : virtual B { };
struct E : C, virtual D { };
class F : virtual E { };
struct G : virtual E, F { };

SA(0, sizeof(G) == 24);

}

namespace Test2 {

// Test that this somewhat complex class structure is laid out correctly.
struct A { };
struct B : A { virtual void b(); };
struct C : virtual B { };
struct D : virtual A { };
struct E : virtual B, D { };
struct F : E, virtual C { };
struct G : virtual F, A { };
struct H { G g; };

SA(0, sizeof(H) == 24);

}

namespace PR16537 {
namespace test1 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    char tail_padding;
  };
    
  struct might_use_tail_padding : public tail_padded_pod_in_11_only {
    char may_go_into_tail_padding;
  };

  SA(0, sizeof(might_use_tail_padding) == 16);
}

namespace test2 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11 __attribute__((aligned(16)));
  };
    
  struct might_use_tail_padding : public tail_padded_pod_in_11_only {
    char may_go_into_tail_padding;
  };

  SA(0, sizeof(might_use_tail_padding) == 16);
}

namespace test3 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    char tail_padding;
  };

  struct second_base {
      char foo;
  };
    
  struct might_use_tail_padding : public tail_padded_pod_in_11_only, public second_base {

  };
  SA(0, sizeof(might_use_tail_padding) == 16);
}

namespace test4 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    char tail_padding;
  };

  struct second_base {
    char foo;
  };
    
  struct might_use_tail_padding : public tail_padded_pod_in_11_only, public second_base {
    char may_go_into_tail_padding;
  };
  SA(0, sizeof(might_use_tail_padding) == 16);
}

namespace test5 {
  struct pod_in_11_only {
  private:
    long long x;
  };

  struct pod_in_11_only2 {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    char tail_padding;
  };

  struct second_base {
    pod_in_11_only2 two;
    char foo;
  };
    
  struct might_use_tail_padding : public tail_padded_pod_in_11_only, public second_base {
    char may_go_into_tail_padding;
  };
  SA(0, sizeof(might_use_tail_padding) == 32);
}

namespace test6 {
  struct pod_in_11_only {
  private:
    long long x;
  };

  struct pod_in_11_only2 {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    char tail_padding;
  };

  struct second_base {
    pod_in_11_only2 two;
    char foo;
  };
    
  struct might_use_tail_padding : public tail_padded_pod_in_11_only, public second_base {
    char may_go_into_tail_padding;
  };
  SA(0, sizeof(might_use_tail_padding) == 32);
}

namespace test7 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    pod_in_11_only pod12;
    char tail_padding;
  };
    
  struct might_use_tail_padding : public tail_padded_pod_in_11_only {
    char may_go_into_tail_padding;
  };

  SA(0, sizeof(might_use_tail_padding) == 24);
}

namespace test8 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    char tail_padding;
  };

  struct another_layer {
    tail_padded_pod_in_11_only pod;
    char padding;
  };
    
  struct might_use_tail_padding : public another_layer {
    char may_go_into_tail_padding;
  };

  SA(0, sizeof(might_use_tail_padding) == 24);
}

namespace test9 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct tail_padded_pod_in_11_only {
    pod_in_11_only pod11;
    char tail_padding;
  };

  struct another_layer : tail_padded_pod_in_11_only {
  };
    
  struct might_use_tail_padding : public another_layer {
    char may_go_into_tail_padding;
  };

  SA(0, sizeof(might_use_tail_padding) == 16);
}

namespace test10 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A {
    pod_in_11_only a;
    char apad;
  };

  struct B {
    char b;
  };

  struct C {
    pod_in_11_only c;
    char cpad;
  };

  struct D {
    char d;
  };
    
  struct might_use_tail_padding : public A, public B, public C, public D {
  };

  SA(0, sizeof(might_use_tail_padding) == 32);
}

namespace test11 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A {
    pod_in_11_only a;
    char apad;
  };

  struct B {
    char b_pre;
    pod_in_11_only b;
    char bpad;
  };

  struct C {
    char c_pre;
    pod_in_11_only c;
    char cpad;
  };

  struct D {
    char d_pre;
    pod_in_11_only d;
    char dpad;
  };
    
  struct might_use_tail_padding : public A, public B, public C, public D {
    char m;
  };

  SA(0, sizeof(might_use_tail_padding) == 88);
}

namespace test12 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A {
    pod_in_11_only a __attribute__((aligned(128)));
  };

  struct B {
    char bpad;
  };

  struct C {
    char cpad;
  };

  struct D {
    char dpad;
  };
    
  struct might_use_tail_padding : public A, public B, public C, public D {
    char m;
  };
  SA(0, sizeof(might_use_tail_padding) == 128);
}

namespace test13 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A {
    pod_in_11_only a;
    char apad;
  };

  struct B {
  };

  struct C {
    char c_pre;
    pod_in_11_only c;
    char cpad;
  };

  struct D {
  };
    
  struct might_use_tail_padding : public A, public B, public C, public D {
    char m;
  };
  SA(0, sizeof(might_use_tail_padding) == 40);
}

namespace test14 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A {
    pod_in_11_only a;
    char apad;
  };

  struct might_use_tail_padding : public A {
    struct {
      int : 0;
    } x;
  };
  SA(0, sizeof(might_use_tail_padding) == 16);
}

namespace test15 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A {
    pod_in_11_only a;
    char apad;
  };

  struct might_use_tail_padding : public A {
    struct {
      char a:1;
      char b:2;
      char c:2;
      char d:2;
      char e:1;
    } x;
  };
  SA(0, sizeof(might_use_tail_padding) == 16);
}

namespace test16 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A  {
    pod_in_11_only a;
    char apad;
  };

  struct B {
    char bpod;
    pod_in_11_only b;
    char bpad;
  };

  struct C : public A, public B {
  };
  
  struct D : public C {
  };

  struct might_use_tail_padding : public D {
    char m;
  };
  SA(0, sizeof(might_use_tail_padding) == 40);
}

namespace test17 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A {
    pod_in_11_only a __attribute__((aligned(512)));
  };

  struct B {
    char bpad;
    pod_in_11_only foo;
    char btail;
  };

  struct C {
    char cpad;
  };

  struct D {
    char dpad;
  };
    
  struct might_use_tail_padding : public A, public B, public C, public D {
    char a;
  };
  SA(0, sizeof(might_use_tail_padding) == 512);
}

namespace test18 {
  struct pod_in_11_only {
  private:
    long long x;
  };
   
  struct A  {
    pod_in_11_only a;
    char apad;
  };

  struct B {
    char bpod;
    pod_in_11_only b;
    char bpad;
  };

  struct A1  {
    pod_in_11_only a;
    char apad;
  };

  struct B1 {
    char bpod;
    pod_in_11_only b;
    char bpad;
  };

  struct C : public A, public B {
  };

  struct D : public A1, public B1 {
  };

  struct E : public D, public C {
  };

  struct F : public E {
  };

  struct might_use_tail_padding : public F {
    char m;
  };
  SA(0, sizeof(might_use_tail_padding) == 80);
}
} // namespace PR16537

namespace PR37275 {
  struct X { char c; };

  struct A { int n; };
  _Static_assert(_Alignof(A) == _Alignof(int), "");

  // __attribute__((packed)) does not apply to base classes.
  struct __attribute__((packed)) B : X, A {};
#if defined(CLANG_ABI_COMPAT) && CLANG_ABI_COMPAT <= 6
  _Static_assert(_Alignof(B) == 1, "");
  _Static_assert(__builtin_offsetof(B, n) == 1, "");
#else
  _Static_assert(_Alignof(B) == _Alignof(int), "");
  _Static_assert(__builtin_offsetof(B, n) == 4, "");
#endif

  // #pragma pack does, though.
#pragma pack(push, 2)
  struct C : X, A {};
  _Static_assert(_Alignof(C) == 2, "");
  _Static_assert(__builtin_offsetof(C, n) == 2, "");

  struct __attribute__((packed)) D : X, A {};
#if defined(CLANG_ABI_COMPAT) && CLANG_ABI_COMPAT <= 6
  _Static_assert(_Alignof(D) == 1, "");
  _Static_assert(__builtin_offsetof(D, n) == 1, "");
#else
  _Static_assert(_Alignof(D) == 2, "");
  _Static_assert(__builtin_offsetof(D, n) == 2, "");
#endif
#pragma pack(pop)
}
