// RUN: clang-cc -fsyntax-only -verify %s
namespace N {
  struct Outer {
    struct Inner {
      template<typename T>
      struct InnerTemplate {
        struct VeryInner {
          typedef T type;

          static enum K1 { K1Val = sizeof(T) } Kind1;
          static enum { K2Val = sizeof(T)*2 } Kind2;
          enum { K3Val = sizeof(T)*2 } Kind3;

          void foo() {
            K1 k1 = K1Val;
            Kind1 = K1Val;
            Outer::Inner::InnerTemplate<type>::VeryInner::Kind2 = K2Val;
            Kind3 = K3Val;
          }

          struct UeberInner {
            void bar() {
              K1 k1 = K1Val;
              Kind1 = K1Val;
              Outer::Inner::InnerTemplate<type>::VeryInner::Kind2 = K2Val;

              InnerTemplate t;
              InnerTemplate<type> t2;
            }
          };
        };
      };
    };
  };
}

typedef int INT;
template struct N::Outer::Inner::InnerTemplate<INT>::VeryInner;
template struct N::Outer::Inner::InnerTemplate<INT>::UeberInner; // expected-error{{'UeberInner' does not name a tag member}}

namespace N2 {
  struct Outer2 {
    template<typename T, typename U = T>
    struct Inner {
      void foo() {
        enum { K1Val = sizeof(T) } k1;
        enum K2 { K2Val = sizeof(T)*2 } k2a;

        K2 k2b = K2Val;

        struct S { T x, y; } s1;
        struct { U x, y; } s2;
        s1.x = s2.x; // expected-error{{incompatible}}

        typedef T type;
        type t2 = s1.x;

        typedef struct { T z; } type2;
        type2 t3 = { s1.x };

        Inner i1;
        i1.foo();
        Inner<T> i2;
        i2.foo();
      }
    };
  };
}

template struct N2::Outer2::Inner<float>;
template struct N2::Outer2::Inner<int*, float*>; // expected-note{{instantiation}}

// Test dependent pointer-to-member expressions.
template<typename T>
struct smart_ptr {
  struct safe_bool {
    int member;
  };
  
  operator int safe_bool::*() const { 
    return ptr? &safe_bool::member : 0;
  }
  
  T* ptr;
};

void test_smart_ptr(smart_ptr<int> p) {
  if (p) { }
}
