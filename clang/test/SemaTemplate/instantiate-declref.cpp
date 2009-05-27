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
