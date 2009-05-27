// RUN: clang-cc -fsyntax-only %s

namespace N {
  struct Outer {
    struct Inner {
      template<typename T>
      struct InnerTemplate {
        struct VeryInner {
          typedef T type;

          static enum K1 { K1Val = sizeof(T) } Kind1;
          static enum { K2Val = sizeof(T)*2 } Kind2;

          void foo() {
            K1 k1 = K1Val;
            Kind1 = K1Val;
            Outer::Inner::InnerTemplate<type>::VeryInner::Kind2 = K2Val;
          }
        };
      };
    };
  };
}

typedef int INT;
template struct N::Outer::Inner::InnerTemplate<INT>::VeryInner;
