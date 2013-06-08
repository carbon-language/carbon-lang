// RUN: %clang_cc1 -std=c++1y -verify %s
// RUN: %clang_cc1 -std=c++1y -verify %s -fdelayed-template-parsing

namespace nested_local_templates_1 {

template <class T> struct Outer {
  template <class U> int outer_mem(T t, U u) {
    struct Inner {
      template <class V> int inner_mem(T t, U u, V v) {
        struct InnerInner {
          template <class W> int inner_inner_mem(W w, T t, U u, V v) {
            return 0;
          }
        };
        InnerInner().inner_inner_mem("abc", t, u, v);
        return 0;
      }
    };
    Inner i;
    i.inner_mem(t, u, 3.14);
    return 0;
  }

  template <class U> int outer_mem(T t, U *u);
};

template int Outer<int>::outer_mem(int, char);

template <class T> template <class U> int Outer<T>::outer_mem(T t, U *u) {
  struct Inner {
    template <class V>
    int inner_mem(T t, U u, V v) { //expected-note{{candidate function}}
      struct InnerInner {
        template <class W> int inner_inner_mem(W w, T t, U u, V v) { return 0; }
      };
      InnerInner().inner_inner_mem("abc", t, u, v);
      return 0;
    }
  };
  Inner i;
  i.inner_mem(t, U{}, i);
  i.inner_mem(t, u, 3.14); //expected-error{{no matching member function for call to 'inner}}
  return 0;
}

template int Outer<int>::outer_mem(int, char *); //expected-note{{in instantiation of function}}

} // end ns

namespace nested_local_templates_2 {

template <class T> struct Outer {
  template <class U> void outer_mem(T t, U u) {
    struct Inner {
      template <class V> struct InnerTemplateClass {
        template <class W>
        void itc_mem(T t, U u, V v, W w) { //expected-note{{candidate function}}
          struct InnerInnerInner {
            template <class X> void iii_mem(X x) {}
          };
          InnerInnerInner i;
          i.iii_mem("abc");
        }
      };
    };
    Inner i;
    typename Inner::template InnerTemplateClass<Inner> ii;
    ii.itc_mem(t, u, i, "jim");
    ii.itc_mem(t, u, 0, "abd"); //expected-error{{no matching member function}}
  }
};

template void
    Outer<int>::outer_mem(int, char); //expected-note{{in instantiation of}}

}

namespace more_nested_local_templates {

int test() {
  struct Local {
    template <class U> void foo(U u) {
      struct Inner {
        template <class A> auto operator()(A a, U u2)->U { return u2; }
        ;
      };
      Inner GL;
      GL('a', u);
      GL(3.14, u);
    }
  };
  Local l;
  l.foo("nmabc");
  return 0;
}
int t = test();
}