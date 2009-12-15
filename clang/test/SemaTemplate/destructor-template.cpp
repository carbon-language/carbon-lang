// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename A> class s0 {

  template<typename B> class s1 : public s0<A> {
    ~s1() {}
    s0<A> ms0;
  };

};

struct Incomplete;

template<typename T>
void destroy_me(T me) {
  me.~T();
}

template void destroy_me(Incomplete*);
