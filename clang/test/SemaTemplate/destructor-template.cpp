// RUN: clang-cc -fsyntax-only -verify %s

template<typename A> class s0 {

  template<typename B> class s1 : public s0<A> {
    ~s1() {}
    s0<A> ms0;
  };

};


