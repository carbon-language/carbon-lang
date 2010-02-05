// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR5336
template<typename FromCl>
struct isa_impl_cl {
 template<class ToCl>
 static void isa(const FromCl &Val) { }
};

template<class X, class Y>
void isa(const Y &Val) {   return isa_impl_cl<Y>::template isa<X>(Val); }

class Value;
void f0(const Value &Val) { isa<Value>(Val); }

// Implicit template-ids.
template<typename T>
struct X0 {
  template<typename U>
  void f1();
  
  template<typename U>
  void f2(U) {
    f1<U>();
  }
};

void test_X0_int(X0<int> xi, float f) {
  xi.f2(f);
}

// Not template-id expressions, but they almost look like it.
template<typename F>
struct Y {
  Y(const F&);
};

template<int I>
struct X {
  X(int, int);
  void f() { 
    Y<X<I> >(X<I>(0, 0)); 
    Y<X<I> >(::X<I>(0, 0)); 
  }
};

template struct X<3>;
