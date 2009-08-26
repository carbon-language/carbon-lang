// RUN: clang-cc -fsyntax-only -verify %s

template<typename T, typename U> // expected-note{{previous template}}
class X0 {
public:
  typedef int size_type;
  
  X0(int);
  ~X0();
  
  void f0(const T&, const U&);
  
  T& operator[](int i) const;
  
  void f1(size_type) const;
  void f2(size_type) const;
  void f3(size_type) const;
  void f4() ;
  
  operator T*() const;
  
  T value;
};

template<typename T, typename U>
void X0<T, U>::f0(const T&, const U&) { // expected-note{{previous definition}}
}

template<class X, class Y>
X& X0<X, Y>::operator[](int i) const {
  (void)i;
  return value;
}

template<class X, class Y>
void X0<X, Y>::f1(int) const { }

template<class X, class Y>
void X0<X, Y>::f2(size_type) const { }

template<class X, class Y, class Z> // expected-error{{too many template parameters}}
void X0<X, Y>::f3(size_type) const {
}

template<class X, class Y> 
void X0<Y, X>::f4() { } // expected-error{{does not refer}}

// FIXME: error message should probably say, "redefinition of 'X0<T, U>::f0'"
// rather than just "redefinition of 'f0'"
template<typename T, typename U>
void X0<T, U>::f0(const T&, const U&) { // expected-error{{redefinition}}
}

// Test out-of-line constructors, destructors
template<typename T, typename U>
X0<T, U>::X0(int x) : value(x) { }

template<typename T, typename U>
X0<T, U>::~X0() { }

// Test out-of-line conversion functions.
template<typename T, typename U>
X0<T, U>::operator T*() const {
  return &value;
}
