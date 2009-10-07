// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct X0 {
  template<typename U> struct Inner0 {
    static const unsigned value = 0;
  };
  
  template<typename U> struct Inner0<U*> { 
    static const unsigned value = 1;
  };
};

template<typename T> template<typename U>
struct X0<T>::Inner0<const U*> {
  static const unsigned value = 2;
};

// FIXME: Test instantiation of these partial specializations (once they are
// implemented).
