// RUN: clang-cc -fsyntax-only -verify %s

// @encode expressions

template <typename T> struct Encode {
  static const char *encode(T t) { 
    return @encode(T);
  }
};

template struct Encode<int>;
template struct Encode<double>;
