// RUN: clang-cc -fsyntax-only %s

template<typename T> struct X0 { };
struct X1 { };

struct Y {
  template<typename T> union X0;
  template<typename T> friend union X0;
  
  union X1;
  friend union X1;
};


// FIXME: Woefully inadequate for testing
