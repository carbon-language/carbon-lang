// RUN: clang-cc -fsyntax-only -verify %s

// PR5061
namespace a {
  template <typename T> class C {};
}
namespace b {
  template<typename T> void f0(a::C<T> &a0) { }
}
