// RUN: %clang_cc1 -fsyntax-only -verify %s

// Core issue 150: Template template parameters and default arguments

namespace PR9353 {
  template<class _T, class Traits> class IM;

  template <class T, class Trt, 
            template<class _T, class Traits = int> class IntervalMap>
  void foo(IntervalMap<T,Trt>* m) { typedef IntervalMap<int> type; }

  void f(IM<int, int>* m) { foo(m); }
}
