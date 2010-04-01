// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++0x [temp.arg.nontype]p1:
//
//   A template-argument for a non-type, non-template template-parameter shall
//   be one of:
//   -- an integral constant expression; or
//   -- the name of a non-type template-parameter ; or
namespace non_type_tmpl_param {
  template <int N> struct X0 { X0(); };
  template <int N> X0<N>::X0() { }
  template <int* N> struct X1 { X1(); };
  template <int* N> X1<N>::X1() { }
  template <int& N> struct X3 { X3(); };
  template <int& N> X3<N>::X3() { }
  template <int (*F)(int)> struct X4 { X4(); };
  template <int (*F)(int)> X4<F>::X4() { }
  template <typename T, int (T::* M)(int)> struct X5 { X5(); };
  template <typename T, int (T::* M)(int)> X5<T, M>::X5() { }
}

//   -- the address of an object or function with external linkage, including
//      function templates and function template-ids but excluding non-static
//      class members, expressed as & id-expression where the & is optional if
//      the name refers to a function or array, or if the corresponding
//      template-parameter is a reference; or
namespace addr_of_obj_or_func {
  template <int* p> struct X0 { };
  template <int (*fp)(int)> struct X1 { };
  // FIXME: Add reference template parameter tests.

  int i = 42;
  int iarr[10];
  int f(int i);
  template <typename T> T f_tmpl(T t);
  void test() {
    X0<&i> x0a;
    X0<iarr> x0b;
    X1<&f> x1a;
    X1<f> x1b;
    X1<f_tmpl> x1c;
    X1<f_tmpl<int> > x1d;
  }
}

//   -- a constant expression that evaluates to a null pointer value (4.10); or
//   -- a constant expression that evaluates to a null member pointer value
//      (4.11); or
//   -- a pointer to member expressed as described in 5.3.1.

namespace bad_args {
  template <int* N> struct X0 { }; // expected-note 2{{template parameter is declared here}}
  int i = 42;
  X0<&i + 2> x0a; // expected-error{{non-type template argument does not refer to any declaration}}
  int* iptr = &i;
  X0<iptr> x0b; // expected-error{{non-type template argument for template parameter of pointer type 'int *' must have its address taken}}
}
