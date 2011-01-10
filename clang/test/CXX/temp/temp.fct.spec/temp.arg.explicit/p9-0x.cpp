// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Metafunction to extract the Nth type from a set of types.
template<unsigned N, typename ...Types> struct get_nth_type;

template<unsigned N, typename Head, typename ...Tail>
struct get_nth_type<N, Head, Tail...> : get_nth_type<N-1, Tail...> { };

template<typename Head, typename ...Tail>
struct get_nth_type<0, Head, Tail...> {
  typedef Head type;
};

// Placeholder type  when get_nth_type fails.
struct no_type {};

template<unsigned N>
struct get_nth_type<N> {
  typedef no_type type;
};

template<typename ...Args>
typename get_nth_type<0, Args...>::type first_arg(Args...);

template<typename ...Args>
typename get_nth_type<1, Args...>::type second_arg(Args...);

// Test explicit specification of function template arguments.
void test_explicit_spec_simple() {
  int *ip1 = first_arg<int *>(0);
  int *ip2 = first_arg<int *, float*>(0, 0);
  float *fp1 = first_arg<float *, double*, int*>(0, 0, 0);
}

// Template argument deduction can extend the sequence of template
// arguments corresponding to a template parameter pack, even when the
// sequence contains explicitly specified template arguments.
void test_explicit_spec_extension(double *dp) {
  int *ip1 = first_arg<int *>(0, 0);
  int *ip2 = first_arg<int *, float*>(0, 0, 0, 0);
  float *fp1 = first_arg<float *, double*, int*>(0, 0, 0);  
  int *i1 = second_arg<float *>(0, (int*)0, 0);  
  double *dp1 = first_arg<>(dp);
}

