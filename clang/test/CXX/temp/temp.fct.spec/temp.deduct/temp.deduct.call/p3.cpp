// RUN: clang-cc -fsyntax-only %s

template<typename T> struct A { };

// Top-level cv-qualifiers of P's type are ignored for type deduction.
template<typename T> A<T> f0(const T);

void test_f0(int i, const int ci) {
  A<int> a0 = f0(i);
  A<int> a1 = f0(ci);
}

// If P is a reference type, the type referred to by P is used for type 
// deduction.
template<typename T> A<T> f1(T&);

void test_f1(int i, const int ci, volatile int vi) {
  A<int> a0 = f1(i);
  A<const int> a1 = f1(ci);
  A<volatile int> a2 = f1(vi);
}

template<typename T, unsigned N> struct B { };
template<typename T, unsigned N> B<T, N> g0(T (&array)[N]);

void test_g0() {
  int array0[5];
  B<int, 5> b0 = g0(array0);
  const int array1[] = { 1, 2, 3};
  B<const int, 3> b1 = g0(array1);
}

template<typename T> B<T, 0> g1(const A<T>&);

void test_g1(A<float> af) {
  B<float, 0> b0 = g1(af);
  B<int, 0> b1 = g1(A<int>());
}

//   - If the original P is a reference type, the deduced A (i.e., the type
//     referred to by the reference) can be more cv-qualified than the 
//     transformed A.
template<typename T> A<T> f2(const T&);

void test_f2(int i, const int ci, volatile int vi) {
  A<int> a0 = f2(i);
  A<int> a1 = f2(ci);
  A<volatile int> a2 = f2(vi);
}

// FIXME: the next two bullets require a bit of effort.