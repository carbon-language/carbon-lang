// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct X { };
struct Y { };

// CHECK: @unmangled_variable = global
// CHECK: @_ZN1N1iE = global
// CHECK: @_ZZN1N1fEiiE1b = internal global
// CHECK: @_ZZN1N1gEvE1a = internal global
// CHECK: @_ZGVZN1N1gEvE1a = internal global

// CHECK: define zeroext i1 @_ZplRK1YRA100_P1X
bool operator+(const Y&, X* (&xs)[100]) { return false; }

// CHECK: define void @_Z1f1s
typedef struct { int a; } s;
void f(s) { }

// CHECK: define void @_Z1f1e
typedef enum { foo } e;
void f(e) { }

// CHECK: define void @_Z1f1u
typedef union { int a; } u;
void f(u) { }

// CHECK: define void @_Z1f1x
typedef struct { int a; } x,y;
void f(y) { }

// CHECK: define void @_Z1fv
void f() { }

// CHECK: define void @_ZN1N1fEv
namespace N { void f() { } }

// CHECK: define void @_ZN1N1N1fEv
namespace N { namespace N { void f() { } } }

// CHECK: define void @unmangled_function
extern "C" { namespace N { void unmangled_function() { } } }

extern "C" { namespace N { int unmangled_variable = 10; } }

namespace N { int i; }

namespace N { int f(int, int) { static int b; return b; } }

namespace N { int h(); void g() { static int a = h(); } }

// CHECK: define void @_Z1fno
void f(__int128_t, __uint128_t) { } 

template <typename T> struct S1 {};

// CHECK: define void @_Z1f2S1IiE
void f(S1<int>) {}

// CHECK: define void @_Z1f2S1IdE
void f(S1<double>) {}

template <int N> struct S2 {};
// CHECK: define void @_Z1f2S2ILi100EE
void f(S2<100>) {}

// CHECK: define void @_Z1f2S2ILin100EE
void f(S2<-100>) {}

template <bool B> struct S3 {};

// CHECK: define void @_Z1f2S3ILb1EE
void f(S3<true>) {}

// CHECK: define void @_Z1f2S3ILb0EE
void f(S3<false>) {}

// CHECK: define void @_Z2f22S3ILb1EE
void f2(S3<100>) {}

struct S;

// CHECK: define void @_Z1fM1SKFvvE
void f(void (S::*)() const) {}

// CHECK: define void @_Z1fM1SFvvE
void f(void (S::*)()) {}

// CHECK: define void @_Z1fi
void f(const int) { }

template<typename T, typename U> void ft1(U u, T t) { }

template<typename T> void ft2(T t, void (*)(T), void (*)(T)) { }

template<typename T, typename U = S1<T> > struct S4 { };
template<typename T> void ft3(S4<T>*) {  }

namespace NS {
  template<typename T> void ft1(T) { }
}

void g1() {
  // CHECK: @_Z3ft1IidEvT0_T_
  ft1<int, double>(1, 0);
  
  // CHECK: @_Z3ft2IcEvT_PFvS0_ES2_
  ft2<char>(1, 0, 0);
  
  // CHECK: @_Z3ft3IiEvP2S4IT_2S1IS1_EE
  ft3<int>(0);
  
  // CHECK: @_ZN2NS3ft1IiEEvT_
  NS::ft1<int>(1);
}

// Expressions
template<int I> struct S5 { };

template<int I> void ft4(S5<I>) { }
void g2() {
  // CHECK: @_Z3ft4ILi10EEv2S5IXT_EE
  ft4(S5<10>());
  
  // CHECK: @_Z3ft4ILi20EEv2S5IXT_EE
  ft4(S5<20>());
}

extern "C++" {
  // CHECK: @_Z1hv
 void h() { } 
}

// PR5019
extern "C" { struct a { int b; }; }

// CHECK: @_Z1fP1a
int f(struct a *x) {
    return x->b;
}

// PR5017
extern "C" {
struct Debug {
 const Debug& operator<< (unsigned a) const { }
};
Debug dbg;
// CHECK: @_ZNK5DebuglsEj
int main(void) {  dbg << 32 ;}
}

template<typename T> struct S6 {
  typedef int B;
};

template<typename T> void ft5(typename S6<T>::B) { }
// CHECK: @_Z3ft5IiEvN2S6IT_E1BE
template void ft5<int>(int);

template<typename T> class A {};

namespace NS {
template<typename T> bool operator==(const A<T>&, const A<T>&) { return true; }
}

// CHECK: @_ZN2NSeqIcEEbRK1AIT_ES5_
template bool NS::operator==(const ::A<char>&, const ::A<char>&);

namespace std {
template<typename T> bool operator==(const A<T>&, const A<T>&) { return true; }
}

// CHECK: @_ZSteqIcEbRK1AIT_ES4_
template bool std::operator==(const ::A<char>&, const ::A<char>&);

struct S {
  typedef int U;
};

template <typename T> typename T::U ft6(const T&) { return 0; }

// CHECK: @_Z3ft6I1SENT_1UERKS1_
template int ft6<S>(const S&);

template<typename> struct __is_scalar {
  enum { __value = 1 };
};

template<bool, typename> struct __enable_if { };

template<typename T> struct __enable_if<true, T> {
  typedef T __type;
};

// PR5063
template<typename T> typename __enable_if<__is_scalar<T>::__value, void>::__type ft7() { }

// CHECK: @_Z3ft7IiEN11__enable_ifIXsr11__is_scalarIT_E7__valueEvE6__typeEv
template void ft7<int>();
// CHECK: @_Z3ft7IPvEN11__enable_ifIXsr11__is_scalarIT_E7__valueEvE6__typeEv
template void ft7<void*>();

// PR5144
extern "C" {
void extern_f(void);
};

// CHECK: @extern_f
void extern_f(void) { }

struct S7 {
  struct S { S(); };
  
  struct {
    S s;
  } a;
};

// PR5139
// CHECK: @_ZN2S7C1Ev
// CHECK: @_ZN2S7C2Ev
// CHECK: @"_ZN2S73$_0C1Ev"
S7::S7() {}

// PR5063
template<typename T> typename __enable_if<(__is_scalar<T>::__value), void>::__type ft8() { }
// CHECK: @_Z3ft8IiEN11__enable_ifIXsr11__is_scalarIT_E7__valueEvE6__typeEv
template void ft8<int>();
// CHECK: @_Z3ft8IPvEN11__enable_ifIXsr11__is_scalarIT_E7__valueEvE6__typeEv
template void ft8<void*>();
