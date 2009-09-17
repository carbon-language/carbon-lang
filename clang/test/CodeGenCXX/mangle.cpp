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

void g() {
  // CHECK: @_Z3ft1IidEvT0_T_
  ft1<int, double>(1, 0);
  
  // CHECK: @_Z3ft2IcEvT_PFvS0_ES2_
  ft2<char>(1, 0, 0);
}

extern "C++" {
  // CHECK: @_Z1hv
 void h() { } 
}
