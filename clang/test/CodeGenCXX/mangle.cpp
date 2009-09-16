// RUN: clang-cc -emit-llvm %s -o %t -triple=x86_64-apple-darwin9 && 

// FIXME: This test is intentionally trivial, because we can't yet
// CodeGen anything real in C++.
struct X { };
struct Y { };

// RUN: grep _ZplRK1YRA100_P1X %t | count 1 &&
bool operator+(const Y&, X* (&xs)[100]) { return false; }

// RUN: grep _Z1f1s %t | count 1 &&
typedef struct { int a; } s;
void f(s) { }

// RUN: grep _Z1f1e %t| count 1 &&
typedef enum { foo } e;
void f(e) { }

// RUN: grep _Z1f1u %t | count 1 &&
typedef union { int a; } u;
void f(u) { }

// RUN: grep _Z1f1x %t | count 1 &&
typedef struct { int a; } x,y;
void f(y) { }

// RUN: grep _Z1fv %t | count 1 &&
void f() { }

// RUN: grep _ZN1N1fEv %t | count 1 &&
namespace N { void f() { } }

// RUN: grep _ZN1N1N1fEv %t | count 1 &&
namespace N { namespace N { void f() { } } }

// RUN: grep unmangled_function %t | count 1 &&
extern "C" { namespace N { void unmangled_function() { } } }

// RUN: grep unmangled_variable %t | count 1 &&
extern "C" { namespace N { int unmangled_variable = 10; } }

// RUN: grep _ZN1N1iE %t | count 1 &&
namespace N { int i; }

// RUN: grep _ZZN1N1fEiiE1b %t | count 2 &&
namespace N { int f(int, int) { static int b; return b; } }

// RUN: grep "_ZZN1N1gEvE1a =" %t | count 1 &&
// RUN: grep "_ZGVZN1N1gEvE1a =" %t | count 1 &&
namespace N { int h(); void g() { static int a = h(); } }

// RUN: grep "_Z1fno" %t | count 1 &&
void f(__int128_t, __uint128_t) { } 

template <typename T> struct S1 {};

// RUN: grep "_Z1f2S1IiE" %t | count 1 &&
void f(S1<int>) {}

// RUN: grep "_Z1f2S1IdE" %t | count 1 &&
void f(S1<double>) {}

template <int N> struct S2 {};
// RUN: grep "_Z1f2S2ILi100EE" %t | count 1 &&
void f(S2<100>) {}

// RUN: grep "_Z1f2S2ILin100EE" %t | count 1 &&
void f(S2<-100>) {}

template <bool B> struct S3 {};

// RUN: grep "_Z1f2S3ILb1EE" %t | count 1 &&
void f(S3<true>) {}

// RUN: grep "_Z1f2S3ILb0EE" %t | count 1 &&
void f(S3<false>) {}

// RUN: grep "_Z2f22S3ILb1EE" %t | count 1 &&
void f2(S3<100>) {}

struct S;

// RUN: grep "_Z1fM1SKFvvE" %t | count 1 &&
void f(void (S::*)() const) {}

// RUN: grep "_Z1fM1SFvvE" %t | count 1 &&
void f(void (S::*)()) {}

// RUN: grep "_Z1fi" %t | count 1 &&
void f(const int) { }

// RUN: true
