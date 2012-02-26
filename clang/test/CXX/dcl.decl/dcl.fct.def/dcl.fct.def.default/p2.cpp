// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// An explicitly-defaulted function may be declared constexpr only if it would
// have been implicitly declared as constexpr.
struct S1 {
  constexpr S1() = default; // expected-error {{defaulted definition of default constructor is not constexpr}}
  constexpr S1(const S1&) = default;
  constexpr S1(S1&&) = default;
  constexpr S1 &operator=(const S1&) = default; // expected-error {{explicitly-defaulted copy assignment operator may not have}}
  constexpr S1 &operator=(S1&&) = default; // expected-error {{explicitly-defaulted move assignment operator may not have}}
  constexpr ~S1() = default; // expected-error {{destructor cannot be marked constexpr}}
  int n;
};
struct NoCopyMove {
  constexpr NoCopyMove() {}
  NoCopyMove(const NoCopyMove&);
  NoCopyMove(NoCopyMove&&);
};
struct S2 {
  constexpr S2() = default;
  constexpr S2(const S2&) = default; // expected-error {{defaulted definition of copy constructor is not constexpr}}
  constexpr S2(S2&&) = default; // expected-error {{defaulted definition of move constructor is not constexpr}}
  NoCopyMove ncm;
};

// If a function is explicitly defaulted on its first declaration
//   -- it is implicitly considered to be constexpr if the implicit declaration
//      would be
struct S3 {
  S3() = default; // expected-note {{here}}
  S3(const S3&) = default;
  S3(S3&&) = default;
  constexpr S3(int n) : n(n) {}
  int n;
};
constexpr S3 s3a = S3(0);
constexpr S3 s3b = s3a;
constexpr S3 s3c = S3();
constexpr S3 s3d; // expected-error {{constant expression}} expected-note {{non-constexpr constructor}}

struct S4 {
  S4() = default;
  S4(const S4&) = default; // expected-note {{here}}
  S4(S4&&) = default; // expected-note {{here}}
  NoCopyMove ncm;
};
constexpr S4 s4a; // ok
constexpr S4 s4b = S4(); // expected-error {{constant expression}} expected-note {{non-constexpr constructor}}
constexpr S4 s4c = s4a; // expected-error {{constant expression}} expected-note {{non-constexpr constructor}}

struct S5 {
  constexpr S5();
  int n = 1, m = n + 3;
};
constexpr S5::S5() = default;
static_assert(S5().m == 4, "");
