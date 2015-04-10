// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace PR23186 {
decltype(ned);  // expected-error-re {{use of undeclared identifier 'ned'{{$}}}}
// The code below was triggering an UNREACHABLE in ASTContext::getTypeInfoImpl
// once the above code failed to recover properly after making the bogus
// correction of 'ned' to 'new'.
template <typename>
struct S {
  enum { V };
  void f() {
    switch (0)
    case V:
      ;
  }
};
}
