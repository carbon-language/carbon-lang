// RUN: %clang_cc1 -triple i686-windows-gnu  -verify -std=c++03 %s
// RUN: %clang_cc1 -triple i686-windows-gnu  -verify -std=c++11 %s
// RUN: %clang_cc1 -triple i686-windows-msvc -verify -std=c++11 %s

// FIXME: For C++03 MS ABI we erroneously try to synthesize default ctor, etc. for S.

// expected-no-diagnostics

struct NonCopyable {
private:
  NonCopyable();
};

struct __declspec(dllexport) S {
  NonCopyable member;
};
