// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -triple i386-pc-win32 %s

// C++ mode with -fno-wchar works the same as C mode for wchar_t.
// RUN: %clang_cc1 -x c++ -fno-wchar -fsyntax-only -verify -fms-extensions -triple i386-pc-win32 %s

wchar_t f(); // expected-error{{unknown type name 'wchar_t'}}

// __wchar_t is available as an MS extension.
__wchar_t g(); // No error.

// __wchar_t is the same as the target's integer type for wide chars.
unsigned short g(); // No error.
