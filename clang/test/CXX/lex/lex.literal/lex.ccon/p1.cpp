// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s
// Runs in c++0x mode so that char16_t and char32_t are available.

// Check types of char literals
extern char a;
extern __typeof('a') a;
extern int b;
extern __typeof('asdf') b;
extern wchar_t c;
extern __typeof(L'a') c;
extern char16_t d;
extern __typeof(u'a') d;
extern char32_t e;
extern __typeof(U'a') e;
