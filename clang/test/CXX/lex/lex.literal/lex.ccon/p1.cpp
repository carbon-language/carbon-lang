// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check types of char literals
extern char a;
extern __typeof('a') a;
extern int b;
extern __typeof('asdf') b;
extern wchar_t c;
extern __typeof(L'a') c;
