// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin -verify %s

// PR6762
#define a_list __builtin_va_list
extern a_list l;
extern int f (a_list arg);
namespace n {
int f(a_list arguments);
void y() {
  f(l);
}
}
