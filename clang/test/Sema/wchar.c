// RUN: clang-cc %s -fsyntax-only -verify 
#include <wchar.h>

int check_wchar_size[sizeof(*L"") == sizeof(wchar_t) ? 1 : -1];

void foo() {
  int t1[] = L"x";
  wchar_t tab[] = L"x";

  int t2[] = "x";     // expected-error {{initialization}}
  char t3[] = L"x";   // expected-error {{initialization}}
}
