// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

static_assert(__is_literal(int), "fail");
static_assert(__is_literal_type(int), "fail"); // alternate spelling for GCC
static_assert(__is_literal(void*), "fail");
enum E { E1 };
static_assert(__is_literal(E), "fail");
static_assert(__is_literal(decltype(E1)), "fail");
typedef int IAR[10];
static_assert(__is_literal(IAR), "fail");
// FIXME: Records
