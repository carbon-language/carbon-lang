// RUN: %clang_cc1 -verify -fsyntax-only -Wshadow -Wold-style-cast %s

// Test that macro expansions from system headers don't trigger 'syntactic'
// warnings that are not actionable.

#ifdef IS_SYSHEADER
#pragma clang system_header

#define SANITY(a) (a / 0)

#define SHADOW(a) __extension__({ int v = a; v; })

#define OLD_STYLE_CAST(a) ((int) (a))

#else

#define IS_SYSHEADER
#include __FILE__

void testSanity() {
  // Validate that the test is set up correctly
  int i = SANITY(0); // expected-warning {{division by zero is undefined}}
}

void PR16093() {
  // no -Wshadow in system macro expansion
  int i = SHADOW(SHADOW(1));
}

void PR18147() {
  // no -Wold_style_cast in system macro expansion
  int i = OLD_STYLE_CAST(0);
}

#endif
