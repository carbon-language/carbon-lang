// RUN: %clang_cc1 -fsyntax-only -verify %s

#if __has_feature(__builtin_add_overflow)
#warning defined as expected
// expected-warning@-1 {{defined as expected}}
#endif

void test(void) {
  unsigned r;
  const char * c;
  float f;
  const unsigned q;

  __builtin_add_overflow();  // expected-error {{too few arguments to function call, expected 3, have 0}}
  __builtin_add_overflow(1, 1, 1, 1);  // expected-error {{too many arguments to function call, expected 3, have 4}}

  __builtin_add_overflow(c, 1, &r);  // expected-error {{operand argument to overflow builtin must be an integer ('const char *' invalid)}}
  __builtin_add_overflow(1, c, &r);  // expected-error {{operand argument to overflow builtin must be an integer ('const char *' invalid)}}
  __builtin_add_overflow(1, 1, 3);  // expected-error {{result argument to overflow builtin must be a pointer to a non-const integer ('int' invalid)}}
  __builtin_add_overflow(1, 1, &f);  // expected-error {{result argument to overflow builtin must be a pointer to a non-const integer ('float *' invalid)}}
  __builtin_add_overflow(1, 1, &q);  // expected-error {{result argument to overflow builtin must be a pointer to a non-const integer ('const unsigned int *' invalid)}}
}
