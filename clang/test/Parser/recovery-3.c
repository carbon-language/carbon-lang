// RUN: clang -fsyntax-only -verify -pedantic %s

// Testcase derived from PR2692
static char *f (char * (*g) (char **, int), char **p, ...) {
    char *s;
    va_list v;                              // expected-error {{identifier}}
    s = g (p, __builtin_va_arg(v, int));    // expected-error {{identifier}} expected-warning {{extension}}
}

