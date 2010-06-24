// RUN: %clang_cc1 -fsyntax-only %s

// rdar: // 8125274
static int a16[];  // expected-warning {{tentative array definition assumed to have one element}}

void f16(void) {
    extern int a16[];
}

