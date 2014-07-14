// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64 %s -S -o /dev/null -DWARN -verify
// RUN: %clang_cc1 -triple x86_64 %s -S -o /dev/null -Werror -verify
void f() {
  asm("movaps %xmm3, (%esi, 2)");
// expected-note@1 {{instantiated into assembly here}}
#ifdef WARN
// expected-warning@-3 {{scale factor without index register is ignored}}
#else
// expected-error@-5 {{scale factor without index register is ignored}}
#endif
}

static unsigned var[1] = {};
void g(void) { asm volatile("movd %%xmm0, %0"
                            :
                            : "m"(var)); }
