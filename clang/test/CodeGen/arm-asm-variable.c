// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -w -o - %s | FileCheck %s
#include <stdint.h>

#define ldrex_func(p, rl, rh) \
  __asm__ __volatile__( \
		       "ldrexd%[_rl], %[_rh], [%[_p]]" \
		       : [_rl] "=&r" (rl), [_rh] "=&r" (rh) \
		       : [_p] "p" (p) : "memory")

int64_t foo(int64_t v, volatile int64_t *p)
{
  register uint32_t rl asm("r1");
  register uint32_t rh asm("r2");

  int64_t r;
  uint32_t t;

  __asm__ __volatile__(							\
		       "ldrexd%[_rl], %[_rh], [%[_p]]"			\
		       : [_rl] "=&r" (rl), [_rh] "=&r" (rh)		\
		       : [_p] "p" (p) : "memory");

  // CHECK: %0 = call %0 asm sideeffect "ldrexd$0, $1, [$2]", "={r1},={r2},r,~{memory}"(i64*

  return r;
}
