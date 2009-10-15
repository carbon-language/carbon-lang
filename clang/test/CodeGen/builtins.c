// RUN: clang-cc -emit-llvm -o %t %s &&
// RUN: not grep __builtin %t

#include <stdio.h>

void p(char *str, int x) {
  printf("%s: %d\n", str, x);
}
void q(char *str, double x) {
  printf("%s: %f\n", str, x);
}

int main() {
  int N = random();
#define P(n,args) p(#n #args, __builtin_##n args)
#define Q(n,args) q(#n #args, __builtin_##n args)
#define V(n,args) p(#n #args, (__builtin_##n args, 0))
  P(types_compatible_p, (int, float));
  P(choose_expr, (0, 10, 20));
  P(constant_p, (sizeof(10)));
  P(expect, (N == 12, 0)); 
  V(prefetch, (&N));
  V(prefetch, (&N, 1));
  V(prefetch, (&N, 1, 0));
  
  // Numeric Constants

  Q(huge_val, ());
  Q(huge_valf, ());
  Q(huge_vall, ());
  Q(inf, ());
  Q(inff, ());
  Q(infl, ());

  // FIXME:
  // XXX note funny semantics for the (last) argument
  //  P(fpclassify, (FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, 1.0));
  //  P(isinf_sign, (1.0));

  Q(nan, (""));
  Q(nanf, (""));
  Q(nanl, (""));
  Q(nans, (""));
  Q(nan, ("10"));
  Q(nanf, ("10"));
  Q(nanl, ("10"));
  Q(nans, ("10"));

  P(isgreater, (1., 2.));
  P(isgreaterequal, (1., 2.));
  P(isless, (1., 2.));
  P(islessequal, (1., 2.));
  P(islessgreater, (1., 2.));
  P(isunordered, (1., 2.));

  P(isnan, (1.));

  // Bitwise & Numeric Functions

  P(abs, (N));

  P(clz, (N));
  P(clzl, (N));
  P(clzll, (N));
  P(ctz, (N));
  P(ctzl, (N));
  P(ctzll, (N));
  P(ffs, (N));
  P(ffsl, (N));
  P(ffsll, (N));
  P(parity, (N));
  P(parityl, (N));
  P(parityll, (N));
  P(popcount, (N));
  P(popcountl, (N));
  P(popcountll, (N));
  Q(powi, (1.2f, N));
  Q(powif, (1.2f, N));
  Q(powil, (1.2f, N));

  // Lib functions
  int a, b, n = random(); // Avoid optimizing out.
  char s0[10], s1[] = "Hello";
  V(strcat, (s0, s1));
  V(strcmp, (s0, s1));
  V(strncat, (s0, s1, n));
  V(strchr, (s0, s1[0]));
  V(strrchr, (s0, s1[0]));
  V(strcpy, (s0, s1));
  V(strncpy, (s0, s1, n));
  
  // Object size checking
  V(__memset_chk, (s0, 0, sizeof s0, n));
  V(__memcpy_chk, (s0, s1, sizeof s0, n));
  V(__memmove_chk, (s0, s1, sizeof s0, n));
  V(__mempcpy_chk, (s0, s1, sizeof s0, n));
  V(__strncpy_chk, (s0, s1, sizeof s0, n));
  V(__strcpy_chk, (s0, s1, n));
  s0[0] = 0;
  V(__strcat_chk, (s0, s1, n));
  P(object_size, (s0, 0));
  P(object_size, (s0, 1));
  P(object_size, (s0, 2));
  P(object_size, (s0, 3));

  // Whatever

  P(bswap32, (N));
  P(bswap64, (N));
  // FIXME
  // V(clear_cache, (&N, &N+1));
  V(trap, ());
  P(extract_return_addr, (&N));

  return 0;
}



void strcat() {}

void foo() {
 __builtin_strcat(0, 0);
}

