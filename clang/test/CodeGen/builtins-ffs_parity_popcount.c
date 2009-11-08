// RUN: clang-cc -emit-llvm -o - %s > %t
// RUN: not grep "__builtin" %t

#include <stdio.h>

void test(int M, long long N) {
  printf("%d %lld: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
         M, N,
         __builtin_ffs(M), __builtin_ffsl(M), __builtin_ffsll(M),
         __builtin_parity(M), __builtin_parityl(M), __builtin_parityll(M),
         __builtin_popcount(M), __builtin_popcountl(M), __builtin_popcountll(M),
         __builtin_ffs(N), __builtin_ffsl(N), __builtin_ffsll(N),
         __builtin_parity(N), __builtin_parityl(N), __builtin_parityll(N),
         __builtin_popcount(N), __builtin_popcountl(N), __builtin_popcountll(N));
}
