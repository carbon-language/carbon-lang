// RUN: %clang_cc1 %s -emit-llvm -o %t
// RUN: not grep call*__builtin %t

int G, H, I;
void foo(int P) {
  G = __builtin_clz(P);
  H = __builtin_ctz(P);
  I = __builtin_popcount(P);
}

long long g, h, i;
void fooll(float P) {
  g = __builtin_clzll(P);
  g = __builtin_clzll(P);
  h = __builtin_ctzll(P);
  i = __builtin_popcountll(P);
}
