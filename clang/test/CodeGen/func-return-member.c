// RUN: clang-cc -emit-llvm < %s 2>&1 | not grep 'cannot codegen this l-value expression yet'

struct frk { float _Complex c; int x; };
struct faz { struct frk f; };
struct fuz { struct faz f; };

extern struct fuz foo(void);

int X;
struct frk F;
float _Complex C;

void bar(void) {
  X = foo().f.f.x;
}

void bun(void) {
  F = foo().f.f;
}

void ban(void) {
  C = foo().f.f.c;
}
