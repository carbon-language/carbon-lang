// RUN: clang-cc -analyze -checker-cfref %s --analyzer-store=basic -verify &&
// RUN: clang-cc -analyze -checker-cfref %s --analyzer-store=region -verify

unsigned foo();
typedef struct bf { unsigned x:2; } bf;
void bar() {
  bf y;
  *(unsigned*)&y = foo();
  y.x = 1;
}
