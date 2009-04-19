// RUN: clang-cc -triple i686-pc-linux-gnu -emit-llvm -o - %s | grep thread_local | count 4

__thread int a;
extern __thread int b;
int c() { return &b; }
int d() {
  __thread static int e;
  __thread static union {float a; int b;} f = {.b = 1};
}

