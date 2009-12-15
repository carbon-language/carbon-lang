// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm-bc -o - | opt -std-compile-opts | llvm-dis | grep 'ret i32 6'

static enum { foo, bar = 1U } z;

int main (void)
{
  int r = 0;

  if (bar - 2 < 0)
    r += 4;
  if (foo - 1 < 0)
    r += 2;
  if (z - 1 < 0)
    r++;

  return r;
}

