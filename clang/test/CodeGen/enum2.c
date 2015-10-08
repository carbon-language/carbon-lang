// RUN: %clang_cc1 -triple i386-unknown-unknown %s -debug-info-kind=limited -emit-llvm -o /dev/null
int v;
enum e { MAX };

void foo (void)
{
  v = MAX;
}
