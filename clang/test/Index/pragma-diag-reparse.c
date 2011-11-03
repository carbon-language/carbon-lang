#pragma clang diagnostic ignored "-Wtautological-compare"

int main (int argc, const char * argv[])
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  int x=0;
#pragma clang diagnostic pop

  return x;
}

void foo() { int b=0; while (b==b); }

// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_FAILONERROR=1 c-index-test -test-load-source-reparse 5 local \
// RUN:   %s -Wall -Werror | FileCheck %s

// CHECK: pragma-diag-reparse.c:7:7: VarDecl=x:7:7 (Definition) Extent=[7:3 - 7:10]
