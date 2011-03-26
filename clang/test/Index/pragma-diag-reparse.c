// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local %s | FileCheck %s

int main (int argc, const char * argv[])
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  int x;
#pragma clang diagnostic pop

  return 0;
}

// CHECK: pragma-diag-reparse.c:7:7: VarDecl=x:7:7 (Definition) Extent=[7:3 - 7:8]
