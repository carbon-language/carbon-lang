// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* test that locals are renamed with . notation */

void abc(void *);

void Test5(double X) {
  abc(&X);
  {
    int X;
    abc(&X);
    {
      float X;
      abc(&X);
    }
  }
}

