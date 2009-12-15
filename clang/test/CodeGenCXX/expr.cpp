// RUN: %clang_cc1 -emit-llvm -x c++ < %s

void test0(int x) {
          if (x != 0) return;
}


// PR5211
void test1() {
  char *xpto;
  while ( true && xpto[0] );
}
