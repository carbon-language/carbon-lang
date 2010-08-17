// RUN: %clang_cc1 -emit-llvm -x c++ < %s

void test0(int x) {
          if (x != 0) return;
}


// PR5211
void test1() {
  char *xpto;
  while ( true && xpto[0] );
}

// PR5514
int a;
void test2() { ++a+=10; }

// PR7892
int test3(const char*);
int test3g = test3(__PRETTY_FUNCTION__);
