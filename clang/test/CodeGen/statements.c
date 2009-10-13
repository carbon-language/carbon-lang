// RUN: clang-cc < %s -emit-llvm

void test1(int x) {
switch (x) {
case 111111111111111111111111111111111111111:
bar();
}
}

// Mismatched type between return and function result.
int test2() { return; }
void test3() { return 4; }


void test4() {
bar:
baz:
blong:
bing:
 ;
static long x = &&bar; // - &&baz;
static long y = &&baz;
  &&bing;
  &&blong;
  if (y)
    goto *y;

  goto *x;
}

// PR3869
int test5(long long b) { goto *b; }

