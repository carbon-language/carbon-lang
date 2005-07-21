// RUN: %llvmgxx -xc++ %s -c -o - | llvm-dis | not grep cast

void foo(int*);

struct FOO {
  int X;
};

struct BAR : virtual FOO { BAR(); };

int testfn() {
  BAR B;
  foo(&B.X);
}
