// RUN: %llvmgxx -xc++ %s -c -o - | opt -die | llvm-dis | grep -v llvm.noinline | not grep cast

void foo(int*);

struct FOO {
  int X;
};

struct BAR : virtual FOO { BAR(); };

int testfn() {
  BAR B;
  foo(&B.X);
}
