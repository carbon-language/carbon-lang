// RUN: %llvmgxx -S %s -o - | llvm-as -o /dev/null

struct test {
  bool A : 1;
  bool B : 1;
};

void foo(test *T) {
  T->B = true;
}

