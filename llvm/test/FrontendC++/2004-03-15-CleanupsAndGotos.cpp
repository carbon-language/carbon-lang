// RUN: %llvmgxx -S %s -o - | llvm-as -o /dev/null

// Testcase from Bug 291

struct X {
  ~X();
};

void foo() {
  X v;

TryAgain:
  goto TryAgain;
}
