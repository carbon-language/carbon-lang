// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null

int test() {
  __complex__ double C;
  double D;
  C / D;
}
