// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null


struct S { };

int xxxx(int a) {
  struct S comps[a];
  comps[0];
}

