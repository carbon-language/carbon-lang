// RUN: %llvmgcc -w -S %s -o - | llvm-as -o /dev/null

void test(enum foo *X) {
}

