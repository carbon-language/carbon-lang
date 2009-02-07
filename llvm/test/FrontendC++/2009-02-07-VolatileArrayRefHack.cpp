// RUN: %llvmgxx -S %s -o - | grep {volatile load}
// PR3320

void test(volatile int *a) {
    // should be a volatile load.
    a[0];
}
