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

