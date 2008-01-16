// RUN: clang %s -emit-llvm

void foo(int x) {
switch (x) {
case 111111111111111111111111111111111111111:
bar();
}
}

