// RUN: clang-cc -emit-llvm -o %t %s &&
// RUN: grep '@unreachable' %t | count 0

extern int unreachable();

int f0() {
  return 0;
  unreachable();
}

int f1(int i) {
  goto L0;
  int a = unreachable();
 L0:
  return 0;
}

int f2(int i) {
  goto L0;
  unreachable();
  int a;
  unreachable();
 L0:
  a = i + 1;
  return a;
}
