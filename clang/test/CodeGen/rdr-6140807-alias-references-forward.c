// RUN: clang -emit-llvm -o %t %s &&
// RUN: grep -e "@f = alias" %t | count 1 &&
// RUN: grep -e "bitcast (i32 (i32)\\* @f to i32 (float)\\*)" %t | count 1
// <rdar://problem/6140807>

int f(float) __attribute__((weak, alias("x")));

// Make sure we replace uses properly...
int y() {
  return f(1.);
}

int x(int) {
}
