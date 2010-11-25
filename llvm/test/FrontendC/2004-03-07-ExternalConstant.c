// RUN: %llvmgcc -xc %s -S -o - | grep constant

extern const int a[];   // 'a' should be marked constant even though it's external!
int foo () {
  return a[0];
}

