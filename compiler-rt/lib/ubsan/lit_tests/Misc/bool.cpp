// RUN: %clang -fsanitize=bool %s -O3 -o %t && %t 2>&1 | FileCheck %s

unsigned char NotABool = 123;

int main(int argc, char **argv) {
  bool *p = (bool*)&NotABool;

  // CHECK: error: load of value 123, which is not a valid value for type 'bool'
  return *p;
}
