// RUN: %clang -fsanitize=bool %s -O3 -o %T/bool.exe && %T/bool.exe 2>&1 | FileCheck %s

unsigned char NotABool = 123;

int main(int argc, char **argv) {
  bool *p = (bool*)&NotABool;

  // FIXME: Provide a better source location here.
  // CHECK: bool.exe:0x{{[0-9a-f]*}}: runtime error: load of value 123, which is not a valid value for type 'bool'
  return *p;
}
