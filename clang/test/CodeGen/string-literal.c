// RUN: clang -emit-llvm %s -o %t

int main() {
  char a[10] = "abc";
}
