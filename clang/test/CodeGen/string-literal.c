// RUN: clang -emit-llvm -verify %s

int main() {
  char a[10] = "abc";
}
