// RUN: xcc -arch ppc -arch i386 -arch x86_64 %s -o %t &&
// RUN: %t | grep "Hello, World"

int main() {
  printf("Hello, World!\n");
  return 0;
}
