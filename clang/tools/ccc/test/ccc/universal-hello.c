// RUN: xcc -ccc-no-clang -arch ppc -arch i386 -arch x86_64 %s -o %t &&
// RUN: %t | grep "Hello, World" &&

// RUN: xcc -ccc-no-clang -pipe -arch ppc -arch i386 -arch x86_64 %s -o %t &&
// RUN: %t | grep "Hello, World" &&

// Check that multiple archs are handled properly.
// RUN: xcc -ccc-print-phases -### -arch ppc -arch ppc %s | grep linker- | count 1 &&

// Check that -ccc-clang-archs is honored.
// RUN: xcc -ccc-clang-archs i386 -### -arch ppc -arch i386 %s 2>&1 | grep clang | count 1

int main() {
  printf("Hello, World!\n");
  return 0;
}
