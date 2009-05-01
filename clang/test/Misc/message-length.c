// RUN: clang -fsyntax-only -fmessage-length=72 %s

/* It's tough to verify the results of this test mechanically, since
   the length of the filename (and, therefore, how the word-wrapping
   behaves) changes depending on where the test-suite resides in the
   file system. */
void f(int, float, char, float);

void g() {
      int (*fp1)(int, float, short, float) = f;

  int (*fp2)(int, float, short, float) = f;
}
