//===-- io.c - IO routines for LLVM libc Library ------------------*- C -*-===//
// 
// A lot of this code is ripped gratuitously from glibc and libiberty.
//
//===----------------------------------------------------------------------===//

int putchar(int);

// The puts() function writes the string pointed to by s, followed by a 
// NEWLINE character, to the standard output stream stdout. On success the 
// number of characters written is returned; otherwise they return EOF.
//
int puts(const char *S) {
  const char *Str = S;
  while (*Str) putchar(*Str++);
  putchar('\n');
  return Str+1-S;
}
