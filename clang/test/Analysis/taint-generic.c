// RUN: %clang_cc1  -analyze -analyzer-checker=experimental.security.taint,experimental.security.ArrayBoundV2 -verify %s

int scanf(const char *restrict format, ...);
int getchar(void);

#define BUFSIZE 10

int Buffer[BUFSIZE];
void bufferFoo1(void)
{
  int n;
  scanf("%d", &n);
  Buffer[n] = 1; // expected-warning {{Out of bound memory access }}
}
