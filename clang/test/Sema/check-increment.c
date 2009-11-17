// RUN: clang-cc -fsyntax-only -verify %s

int printf(const char *, ...);
typedef int *pint;
int main() {
   int a[5] = {0};
   pint p = a;
   p++;
   printf("%d\n", *p);
}
