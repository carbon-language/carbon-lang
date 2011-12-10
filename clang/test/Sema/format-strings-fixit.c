// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -Wall -fixit %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror %t
// RUN: %clang_cc1 -E -o - %t | FileCheck %s

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

int printf(char const *, ...);

typedef __SIZE_TYPE__ size_t;
typedef __INTMAX_TYPE__ intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

void test() {
  // Basic types
  printf("%s", (int) 123);
  printf("abc%0f", "testing testing 123");
  printf("%u", (long) -12);
  printf("%p", 123);
  printf("%c\n", "x");
  printf("%c\n", 1.23);

  // Larger types
  printf("%+.2d", (unsigned long long) 123456);
  printf("%1d", (long double) 1.23);

  // Flag handling
  printf("%0+s", (unsigned) 31337); // 0 flag should stay
  printf("%#p", (void *) 0);
  printf("% +f", 1.23); // + flag should stay
  printf("%0-f", 1.23); // - flag should stay

  // Positional arguments
  printf("%1$f:%2$.*3$f:%4$.*3$f\n", 1, 2, 3, 4);

  // Precision
  printf("%10.5d", 1l); // (bug 7394)
  printf("%.2c", 'a');

  // Ignored flags
  printf("%0-f", 1.23);

  // Bad length modifiers
  printf("%hhs", "foo");
  printf("%1$zp", (void *)0);

  // Perserve the original formatting for unsigned integers.
  unsigned long val = 42;
  printf("%X", val);

  // size_t, etc.
  printf("%f", (size_t) 42);
  printf("%f", (intmax_t) 42);
  printf("%f", (uintmax_t) 42);
  printf("%f", (ptrdiff_t) 42);

  // string
  printf("%ld", "foo");
}

int scanf(char const *, ...);

void test2() {
  char str[100];
  short shortVar;
  unsigned short uShortVar;
  int intVar;
  unsigned uIntVar;
  float floatVar;
  double doubleVar;
  long double longDoubleVar;
  long longVar;
  unsigned long uLongVar;
  long long longLongVar;
  unsigned long long uLongLongVar;
  size_t sizeVar;
  intmax_t intmaxVar;
  uintmax_t uIntmaxVar;
  ptrdiff_t ptrdiffVar;

  scanf("%lf", str);
  scanf("%f", &shortVar);
  scanf("%f", &uShortVar);
  scanf("%p", &intVar);
  scanf("%Lf", &uIntVar);
  scanf("%ld", &floatVar);
  scanf("%f", &doubleVar);
  scanf("%d", &longDoubleVar);
  scanf("%f", &longVar);
  scanf("%f", &uLongVar);
  scanf("%f", &longLongVar);
  scanf("%f", &uLongLongVar);

  // Some named ints.
  scanf("%f", &sizeVar);
  scanf("%f", &intmaxVar);
  scanf("%f", &uIntmaxVar);
  scanf("%f", &ptrdiffVar);

  // Perserve the original formatting for unsigned integers.
  scanf("%o", &uLongVar);
  scanf("%x", &uLongVar);
  scanf("%X", &uLongVar);
}

// Validate the fixes...
// CHECK: printf("%d", (int) 123);
// CHECK: printf("abc%s", "testing testing 123");
// CHECK: printf("%ld", (long) -12);
// CHECK: printf("%d", 123);
// CHECK: printf("%s\n", "x");
// CHECK: printf("%f\n", 1.23);
// CHECK: printf("%.2llu", (unsigned long long) 123456);
// CHECK: printf("%1Lf", (long double) 1.23);
// CHECK: printf("%0u", (unsigned) 31337);
// CHECK: printf("%p", (void *) 0);
// CHECK: printf("%+f", 1.23);
// CHECK: printf("%-f", 1.23);
// CHECK: printf("%1$d:%2$.*3$d:%4$.*3$d\n", 1, 2, 3, 4);
// CHECK: printf("%10.5ld", 1l);
// CHECK: printf("%c", 'a');
// CHECK: printf("%-f", 1.23);
// CHECK: printf("%s", "foo");
// CHECK: printf("%1$p", (void *)0);
// CHECK: printf("%lX", val);
// CHECK: printf("%zu", (size_t) 42);
// CHECK: printf("%jd", (intmax_t) 42);
// CHECK: printf("%ju", (uintmax_t) 42);
// CHECK: printf("%td", (ptrdiff_t) 42);
// CHECK: printf("%s", "foo");

// CHECK: scanf("%s", str);
// CHECK: scanf("%hd", &shortVar);
// CHECK: scanf("%hu", &uShortVar);
// CHECK: scanf("%d", &intVar);
// CHECK: scanf("%u", &uIntVar);
// CHECK: scanf("%f", &floatVar);
// CHECK: scanf("%lf", &doubleVar);
// CHECK: scanf("%Lf", &longDoubleVar);
// CHECK: scanf("%ld", &longVar);
// CHECK: scanf("%lu", &uLongVar);
// CHECK: scanf("%lld", &longLongVar);
// CHECK: scanf("%llu", &uLongLongVar);
// CHECK: scanf("%zu", &sizeVar);
// CHECK: scanf("%jd", &intmaxVar);
// CHECK: scanf("%ju", &uIntmaxVar);
// CHECK: scanf("%td", &ptrdiffVar);
// CHECK: scanf("%lo", &uLongVar);
// CHECK: scanf("%lx", &uLongVar);
// CHECK: scanf("%lX", &uLongVar);
