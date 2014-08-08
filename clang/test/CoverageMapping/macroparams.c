// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams.c %s | FileCheck %s

#define OUTBUFSIZ 1024

#define put_byte(c) {outbuf[outcnt++]=c;}

/* Output a 16 bit value, lsb first */
#define put_short(w) \
{ if (outcnt < OUTBUFSIZ-2) { \
    outbuf[outcnt++] = ((w) & 0xff); \
    outbuf[outcnt++] = ((w) >> 8); \
  } else { \
    put_byte(((w) & 0xff)); \
    put_byte(((w) >> 8)); \
  } \
}

#define MACRO2(X2) (X2 + 2)
#define MACRO(X) MACRO2(x)

int main() {
  char outbuf[OUTBUFSIZ];
  unsigned outcnt = 0;
  put_short(2);
  unsigned short i = 42;
  put_short(i);
  do {
    int x = 0;
    MACRO(x);
  } while(0);
  return 0;
}

// CHECK: File 0, 8:22 -> 16:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 0, 9:16 -> 9:25 = #0 (HasCodeBefore = 0, Expanded file = 2)
// CHECK-NEXT: File 0, 9:29 -> 12:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 12:10 -> 15:4 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 0, 13:5 -> 13:13 = (#0 - #1) (HasCodeBefore = 0, Expanded file = 3)
// CHECK-NEXT: File 0, 13:14 -> 13:16 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 13:17 -> 13:26 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 0, 14:5 -> 14:13 = (#0 - #1) (HasCodeBefore = 0, Expanded file = 4)
// CHECK-NEXT: File 0, 14:14 -> 14:16 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:17 -> 14:24 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 21:12 -> 32:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 24:3 -> 24:12 = #0 (HasCodeBefore = 0, Expanded file = 0)
// CHECK-NEXT: Expansion,File 1, 26:3 -> 26:12 = #0 (HasCodeBefore = 0, Expanded file = 5)
// CHECK-NEXT: File 1, 26:13 -> 26:14 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 26:13 -> 26:14 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 26:13 -> 26:14 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 26:13 -> 26:14 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 27:6 -> 30:12 = (#0 + #3) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 29:5 -> 29:10 = (#0 + #3) (HasCodeBefore = 0, Expanded file = 10)
// CHECK-NEXT: File 2, 3:19 -> 3:23 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 3, 5:21 -> 5:42 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 4, 5:21 -> 5:42 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 8:22 -> 16:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 5, 9:16 -> 9:25 = #0 (HasCodeBefore = 0, Expanded file = 6)
// CHECK-NEXT: File 5, 9:29 -> 12:4 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 12:10 -> 15:4 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 5, 13:5 -> 13:13 = (#0 - #2) (HasCodeBefore = 0, Expanded file = 7)
// CHECK-NEXT: File 5, 13:14 -> 13:16 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 13:17 -> 13:26 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 5, 14:5 -> 14:13 = (#0 - #2) (HasCodeBefore = 0, Expanded file = 8)
// CHECK-NEXT: File 5, 14:14 -> 14:16 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 14:17 -> 14:24 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 6, 3:19 -> 3:23 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 7, 5:21 -> 5:42 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 8, 5:21 -> 5:42 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 9, 18:20 -> 18:28 = (#0 + #3) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 10, 19:18 -> 19:24 = (#0 + #3) (HasCodeBefore = 0, Expanded file = 9)
// CHECK-NEXT: File 10, 19:25 -> 19:26 = (#0 + #3) (HasCodeBefore = 0)
