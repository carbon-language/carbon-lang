// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name label.cpp %s | FileCheck %s

void func() {
  int i = 0;
  for(i = 0; i < 10; ++i) {
    if(i < 5) {
      {
        x:
          int j = 1;
      }
      int m = 2;
    } else
      goto x;
    int k = 3;
  }
  static int j = 0;
  ++j;
  if(j == 1)
    goto x;
}

// CHECK: func
// CHECK-NEXT: File 0, 3:13 -> 20:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:14 -> 5:20 = (#0 + #3) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:22 -> 5:25 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:27 -> 15:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 6:15 -> 12:6 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 8:9 -> 14:14 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 13:7 -> 13:13 = (#1 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 16:3 -> 18:12 = ((#0 + #3) - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:5 -> 19:11 = #4 (HasCodeBefore = 0)

void test1(int x) {
  if(x == 0)
    goto a;
  goto b;
a:
b:
  x = x + 1;
}

// CHECK-NEXT: test1
// CHECK-NEXT: File 0, 33:19 -> 40:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 35:5 -> 35:11 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 36:3 -> 36:9 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 37:1 -> 37:2 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 38:1 -> 39:12 = #3 (HasCodeBefore = 0)

void test2(int x) {
  if(x == 0)
    goto a;
  else if(x == 1) goto b;
a:
b:
  x = x + 1;
}

// CHECK-NEXT: test2
// CHECK-NEXT: File 0, 49:19 -> 56:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 51:5 -> 51:11 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 52:8 -> 52:17 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 52:19 -> 52:25 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 53:1 -> 53:2 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 54:1 -> 55:12 = #4 (HasCodeBefore = 0)

int main() {
  int j = 0;
  for(int i = 0; i < 10; ++i) {
  a:
    if(i < 3)
      goto e;
    goto c;
  b:
    j = 2;
  c:
    j = 1;
  e: f: ;
  }
  func();
  test1(0);
  test2(2);
}

// CHECK-NEXT: main
// CHECK-NEXT: File 0, 66:12 -> 82:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 68:18 -> 68:24 = (#0 + #7) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 68:26 -> 68:29 = #7 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 68:31 -> 78:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 69:3 -> 70:13 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 71:7 -> 71:13 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 72:5 -> 72:11 = (#2 - #3) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 73:3 -> 74:10 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 75:3 -> 76:10 = #5 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 77:3 -> 77:4 = #6 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 77:6 -> 77:10 = #7 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 79:3 -> 81:11 = ((#0 + #7) - #1) (HasCodeBefore = 0)
