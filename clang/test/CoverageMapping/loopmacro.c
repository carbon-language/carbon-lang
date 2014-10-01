// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name loopmacro.c %s | FileCheck %s

#   define HASH_BITS  15
#define MIN_MATCH  3
#define H_SHIFT  ((HASH_BITS+MIN_MATCH-1)/MIN_MATCH)
#define WMASK 0xFFFF
#define HASH_MASK 0xFFFF
#define UPDATE_HASH(h,c) (h = (((h)<<H_SHIFT) ^ (c)) & HASH_MASK)
#define INSERT_STRING(s, match_head) \
   (UPDATE_HASH(ins_h, window[(s) + MIN_MATCH-1]), \
    prev[(s) & WMASK] = match_head = head[ins_h], \
    head[ins_h] = (s))

int main() {                                // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+12]]:2 = #0 (HasCodeBefore = 0)
  int strstart = 0;
  int hash_head = 2;
  int prev_length = 5;
  int ins_h = 1;
  int prev[32] = { 0 };
  int head[32] = { 0 };
  int window[1024] = { 0 };
  do {                                     // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE+3]]:30 = (#0 + #1) (HasCodeBefore = 0)
      strstart++;
      INSERT_STRING(strstart, hash_head);  // CHECK-NEXT: Expansion,File 0, [[@LINE]]:7 -> [[@LINE]]:20 = (#0 + #1) (HasCodeBefore = 0, Expanded file = 1)
  } while (--prev_length != 0);
}
// CHECK-NEXT: File 0, 24:21 -> 24:29 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 24:21 -> 24:29 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 24:21 -> 24:29 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 24:31 -> 24:40 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 10:4 -> 12:23 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 10:5 -> 10:16 = (#0 + #1) (HasCodeBefore = 0, Expanded file = 2)
// CHECK-NEXT: File 1, 10:17 -> 10:22 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 10:17 -> 10:22 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 10:24 -> 10:32 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 10:33 -> 10:36 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 10:46 -> 10:49 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 2, 8:26 -> 8:66 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 2, 8:38 -> 8:45 = (#0 + #1) (HasCodeBefore = 0, Expanded file = 3)
// CHECK-NEXT: File 3, 5:18 -> 5:53 = (#0 + #1) (HasCodeBefore = 0)
