// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name objc.m -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck %s

@interface A
- (void)bork:(int)msg;
@end

                      // CHECK: func
void func(A *a) {     // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+3]]:2 = #0 (HasCodeBefore = 0)
  if (a)
    [a bork:  20  ];  // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:20 = #1 (HasCodeBefore = 0)
}

@interface NSArray
+ (NSArray*) arrayWithObjects: (id) first, ...;
- (unsigned) count;
@end

                               // CHECK: func2
void func2(NSArray *array) {   // CHECK-NEXT: File 0, [[@LINE]]:28 -> [[@LINE+10]]:2 = #0 (HasCodeBefore = 0)
  int i = 0;
  for (NSArray *x in array) {  // CHECK-NEXT: File 0, [[@LINE]]:29 -> [[@LINE+6]]:4 = #1 (HasCodeBefore = 0)
    if (x) {                   // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = #2 (HasCodeBefore = 0)
      i = 1;
    } else {                   // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = (#1 - #2) (HasCodeBefore = 0)
      i = -1;
    }
  }
  i = 0;
}
