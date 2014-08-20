// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name objc.m -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck %s

@interface A
- (void)bork:(int)msg;
@end

                      // CHECK: func
void func(A *a) {     // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+3]]:2 = #0 (HasCodeBefore = 0)
  if (a)
    [a bork:  20  ];  // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:20 = #1 (HasCodeBefore = 0)
}
