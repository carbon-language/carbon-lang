// RUN: %clang_cc1 -triple armv7-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

id test0(void) {
  extern id test0_helper(void);
  // CHECK:      [[T0:%.*]] = call arm_aapcscc i8* @test0_helper()
  // CHECK-NEXT: ret i8* [[T0]]
  return test0_helper();
}

void test1(void) {
  extern id test1_helper(void);
  // CHECK:      [[T0:%.*]] = call arm_aapcscc i8* @test1_helper()
  // CHECK-NEXT: call void asm sideeffect "mov\09r7, r7
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]],
  // CHECK-NEXT: load
  // CHECK-NEXT: call void @objc_release
  // CHECK-NEXT: ret void
  id x = test1_helper();
}
