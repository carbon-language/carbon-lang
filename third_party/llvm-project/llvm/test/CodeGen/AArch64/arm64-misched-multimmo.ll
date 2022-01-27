; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -enable-misched=0 -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s


@G1 = common dso_local global [100 x i32] zeroinitializer, align 4
@G2 = common dso_local global [100 x i32] zeroinitializer, align 4

; Check that no scheduling dependencies are created between the paired loads and the store during post-RA MI scheduling.
;
; CHECK-LABEL: # Machine code for function foo:
; CHECK: SU(2):   renamable $w{{[0-9]+}}, renamable $w{{[0-9]+}} = LDPWi
; CHECK: Successors:
; CHECK-NOT: ch SU(4)
; CHECK: SU(3)
; CHECK: SU(4):   STRWui $wzr, renamable $x{{[0-9]+}}
define dso_local i32 @foo() {
entry:
  %0 = load i32, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @G2, i64 0, i64 0), align 4
  %1 = load i32, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @G2, i64 0, i64 1), align 4
  store i32 0, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @G1, i64 0, i64 0), align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}
