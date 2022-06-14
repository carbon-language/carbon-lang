; RUN: llc -mtriple thumbv6m-eabi      %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv8m.base-eabi %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv7a-eabi %s      -o - | FileCheck %s
; RUN: llc -mtriple thumbv7m-eabi      %s -o - | FileCheck %s --check-prefix=CHECK-PACBTI

; Check we don't emit PACBTI-M instructions for architectures
; that do not support them.
define hidden i32 @f(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %sub = sub nsw i32 1, %0
  %call = call i32 @g(i32 %sub)
  %add = add nsw i32 1, %call
  ret i32 %add
}
; CHECK-LABEL: f:
; CHECK-NOT:   bti

; CHECK-PACBTI-LABEL: f:
; CHECK-PACBTI:       pacbti
declare dso_local i32 @g(i32)

attributes #0 = { noinline nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
