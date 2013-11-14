; RUN: llc < %s -march=xcore | FileCheck %s
define void @a_val() nounwind {
  ret void
}
@b_val = constant i32 42, section ".cp.rodata"
@c_val = global i32 42

@a = alias void ()* @a_val
@b = alias i32* @b_val
@c = alias i32* @c_val

; CHECK-LABEL: a_addr:
; CHECK: ldap r11, a
; CHECK: retsp
define void ()* @a_addr() nounwind {
entry:
  ret void ()* @a
}

; CHECK-LABEL: b_addr:
; CHECK: ldaw r11, cp[b]
; CHECK: retsp
define i32 *@b_addr() nounwind {
entry:
  ret i32* @b
}

; CHECK-LABEL: c_addr:
; CHECK: ldaw r0, dp[c]
; CHECK: retsp
define i32 *@c_addr() nounwind {
entry:
  ret i32* @c
}
