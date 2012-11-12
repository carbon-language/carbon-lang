; RUN: opt -S -reassociate < %s | FileCheck %s

; t=-a; retval = t*7|t => t-a; retval => a*-7|t
define i32 @mulneg(i32 %a) nounwind uwtable ssp {
entry:
  %sub = sub nsw i32 0, %a
  %tmp1 = mul i32 %sub, 7
  %tmp2 = xor i32 %sub, %tmp1
  ret i32 %tmp2
; CHECK: entry
; CHECK: %tmp1 = mul i32 %a, -7 
; CHECK: ret
}
