; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

; CHECK: ;;#ASMSTART
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: ;;#ASMEND

define void @foo(i32 addrspace(5)* %ptr) #0 {
  %tmp = tail call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm "s_nop 0", "=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65"(i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2)
  %tmp2 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %tmp, 0
  store i32 %tmp2, i32 addrspace(5)* %ptr, align 4
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,768" }
