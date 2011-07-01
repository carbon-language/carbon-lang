; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

; Radar 7449043
%struct.int32x4_t = type { <4 x i32> }

define void @t() nounwind {
entry:
; CHECK: vmov.I64 q15, #0
; CHECK: vmov.32 d30[0],
; CHECK: vmov q8, q15
  %tmp = alloca %struct.int32x4_t, align 16
  call void asm sideeffect "vmov.I64 q15, #0\0Avmov.32 d30[0], $1\0Avmov ${0:q}, q15\0A", "=*w,r,~{d31},~{d30}"(%struct.int32x4_t* %tmp, i32 8192) nounwind
  ret void
}

; Radar 7457110
%struct.int32x2_t = type { <4 x i32> }

define void @t2() nounwind {
entry:
; CHECK: vmov d30, d16
; CHECK: vmov.32 r0, d30[0]
  %asmtmp2 = tail call i32 asm sideeffect "vmov d30, $1\0Avmov.32 $0, d30[0]\0A", "=r,w,~{d30}"(<2 x i32> undef) nounwind
  ret void
}

; Radar 9306086

%0 = type { <8 x i8>, <16 x i8>* }

define hidden void @conv4_8_E() nounwind {
entry:
%asmtmp31 = call %0 asm "vld1.u8  {$0}, [$1, :128]!\0A", "=w,=r,1"(<16 x i8>* undef) nounwind
unreachable
}

; Radar 9037836 & 9119939

define i32 @t3() nounwind {
entry:
tail call void asm sideeffect "flds s15, $0 \0A", "^Uv|m,~{s15}"(float 1.000000e+00) nounwind
ret i32 0
}

; Radar 9037836 & 9119939

@k.2126 = internal unnamed_addr global float 1.000000e+00
define i32 @t4() nounwind {
entry:
call void asm sideeffect "flds s15, $0 \0A", "*^Uv,~{s15}"(float* @k.2126) nounwind
ret i32 0
}

; Radar 9037836 & 9119939

define i32 @t5() nounwind {
entry:
call void asm sideeffect "flds s15, $0 \0A", "*^Uvm,~{s15}"(float* @k.2126) nounwind
ret i32 0
}

; Radar 9307836 & 9119939

define float @t6(float %y) nounwind {
entry:
; CHECK: t6
; CHECK: flds s15, s0
  %0 = tail call float asm "flds s15, $0", "=x"() nounwind
  ret float %0
}
