; RUN: opt < %s -instcombine -S | FileCheck %s
; <rdar://problem/7530098>

define void @func(double* %rhi, double* %rlo, double %xh, double %xl, double %yh, double %yl) nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %tmp = fmul double %xh, 0x41A0000002000000      ; <double> [#uses=2]
  %tmp1 = fsub double %xh, %tmp                   ; <double> [#uses=1]
  %tmp2 = fadd double %tmp1, %tmp                 ; <double> [#uses=3]
  %tmp3 = fsub double %xh, %tmp2                  ; <double> [#uses=2]
  %tmp4 = fmul double %yh, 0x41A0000002000000     ; <double> [#uses=2]
  %tmp5 = fsub double %yh, %tmp4                  ; <double> [#uses=1]
  %tmp6 = fadd double %tmp5, %tmp4                ; <double> [#uses=3]
  %tmp7 = fsub double %yh, %tmp6                  ; <double> [#uses=2]
  %tmp8 = fmul double %xh, %yh                    ; <double> [#uses=3]
  %tmp9 = fmul double %tmp2, %tmp6                ; <double> [#uses=1]
  %tmp10 = fsub double %tmp9, %tmp8               ; <double> [#uses=1]
  %tmp11 = fmul double %tmp2, %tmp7               ; <double> [#uses=1]
  %tmp12 = fadd double %tmp10, %tmp11             ; <double> [#uses=1]
  %tmp13 = fmul double %tmp3, %tmp6               ; <double> [#uses=1]
  %tmp14 = fadd double %tmp12, %tmp13             ; <double> [#uses=1]
  %tmp15 = fmul double %tmp3, %tmp7               ; <double> [#uses=1]
  %tmp16 = fadd double %tmp14, %tmp15             ; <double> [#uses=1]
  %tmp17 = fmul double %xh, %yl                   ; <double> [#uses=1]
  %tmp18 = fmul double %xl, %yh                   ; <double> [#uses=1]
  %tmp19 = fadd double %tmp17, %tmp18             ; <double> [#uses=1]
  %tmp20 = fadd double %tmp19, %tmp16             ; <double> [#uses=2]
  %tmp21 = fadd double %tmp8, %tmp20              ; <double> [#uses=1]
  store double %tmp21, double* %rhi, align 8
  %tmp22 = load double* %rhi, align 8             ; <double> [#uses=1]
  %tmp23 = fsub double %tmp8, %tmp22              ; <double> [#uses=1]
  %tmp24 = fadd double %tmp23, %tmp20             ; <double> [#uses=1]

; CHECK: %tmp23 = fsub double %tmp8, %tmp21
; CHECK: %tmp24 = fadd double %tmp23, %tmp20

  store double %tmp24, double* %rlo, align 8
  ret void
}
