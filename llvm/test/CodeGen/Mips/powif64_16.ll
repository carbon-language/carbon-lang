; RUN: llc -march=mipsel -mcpu=mips16 -mips16-hard-float -soft-float -relocation-model=static < %s | FileCheck %s 

@x = global double 4.500000e+00, align 8
@i = global i32 4, align 4
@y = common global double 0.000000e+00, align 8

; Function Attrs: nounwind optsize
define i32 @main() #0 {
entry:
  %0 = load double* @x, align 8, !tbaa !0
  %1 = load i32* @i, align 4, !tbaa !3
  %2 = tail call double @llvm.powi.f64(double %0, i32 %1)
; CHECK-NOT: .ent	__call_stub_fp_llvm.powi.f64
; CHECK-NOT: {{.*}} jal llvm.powi.f64 
  store double %2, double* @y, align 8, !tbaa !0
  ret i32 0
}

; Function Attrs: nounwind readonly
declare double @llvm.powi.f64(double, i32) #1

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind readonly }

!0 = metadata !{metadata !"double", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"int", metadata !1}
