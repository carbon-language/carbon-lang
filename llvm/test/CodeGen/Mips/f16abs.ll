; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -soft-float -mips16-hard-float -relocation-model=static < %s | FileCheck %s -check-prefix=static

@y = global double -1.450000e+00, align 8
@x = common global double 0.000000e+00, align 8

@y1 = common global float 0.000000e+00, align 4
@x1 = common global float 0.000000e+00, align 4



; Function Attrs: nounwind optsize
define i32 @main() #0 {
entry:
  %0 = load double* @y, align 8
  %call = tail call double @fabs(double %0) #2
  store double %call, double* @x, align 8
; static-NOT: 	.ent	__call_stub_fp_fabs
; static-NOT: 	jal fabs
  %1 = load float* @y1, align 4
  %call2 = tail call float @fabsf(float %1) #2
  store float %call2, float* @x1, align 4
; static-NOT: 	.ent	__call_stub_fp_fabsf
; static-NOT: 	jal fabsf
  ret i32 0
}

; Function Attrs: nounwind optsize readnone
declare double @fabs(double) #1

declare float @fabsf(float) #1

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind optsize readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #2 = { nounwind optsize readnone }



