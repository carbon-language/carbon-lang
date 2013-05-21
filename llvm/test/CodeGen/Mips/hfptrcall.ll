; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -soft-float -mips16-hard-float -relocation-model=pic < %s | FileCheck %s -check-prefix=picel

@ptrsv = global float ()* @sv, align 4
@ptrdv = global double ()* @dv, align 4
@ptrscv = global { float, float } ()* @scv, align 4
@ptrdcv = global { double, double } ()* @dcv, align 4
@x = common global float 0.000000e+00, align 4
@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@xd = common global double 0.000000e+00, align 8
@xy = common global { float, float } zeroinitializer, align 4
@.str1 = private unnamed_addr constant [10 x i8] c"%f + %fi\0A\00", align 1
@xyd = common global { double, double } zeroinitializer, align 8

; Function Attrs: nounwind
define float @sv() #0 {
entry:
  ret float 1.000000e+01
}
; picel: 	.ent	sv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_sf)(${{[0-9]+}})
; picel:	.end	sv

; Function Attrs: nounwind
define double @dv() #0 {
entry:
  ret double 1.500000e+01
}

; picel: 	.ent	dv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_df)(${{[0-9]+}})
; picel:	.end	dv

; Function Attrs: nounwind
define { float, float } @scv() #0 {
entry:
  %retval = alloca { float, float }, align 4
  %real = getelementptr inbounds { float, float }* %retval, i32 0, i32 0
  %imag = getelementptr inbounds { float, float }* %retval, i32 0, i32 1
  store float 5.000000e+00, float* %real
  store float 9.900000e+01, float* %imag
  %0 = load { float, float }* %retval
  ret { float, float } %0
}

; picel: 	.ent	scv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_sc)(${{[0-9]+}})
; picel:	.end	scv

; Function Attrs: nounwind
define { double, double } @dcv() #0 {
entry:
  %retval = alloca { double, double }, align 8
  %real = getelementptr inbounds { double, double }* %retval, i32 0, i32 0
  %imag = getelementptr inbounds { double, double }* %retval, i32 0, i32 1
  store double 0x416BC8B0A0000000, double* %real
  store double 0x41CDCCB763800000, double* %imag
  %0 = load { double, double }* %retval
  ret { double, double } %0
}

; picel: 	.ent	dcv
; picel: 	lw	${{[0-9]+}}, %call16(__mips16_ret_dc)(${{[0-9]+}})
; picel:	.end	dcv


attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="true" }

