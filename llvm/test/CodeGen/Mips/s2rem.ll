; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -relocation-model=pic  < %s | FileCheck %s -check-prefix=NEG

; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -relocation-model=static  < %s | FileCheck %s -check-prefix=NEG

; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -relocation-model=pic  < %s | FileCheck %s 

; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -relocation-model=static  < %s | FileCheck %s 

@xi = common global i32 0, align 4
@x = common global float 0.000000e+00, align 4
@xd = common global double 0.000000e+00, align 8

; Function Attrs: nounwind
define void @it() #0 {
entry:
  %call = call i32 @i(i32 1)
  store i32 %call, i32* @xi, align 4
  ret void
; CHECK: 	.ent	it
; NEG: 	.ent	it
; CHECK: 	save	$ra, $16, $17, [[FS:[0-9]+]]
; NEG-NOT:      save	$ra, $16, $17, [[FS:[0-9]+]], $18
; CHECK: 	restore	$ra, $16, $17, [[FS]]
; NEG-NOT:      restore	$ra, $16, $17, [[FS:[0-9]+]], $18
; CHECK: 	.end	it
; NEG: 	.end	it
}

declare i32 @i(i32) #1

; Function Attrs: nounwind
define void @ft() #0 {
entry:
  %call = call float @f()
  store float %call, float* @x, align 4
  ret void
; CHECK: 	.ent	ft
; CHECK: 	save	$ra, $16, $17, [[FS:[0-9]+]], $18
; CHECK: 	restore	$ra, $16, $17, [[FS]], $18
; CHECK: 	.end	ft
}

declare float @f() #1

; Function Attrs: nounwind
define void @dt() #0 {
entry:
  %call = call double @d()
  store double %call, double* @xd, align 8
  ret void
; CHECK: 	.ent	dt
; CHECK: 	save	$ra, $16, $17, [[FS:[0-9]+]], $18
; CHECK: 	restore	$ra, $16, $17, [[FS]], $18
; CHECK: 	.end	dt
}

declare double @d() #1

; Function Attrs: nounwind
define void @fft() #0 {
entry:
  %0 = load float* @x, align 4
  %call = call float @ff(float %0)
  store float %call, float* @x, align 4
  ret void
; CHECK: 	.ent	fft
; CHECK: 	save	$ra, $16, $17, [[FS:[0-9]+]], $18
; CHECK: 	restore	$ra, $16, $17, [[FS]], $18
; CHECK: 	.end	fft
}

declare float @ff(float) #1

; Function Attrs: nounwind
define void @vft() #0 {
entry:
  %0 = load float* @x, align 4
  call void @vf(float %0)
  ret void
; CHECK: 	.ent	vft
; NEG: 	.ent	vft
; CHECK: 	save	$ra, $16, $17, [[FS:[0-9]+]]
; NEG-NOT:      save	$ra, $16, $17, [[FS:[0-9]+]], $18
; CHECK: 	restore	$ra, $16, $17, [[FS]]
; NEG-NOT:      restore	$ra, $16, $17, [[FS:[0-9]+]], $18
; CHECK: 	.end	vft
; NEG: 	.end	vft
}

declare void @vf(float) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }


