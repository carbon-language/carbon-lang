; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s -mips-mixed-16-32  | FileCheck %s 
; RUN: llc  -march=mipsel -mcpu=mips32 -relocation-model=pic -O3 < %s -mips-mixed-16-32  | FileCheck %s 

define void @foo() #0 {
entry:
  ret void
}

; CHECK: 	.set	mips16
; CHECK:	.ent	foo
; CHECK:	jrc $ra
; CHECK:	.end	foo
attributes #0 = { nounwind "less-precise-fpmad"="false" "mips16" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
