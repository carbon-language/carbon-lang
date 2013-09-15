; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -soft-float -mips16-hard-float -relocation-model=static -mips32-function-mask=10 -mips-os16 < %s | FileCheck %s -check-prefix=fmask1

; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16 -soft-float -mips16-hard-float -relocation-model=static -mips32-function-mask=01 -mips-os16 < %s | FileCheck %s -check-prefix=fmask2 

; Function Attrs: nounwind optsize readnone
define void @foo1()  {
entry:
  ret void
; fmask1: .ent foo1
; fmask1: .set	noreorder
; fmask1: .set	nomacro
; fmask1: .set	noat
; fmask1: .set	at
; fmask1: .set	macro
; fmask1: .set	reorder
; fmask1: .end	foo1
; fmask2: .ent	foo1
; fmask2: save	{{.*}}
; fmask2: .end	foo1
}

; Function Attrs: nounwind optsize readnone
define void @foo2()  {
entry:
  ret void
; fmask2: .ent foo2
; fmask2: .set	noreorder
; fmask2: .set	nomacro
; fmask2: .set	noat
; fmask2: .set	at
; fmask2: .set	macro
; fmask2: .set	reorder
; fmask2: .end	foo2
; fmask1: .ent	foo2
; fmask1: save	{{.*}}
; fmask1: .end	foo2
}

; Function Attrs: nounwind optsize readnone
define void @foo3()  {
entry:
  ret void
; fmask1: .ent foo3
; fmask1: .set	noreorder
; fmask1: .set	nomacro
; fmask1: .set	noat
; fmask1: .set	at
; fmask1: .set	macro
; fmask1: .set	reorder
; fmask1: .end	foo3
; fmask2:  .ent	foo3
; fmask2:  save	{{.*}}
; fmask2:  .end	foo3
}

; Function Attrs: nounwind optsize readnone
define void @foo4()  {
entry:
  ret void
; fmask2: .ent foo4
; fmask2: .set	noreorder
; fmask2: .set	nomacro
; fmask2: .set	noat
; fmask2: .set	at
; fmask2: .set	macro
; fmask2: .set	reorder
; fmask2: .end	foo4
; fmask1: .ent	foo4
; fmask1: save	{{.*}}
; fmask1: .end	foo4
}


