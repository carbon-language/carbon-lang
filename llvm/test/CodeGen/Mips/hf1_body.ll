; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips16  -relocation-model=pic -soft-float -mips16-hard-float < %s | FileCheck %s -check-prefix=picfp16

@x = external global float

; Function Attrs: nounwind
define void @v_sf(float %p) #0 {
entry:
  %p.addr = alloca float, align 4
  store float %p, float* %p.addr, align 4
  %0 = load float* %p.addr, align 4
  store float %0, float* @x, align 4
  ret void
}
; picfp16:	.ent	__fn_stub_v_sf
; picfp16:	.cpload  $25
; picfp16:	.set reorder
; picfp16:	.reloc 0,R_MIPS_NONE,v_sf
; picfp16: 	la $25,$__fn_local_v_sf
; picfp16: 	mfc1 $4,$f12
; picfp16: 	jr $25
; picfp16: 	.end	__fn_stub_v_sf
