; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@iiii = global i32 25, align 4
@jjjj = global i32 35, align 4
@kkkk = global i32 100, align 4
@riii = common global i32 0, align 4
@rjjj = common global i32 0, align 4
@rkkk = common global i32 0, align 4

define void @test() nounwind {
entry:
; 16: 	.frame	$16,16,$ra
; 16: 	save 	$ra, $s0, $s1, 16
; 16: 	move	$16, $sp
; 16:	move	${{[0-9]+}}, $sp
; 16:	subu	$[[REGISTER:[0-9]+]], ${{[0-9]+}}, ${{[0-9]+}}
; 16:	move	$sp, $[[REGISTER]]
  %0 = load i32* @kkkk, align 4
  %1 = mul i32 %0, 25
  %2 = alloca i32, i32 %1, align 4
  %3 = load i32* @jjjj, align 4
  %4 = load i32* @iiii, align 4
  %arrayidx = getelementptr inbounds i32* %2, i32 %4
  store i32 %3, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32* %2, i32 %3
  store i32 %0, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32* %2, i32 %0
  store i32 %4, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32* %2, i32 25
  %5 = load i32* %arrayidx3, align 4
  store i32 %5, i32* @riii, align 4
  %arrayidx4 = getelementptr inbounds i32* %2, i32 35
  %6 = load i32* %arrayidx4, align 4
  store i32 %6, i32* @rjjj, align 4
  %arrayidx5 = getelementptr inbounds i32* %2, i32 100
  %7 = load i32* %arrayidx5, align 4
  store i32 %7, i32* @rkkk, align 4
; 16: 	move	$sp, $16
; 16:	restore 	$ra,  $s0, $s1, 16
  ret void
}


