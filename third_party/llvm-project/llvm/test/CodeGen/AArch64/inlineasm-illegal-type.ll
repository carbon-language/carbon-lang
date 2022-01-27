;RUN:  not llc -mtriple=aarch64-linux-gnu -mattr=-fp-armv8 < %s 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate output register for constraint '{d0}'
; CHECK: error: couldn't allocate output register for constraint 'w'
; CHECK: error: couldn't allocate input reg for constraint 'w'
; CHECK: error: couldn't allocate input reg for constraint 'w'

define hidden double @test1(double %xx) local_unnamed_addr #0 {
entry:
  %0 = tail call double asm "frintp ${0:d}, ${0:d}", "={d0}"()
  ret double %0
}

define hidden double @test2(double %xx) local_unnamed_addr #0 {
entry:
  %0 = tail call double asm "frintp ${0:d}, ${0:d}", "=w"()
  ret double %0
}

define void @test_vector_too_large(<8 x float>* nocapture readonly %0) {
entry:
  %m = load <8 x float>, <8 x float>* %0, align 16
  tail call void asm sideeffect "fadd.4s v4, v4, $0", "w,~{memory}"(<8 x float> %m)
  ret void
}

define void @test_vector_no_mvt(<9 x float>* nocapture readonly %0) {
entry:
  %m = load <9 x float>, <9 x float>* %0, align 16
  tail call void asm sideeffect "fadd.4s v4, v4, $0", "w,~{memory}"(<9 x float> %m)
  ret void
}
