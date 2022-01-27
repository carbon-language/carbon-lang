; RUN: llc -mtriple=i686-linux -pre-RA-sched=source < %s | FileCheck %s
; RUN: opt -disable-output -debugify < %s

declare void @error(i32 %i, i32 %a, i32 %b)

define i32 @test_ifchains(i32 %i, i32* %a, i32 %b) {
; Test a chain of ifs, where the block guarded by the if is error handling code
; that is not expected to run.
; CHECK-LABEL: test_ifchains:
; CHECK: %entry
; CHECK-NOT: .p2align
; CHECK: %else1
; CHECK-NOT: .p2align
; CHECK: %else2
; CHECK-NOT: .p2align
; CHECK: %else3
; CHECK-NOT: .p2align
; CHECK: %else4
; CHECK-NOT: .p2align
; CHECK: %exit
; CHECK: %then1
; CHECK: %then2
; CHECK: %then3
; CHECK: %then4
; CHECK: %then5

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then1, label %else1, !prof !0

then1:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else1

else1:
  %gep2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %gep2
  %cond2 = icmp ugt i32 %val2, 2
  br i1 %cond2, label %then2, label %else2, !prof !0

then2:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else2

else2:
  %gep3 = getelementptr i32, i32* %a, i32 3
  %val3 = load i32, i32* %gep3
  %cond3 = icmp ugt i32 %val3, 3
  br i1 %cond3, label %then3, label %else3, !prof !0

then3:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else3

else3:
  %gep4 = getelementptr i32, i32* %a, i32 4
  %val4 = load i32, i32* %gep4
  %cond4 = icmp ugt i32 %val4, 4
  br i1 %cond4, label %then4, label %else4, !prof !0

then4:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else4

else4:
  %gep5 = getelementptr i32, i32* %a, i32 3
  %val5 = load i32, i32* %gep5
  %cond5 = icmp ugt i32 %val5, 3
  br i1 %cond5, label %then5, label %exit, !prof !0

then5:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %exit

exit:
  ret i32 %b
}

define i32 @test_loop_cold_blocks(i32 %i, i32* %a) {
; Check that we sink cold loop blocks after the hot loop body.
; CHECK-LABEL: test_loop_cold_blocks:
; CHECK: %entry
; CHECK: .p2align
; CHECK: %body1
; CHECK: %body2
; CHECK: %body3
; CHECK-NOT: .p2align
; CHECK: %unlikely1
; CHECK-NOT: .p2align
; CHECK: %unlikely2
; CHECK: %exit

entry:
  br label %body1

body1:
  %iv = phi i32 [ 0, %entry ], [ %next, %body3 ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body3 ]
  %unlikelycond1 = icmp slt i32 %base, 42
  br i1 %unlikelycond1, label %unlikely1, label %body2, !prof !0

unlikely1:
  call void @error(i32 %i, i32 1, i32 %base)
  br label %body2

body2:
  %unlikelycond2 = icmp sgt i32 %base, 21
  br i1 %unlikelycond2, label %unlikely2, label %body3, !prof !0

unlikely2:
  call void @error(i32 %i, i32 2, i32 %base)
  br label %body3

body3:
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body1

exit:
  ret i32 %sum
}

!0 = !{!"branch_weights", i32 1, i32 64}

define i32 @test_loop_early_exits(i32 %i, i32* %a) {
; Check that we sink early exit blocks out of loop bodies.
; CHECK-LABEL: test_loop_early_exits:
; CHECK: %entry
; CHECK: %body1
; CHECK: %body2
; CHECK: %body3
; CHECK: %body4
; CHECK: %exit
; CHECK: %bail1
; CHECK: %bail2
; CHECK: %bail3

entry:
  br label %body1

body1:
  %iv = phi i32 [ 0, %entry ], [ %next, %body4 ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body4 ]
  %bailcond1 = icmp eq i32 %base, 42
  br i1 %bailcond1, label %bail1, label %body2

bail1:
  ret i32 -1

body2:
  %bailcond2 = icmp eq i32 %base, 43
  br i1 %bailcond2, label %bail2, label %body3

bail2:
  ret i32 -2

body3:
  %bailcond3 = icmp eq i32 %base, 44
  br i1 %bailcond3, label %bail3, label %body4

bail3:
  ret i32 -3

body4:
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body1

exit:
  ret i32 %sum
}

; Tail duplication during layout can entirely remove body0 by duplicating it
; into the entry block and into body1. This is a good thing but it isn't what
; this test is looking for. So to make the blocks longer so they don't get
; duplicated, we add some calls to dummy.
declare void @dummy()

define i32 @test_loop_rotate(i32 %i, i32* %a) {
; Check that we rotate conditional exits from the loop to the bottom of the
; loop, eliminating unconditional branches to the top.
; CHECK-LABEL: test_loop_rotate:
; CHECK: %entry
; CHECK: %body0
; CHECK: %body1
; CHECK: %exit

entry:
  br label %body0

body0:
  %iv = phi i32 [ 0, %entry ], [ %next, %body1 ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body1 ]
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  call void @dummy()
  call void @dummy()
  br i1 %exitcond, label %exit, label %body1

body1:
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %bailcond1 = icmp eq i32 %sum, 42
  br label %body0

exit:
  ret i32 %base
}

define i32 @test_no_loop_rotate(i32 %i, i32* %a) {
; Check that we don't try to rotate a loop which is already laid out with
; fallthrough opportunities into the top and out of the bottom.
; CHECK-LABEL: test_no_loop_rotate:
; CHECK: %entry
; CHECK: %body0
; CHECK: %body1
; CHECK: %exit

entry:
  br label %body0

body0:
  %iv = phi i32 [ 0, %entry ], [ %next, %body1 ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body1 ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %bailcond1 = icmp eq i32 %sum, 42
  br i1 %bailcond1, label %exit, label %body1

body1:
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body0

exit:
  ret i32 %base
}

define i32 @test_loop_align(i32 %i, i32* %a) {
; Check that we provide basic loop body alignment with the block placement
; pass.
; CHECK-LABEL: test_loop_align:
; CHECK: %entry
; CHECK: .p2align [[ALIGN:[0-9]+]],
; CHECK-NEXT: %body
; CHECK: %exit

entry:
  br label %body

body:
  %iv = phi i32 [ 0, %entry ], [ %next, %body ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body

exit:
  ret i32 %sum
}

define i32 @test_nested_loop_align(i32 %i, i32* %a, i32* %b) {
; Check that we provide nested loop body alignment.
; CHECK-LABEL: test_nested_loop_align:
; CHECK: %entry
; CHECK: .p2align [[ALIGN]],
; CHECK-NEXT: %loop.body.1
; CHECK: .p2align [[ALIGN]],
; CHECK-NEXT: %inner.loop.body
; CHECK-NOT: .p2align
; CHECK: %exit

entry:
  br label %loop.body.1

loop.body.1:
  %iv = phi i32 [ 0, %entry ], [ %next, %loop.body.2 ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %bidx = load i32, i32* %arrayidx
  br label %inner.loop.body

inner.loop.body:
  %inner.iv = phi i32 [ 0, %loop.body.1 ], [ %inner.next, %inner.loop.body ]
  %base = phi i32 [ 0, %loop.body.1 ], [ %sum, %inner.loop.body ]
  %scaled_idx = mul i32 %bidx, %iv
  %inner.arrayidx = getelementptr inbounds i32, i32* %b, i32 %scaled_idx
  %0 = load i32, i32* %inner.arrayidx
  %sum = add nsw i32 %0, %base
  %inner.next = add i32 %iv, 1
  %inner.exitcond = icmp eq i32 %inner.next, %i
  br i1 %inner.exitcond, label %loop.body.2, label %inner.loop.body

loop.body.2:
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %loop.body.1

exit:
  ret i32 %sum
}

define void @unnatural_cfg1() {
; Test that we can handle a loop with an inner unnatural loop at the end of
; a function. This is a gross CFG reduced out of the single source GCC.
; CHECK-LABEL: unnatural_cfg1
; CHECK: %entry
; CHECK: %loop.header
; CHECK: %loop.body2
; CHECK: %loop.body3

entry:
  br label %loop.header

loop.header:
  br label %loop.body1

loop.body1:
  br i1 undef, label %loop.body3, label %loop.body2

loop.body2:
  %ptr = load i32*, i32** undef, align 4
  br label %loop.body3

loop.body3:
  %myptr = phi i32* [ %ptr2, %loop.body5 ], [ %ptr, %loop.body2 ], [ undef, %loop.body1 ]
  %bcmyptr = bitcast i32* %myptr to i32*
  %val = load i32, i32* %bcmyptr, align 4
  %comp = icmp eq i32 %val, 48
  br i1 %comp, label %loop.body4, label %loop.body5

loop.body4:
  br i1 undef, label %loop.header, label %loop.body5

loop.body5:
  %ptr2 = load i32*, i32** undef, align 4
  br label %loop.body3
}

define void @unnatural_cfg2(i32* %p0, i32 %a0) {
; Test that we can handle a loop with a nested natural loop *and* an unnatural
; loop. This was reduced from a crash on block placement when run over
; single-source GCC.
; CHECK-LABEL: unnatural_cfg2
; CHECK: %entry
; CHECK: %loop.header
; CHECK: %loop.body1
; CHECK: %loop.body2
; CHECK: %loop.body4
; CHECK: %loop.inner2.begin
; CHECK: %loop.inner2.begin
; CHECK: %loop.body3
; CHECK: %loop.inner1.begin
; CHECK: %bail

entry:
  br label %loop.header

loop.header:
  %comp0 = icmp eq i32* %p0, null
  br i1 %comp0, label %bail, label %loop.body1

loop.body1:
  %val0 = load i32*, i32** undef, align 4
  br i1 undef, label %loop.body2, label %loop.inner1.begin

loop.body2:
  br i1 undef, label %loop.body4, label %loop.body3

loop.body3:
  %ptr1 = getelementptr inbounds i32, i32* %val0, i32 0
  %castptr1 = bitcast i32* %ptr1 to i32**
  %val1 = load i32*, i32** %castptr1, align 4
  br label %loop.inner1.begin

loop.inner1.begin:
  %valphi = phi i32* [ %val2, %loop.inner1.end ], [ %val1, %loop.body3 ], [ %val0, %loop.body1 ]
  %castval = bitcast i32* %valphi to i32*
  %comp1 = icmp eq i32 %a0, 48
  br i1 %comp1, label %loop.inner1.end, label %loop.body4

loop.inner1.end:
  %ptr2 = getelementptr inbounds i32, i32* %valphi, i32 0
  %castptr2 = bitcast i32* %ptr2 to i32**
  %val2 = load i32*, i32** %castptr2, align 4
  br label %loop.inner1.begin

loop.body4.dead:
  br label %loop.body4

loop.body4:
  %comp2 = icmp ult i32 %a0, 3
  br i1 %comp2, label %loop.inner2.begin, label %loop.end

loop.inner2.begin:
  br i1 false, label %loop.end, label %loop.inner2.end

loop.inner2.end:
  %comp3 = icmp eq i32 %a0, 1769472
  br i1 %comp3, label %loop.end, label %loop.inner2.begin

loop.end:
  br label %loop.header

bail:
  unreachable
}

define i32 @problematic_switch() {
; This function's CFG caused overlow in the machine branch probability
; calculation, triggering asserts. Make sure we don't crash on it.
; CHECK: problematic_switch

entry:
  switch i32 undef, label %exit [
    i32 879, label %bogus
    i32 877, label %step
    i32 876, label %step
    i32 875, label %step
    i32 874, label %step
    i32 873, label %step
    i32 872, label %step
    i32 868, label %step
    i32 867, label %step
    i32 866, label %step
    i32 861, label %step
    i32 860, label %step
    i32 856, label %step
    i32 855, label %step
    i32 854, label %step
    i32 831, label %step
    i32 830, label %step
    i32 829, label %step
    i32 828, label %step
    i32 815, label %step
    i32 814, label %step
    i32 811, label %step
    i32 806, label %step
    i32 805, label %step
    i32 804, label %step
    i32 803, label %step
    i32 802, label %step
    i32 801, label %step
    i32 800, label %step
    i32 799, label %step
    i32 798, label %step
    i32 797, label %step
    i32 796, label %step
    i32 795, label %step
  ]
bogus:
  unreachable
step:
  br label %exit
exit:
  %merge = phi i32 [ 3, %step ], [ 6, %entry ]
  ret i32 %merge
}

define void @fpcmp_unanalyzable_branch(i1 %cond, double %a0) {
; This function's CFG contains an once-unanalyzable branch (une on floating
; points). As now it becomes analyzable, we should get best layout in which each
; edge in 'entry' -> 'entry.if.then_crit_edge' -> 'if.then' -> 'if.end' is
; fall-through.
; CHECK-LABEL: fpcmp_unanalyzable_branch:
; CHECK:       # %bb.0: # %entry
; CHECK:       # %bb.1: # %entry.if.then_crit_edge
; CHECK:       .LBB10_5: # %if.then
; CHECK:       .LBB10_6: # %if.end
; CHECK:       # %bb.3: # %exit
; CHECK:       jne .LBB10_4
; CHECK-NEXT:  jnp .LBB10_6
; CHECK:       jmp .LBB10_5

entry:
; Note that this branch must be strongly biased toward
; 'entry.if.then_crit_edge' to ensure that we would try to form a chain for
; 'entry' -> 'entry.if.then_crit_edge' -> 'if.then' -> 'if.end'.
  br i1 %cond, label %entry.if.then_crit_edge, label %lor.lhs.false, !prof !1

entry.if.then_crit_edge:
  %.pre14 = load i8, i8* undef, align 1
  br label %if.then

lor.lhs.false:
  br i1 undef, label %if.end, label %exit

exit:
  %cmp.i = fcmp une double 0.000000e+00, %a0
  br i1 %cmp.i, label %if.then, label %if.end, !prof !3

if.then:
  %0 = phi i8 [ %.pre14, %entry.if.then_crit_edge ], [ undef, %exit ]
  %1 = and i8 %0, 1
  store i8 %1, i8* undef, align 4
  br label %if.end

if.end:
  ret void
}

!1 = !{!"branch_weights", i32 1000, i32 1}
!3 = !{!"branch_weights", i32 1, i32 1000}

declare i32 @f()
declare i32 @g()
declare i32 @h(i32 %x)

define i32 @test_global_cfg_break_profitability() {
; Check that our metrics for the profitability of a CFG break are global rather
; than local. A successor may be very hot, but if the current block isn't, it
; doesn't matter. Within this test the 'then' block is slightly warmer than the
; 'else' block, but not nearly enough to merit merging it with the exit block
; even though the probability of 'then' branching to the 'exit' block is very
; high.
; CHECK: test_global_cfg_break_profitability
; CHECK: calll {{_?}}f
; CHECK: calll {{_?}}g
; CHECK: calll {{_?}}h
; CHECK: ret

entry:
  br i1 undef, label %then, label %else, !prof !2

then:
  %then.result = call i32 @f()
  br label %exit

else:
  %else.result = call i32 @g()
  br label %exit

exit:
  %result = phi i32 [ %then.result, %then ], [ %else.result, %else ]
  %result2 = call i32 @h(i32 %result)
  ret i32 %result
}

!2 = !{!"branch_weights", i32 3, i32 1}

declare i32 @__gxx_personality_v0(...)

define void @test_eh_lpad_successor() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; Some times the landing pad ends up as the first successor of an invoke block.
; When this happens, a strange result used to fall out of updateTerminators: we
; didn't correctly locate the fallthrough successor, assuming blindly that the
; first one was the fallthrough successor. As a result, we would add an
; erroneous jump to the landing pad thinking *that* was the default successor.
; CHECK-LABEL: test_eh_lpad_successor
; CHECK: %entry
; CHECK-NOT: jmp
; CHECK: %loop

entry:
  invoke i32 @f() to label %preheader unwind label %lpad

preheader:
  br label %loop

lpad:
  %lpad.val = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %lpad.val

loop:
  br label %loop
}

declare void @fake_throw() noreturn

define void @test_eh_throw() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; For blocks containing a 'throw' (or similar functionality), we have
; a no-return invoke. In this case, only EH successors will exist, and
; fallthrough simply won't occur. Make sure we don't crash trying to update
; terminators for such constructs.
;
; CHECK-LABEL: test_eh_throw
; CHECK: %entry
; CHECK: %cleanup

entry:
  invoke void @fake_throw() to label %continue unwind label %cleanup

continue:
  unreachable

cleanup:
  %0 = landingpad { i8*, i32 }
          cleanup
  unreachable
}

define void @test_unnatural_cfg_backwards_inner_loop() {
; Test that when we encounter an unnatural CFG structure after having formed
; a chain for an inner loop which happened to be laid out backwards we don't
; attempt to merge onto the wrong end of the inner loop just because we find it
; first. This was reduced from a crasher in GCC's single source.
;
; CHECK-LABEL: test_unnatural_cfg_backwards_inner_loop
; CHECK: %entry
; CHECK: %loop2b
; CHECK: %loop3

entry:
  br i1 undef, label %loop2a, label %body

body:
  br label %loop2a

loop1:
  %next.load = load i32*, i32** undef
  br i1 %comp.a, label %loop2a, label %loop2b

loop2a:
  %var = phi i32* [ null, %entry ], [ null, %body ], [ %next.phi, %loop1 ]
  %next.var = phi i32* [ null, %entry ], [ undef, %body ], [ %next.load, %loop1 ]
  %comp.a = icmp eq i32* %var, null
  br label %loop3

loop2b:
  %gep = getelementptr inbounds i32, i32* %var.phi, i32 0
  %next.ptr = bitcast i32* %gep to i32**
  store i32* %next.phi, i32** %next.ptr
  br label %loop3

loop3:
  %var.phi = phi i32* [ %next.phi, %loop2b ], [ %var, %loop2a ]
  %next.phi = phi i32* [ %next.load, %loop2b ], [ %next.var, %loop2a ]
  br label %loop1
}

define void @unanalyzable_branch_to_loop_header(double %a0) {
; Ensure that we can handle unanalyzable branches into loop headers. We
; pre-form chains for unanalyzable branches, and will find the tail end of that
; at the start of the loop. This function uses floating point comparison
; fallthrough because that happens to always produce unanalyzable branches on
; x86.
;
; CHECK-LABEL: unanalyzable_branch_to_loop_header
; CHECK: %entry
; CHECK: %loop
; CHECK: %exit

entry:
  %cmp = fcmp une double 0.000000e+00, %a0
  br i1 %cmp, label %loop, label %exit

loop:
  %cond = icmp eq i8 undef, 42
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

define void @unanalyzable_branch_to_best_succ(i1 %cond, double %a0) {
; Ensure that we can handle unanalyzable branches where the destination block
; gets selected as the optimal successor to merge.
;
; This branch is now analyzable and hence the destination block becomes the
; hotter one. The right order is entry->bar->exit->foo.
;
; CHECK-LABEL: unanalyzable_branch_to_best_succ
; CHECK: %entry
; CHECK: %bar
; CHECK: %exit
; CHECK: %foo

entry:
  ; Bias this branch toward bar to ensure we form that chain.
  br i1 %cond, label %bar, label %foo, !prof !1

foo:
  %cmp = fcmp une double 0.000000e+00, %a0
  br i1 %cmp, label %bar, label %exit

bar:
  call i32 @f()
  br label %exit

exit:
  ret void
}

define void @unanalyzable_branch_to_free_block(float %x) {
; Ensure that we can handle unanalyzable branches where the destination block
; gets selected as the best free block in the CFG.
;
; CHECK-LABEL: unanalyzable_branch_to_free_block
; CHECK: %entry
; CHECK: %a
; CHECK: %b
; CHECK: %c
; CHECK: %exit

entry:
  br i1 undef, label %a, label %b

a:
  call i32 @f()
  br label %c

b:
  %cmp = fcmp une float %x, 0.0
  br i1 %cmp, label %c, label %exit

c:
  call i32 @g()
  br label %exit

exit:
  ret void
}

define void @many_unanalyzable_branches() {
; Ensure that we don't crash as we're building up many unanalyzable branches,
; blocks, and loops.
;
; CHECK-LABEL: many_unanalyzable_branches
; CHECK: %entry
; CHECK: %exit

entry:
  br label %0

  %val0 = load volatile float, float* undef
  %cmp0 = fcmp une float %val0, 0.0
  br i1 %cmp0, label %1, label %0
  %val1 = load volatile float, float* undef
  %cmp1 = fcmp une float %val1, 0.0
  br i1 %cmp1, label %2, label %1
  %val2 = load volatile float, float* undef
  %cmp2 = fcmp une float %val2, 0.0
  br i1 %cmp2, label %3, label %2
  %val3 = load volatile float, float* undef
  %cmp3 = fcmp une float %val3, 0.0
  br i1 %cmp3, label %4, label %3
  %val4 = load volatile float, float* undef
  %cmp4 = fcmp une float %val4, 0.0
  br i1 %cmp4, label %5, label %4
  %val5 = load volatile float, float* undef
  %cmp5 = fcmp une float %val5, 0.0
  br i1 %cmp5, label %6, label %5
  %val6 = load volatile float, float* undef
  %cmp6 = fcmp une float %val6, 0.0
  br i1 %cmp6, label %7, label %6
  %val7 = load volatile float, float* undef
  %cmp7 = fcmp une float %val7, 0.0
  br i1 %cmp7, label %8, label %7
  %val8 = load volatile float, float* undef
  %cmp8 = fcmp une float %val8, 0.0
  br i1 %cmp8, label %9, label %8
  %val9 = load volatile float, float* undef
  %cmp9 = fcmp une float %val9, 0.0
  br i1 %cmp9, label %10, label %9
  %val10 = load volatile float, float* undef
  %cmp10 = fcmp une float %val10, 0.0
  br i1 %cmp10, label %11, label %10
  %val11 = load volatile float, float* undef
  %cmp11 = fcmp une float %val11, 0.0
  br i1 %cmp11, label %12, label %11
  %val12 = load volatile float, float* undef
  %cmp12 = fcmp une float %val12, 0.0
  br i1 %cmp12, label %13, label %12
  %val13 = load volatile float, float* undef
  %cmp13 = fcmp une float %val13, 0.0
  br i1 %cmp13, label %14, label %13
  %val14 = load volatile float, float* undef
  %cmp14 = fcmp une float %val14, 0.0
  br i1 %cmp14, label %15, label %14
  %val15 = load volatile float, float* undef
  %cmp15 = fcmp une float %val15, 0.0
  br i1 %cmp15, label %16, label %15
  %val16 = load volatile float, float* undef
  %cmp16 = fcmp une float %val16, 0.0
  br i1 %cmp16, label %17, label %16
  %val17 = load volatile float, float* undef
  %cmp17 = fcmp une float %val17, 0.0
  br i1 %cmp17, label %18, label %17
  %val18 = load volatile float, float* undef
  %cmp18 = fcmp une float %val18, 0.0
  br i1 %cmp18, label %19, label %18
  %val19 = load volatile float, float* undef
  %cmp19 = fcmp une float %val19, 0.0
  br i1 %cmp19, label %20, label %19
  %val20 = load volatile float, float* undef
  %cmp20 = fcmp une float %val20, 0.0
  br i1 %cmp20, label %21, label %20
  %val21 = load volatile float, float* undef
  %cmp21 = fcmp une float %val21, 0.0
  br i1 %cmp21, label %22, label %21
  %val22 = load volatile float, float* undef
  %cmp22 = fcmp une float %val22, 0.0
  br i1 %cmp22, label %23, label %22
  %val23 = load volatile float, float* undef
  %cmp23 = fcmp une float %val23, 0.0
  br i1 %cmp23, label %24, label %23
  %val24 = load volatile float, float* undef
  %cmp24 = fcmp une float %val24, 0.0
  br i1 %cmp24, label %25, label %24
  %val25 = load volatile float, float* undef
  %cmp25 = fcmp une float %val25, 0.0
  br i1 %cmp25, label %26, label %25
  %val26 = load volatile float, float* undef
  %cmp26 = fcmp une float %val26, 0.0
  br i1 %cmp26, label %27, label %26
  %val27 = load volatile float, float* undef
  %cmp27 = fcmp une float %val27, 0.0
  br i1 %cmp27, label %28, label %27
  %val28 = load volatile float, float* undef
  %cmp28 = fcmp une float %val28, 0.0
  br i1 %cmp28, label %29, label %28
  %val29 = load volatile float, float* undef
  %cmp29 = fcmp une float %val29, 0.0
  br i1 %cmp29, label %30, label %29
  %val30 = load volatile float, float* undef
  %cmp30 = fcmp une float %val30, 0.0
  br i1 %cmp30, label %31, label %30
  %val31 = load volatile float, float* undef
  %cmp31 = fcmp une float %val31, 0.0
  br i1 %cmp31, label %32, label %31
  %val32 = load volatile float, float* undef
  %cmp32 = fcmp une float %val32, 0.0
  br i1 %cmp32, label %33, label %32
  %val33 = load volatile float, float* undef
  %cmp33 = fcmp une float %val33, 0.0
  br i1 %cmp33, label %34, label %33
  %val34 = load volatile float, float* undef
  %cmp34 = fcmp une float %val34, 0.0
  br i1 %cmp34, label %35, label %34
  %val35 = load volatile float, float* undef
  %cmp35 = fcmp une float %val35, 0.0
  br i1 %cmp35, label %36, label %35
  %val36 = load volatile float, float* undef
  %cmp36 = fcmp une float %val36, 0.0
  br i1 %cmp36, label %37, label %36
  %val37 = load volatile float, float* undef
  %cmp37 = fcmp une float %val37, 0.0
  br i1 %cmp37, label %38, label %37
  %val38 = load volatile float, float* undef
  %cmp38 = fcmp une float %val38, 0.0
  br i1 %cmp38, label %39, label %38
  %val39 = load volatile float, float* undef
  %cmp39 = fcmp une float %val39, 0.0
  br i1 %cmp39, label %40, label %39
  %val40 = load volatile float, float* undef
  %cmp40 = fcmp une float %val40, 0.0
  br i1 %cmp40, label %41, label %40
  %val41 = load volatile float, float* undef
  %cmp41 = fcmp une float %val41, undef
  br i1 %cmp41, label %42, label %41
  %val42 = load volatile float, float* undef
  %cmp42 = fcmp une float %val42, 0.0
  br i1 %cmp42, label %43, label %42
  %val43 = load volatile float, float* undef
  %cmp43 = fcmp une float %val43, 0.0
  br i1 %cmp43, label %44, label %43
  %val44 = load volatile float, float* undef
  %cmp44 = fcmp une float %val44, 0.0
  br i1 %cmp44, label %45, label %44
  %val45 = load volatile float, float* undef
  %cmp45 = fcmp une float %val45, 0.0
  br i1 %cmp45, label %46, label %45
  %val46 = load volatile float, float* undef
  %cmp46 = fcmp une float %val46, 0.0
  br i1 %cmp46, label %47, label %46
  %val47 = load volatile float, float* undef
  %cmp47 = fcmp une float %val47, 0.0
  br i1 %cmp47, label %48, label %47
  %val48 = load volatile float, float* undef
  %cmp48 = fcmp une float %val48, 0.0
  br i1 %cmp48, label %49, label %48
  %val49 = load volatile float, float* undef
  %cmp49 = fcmp une float %val49, 0.0
  br i1 %cmp49, label %50, label %49
  %val50 = load volatile float, float* undef
  %cmp50 = fcmp une float %val50, 0.0
  br i1 %cmp50, label %51, label %50
  %val51 = load volatile float, float* undef
  %cmp51 = fcmp une float %val51, 0.0
  br i1 %cmp51, label %52, label %51
  %val52 = load volatile float, float* undef
  %cmp52 = fcmp une float %val52, 0.0
  br i1 %cmp52, label %53, label %52
  %val53 = load volatile float, float* undef
  %cmp53 = fcmp une float %val53, 0.0
  br i1 %cmp53, label %54, label %53
  %val54 = load volatile float, float* undef
  %cmp54 = fcmp une float %val54, 0.0
  br i1 %cmp54, label %55, label %54
  %val55 = load volatile float, float* undef
  %cmp55 = fcmp une float %val55, 0.0
  br i1 %cmp55, label %56, label %55
  %val56 = load volatile float, float* undef
  %cmp56 = fcmp une float %val56, 0.0
  br i1 %cmp56, label %57, label %56
  %val57 = load volatile float, float* undef
  %cmp57 = fcmp une float %val57, 0.0
  br i1 %cmp57, label %58, label %57
  %val58 = load volatile float, float* undef
  %cmp58 = fcmp une float %val58, 0.0
  br i1 %cmp58, label %59, label %58
  %val59 = load volatile float, float* undef
  %cmp59 = fcmp une float %val59, 0.0
  br i1 %cmp59, label %60, label %59
  %val60 = load volatile float, float* undef
  %cmp60 = fcmp une float %val60, 0.0
  br i1 %cmp60, label %61, label %60
  %val61 = load volatile float, float* undef
  %cmp61 = fcmp une float %val61, 0.0
  br i1 %cmp61, label %62, label %61
  %val62 = load volatile float, float* undef
  %cmp62 = fcmp une float %val62, 0.0
  br i1 %cmp62, label %63, label %62
  %val63 = load volatile float, float* undef
  %cmp63 = fcmp une float %val63, 0.0
  br i1 %cmp63, label %64, label %63
  %val64 = load volatile float, float* undef
  %cmp64 = fcmp une float %val64, 0.0
  br i1 %cmp64, label %65, label %64

  br label %exit
exit:
  ret void
}

define void @benchmark_heapsort(i32 %n, double* nocapture %ra) {
; This test case comes from the heapsort benchmark, and exemplifies several
; important aspects to block placement in the presence of loops:
; 1) Loop rotation needs to *ensure* that the desired exiting edge can be
;    a fallthrough.
; 2) The exiting edge from the loop which is rotated to be laid out at the
;    bottom of the loop needs to be exiting into the nearest enclosing loop (to
;    which there is an exit). Otherwise, we force that enclosing loop into
;    strange layouts that are siginificantly less efficient, often times making
;    it discontiguous.
;
; CHECK-LABEL: @benchmark_heapsort
; CHECK: %entry
; First rotated loop top.
; CHECK: .p2align
; CHECK: %while.end
; %for.cond gets completely tail-duplicated away.
; CHECK: %if.then
; CHECK: %if.else
; CHECK: %if.end10
; Second rotated loop top
; CHECK: %while.cond.outer
; Third rotated loop top
; CHECK: .p2align
; CHECK: %if.end20
; CHECK: %while.cond
; CHECK: %while.body
; CHECK: %land.lhs.true
; CHECK: %if.then19
; CHECK: %if.then24
; CHECK: %if.then8
; CHECK: ret

entry:
  %shr = ashr i32 %n, 1
  %add = add nsw i32 %shr, 1
  %arrayidx3 = getelementptr inbounds double, double* %ra, i64 1
  br label %for.cond

for.cond:
  %ir.0 = phi i32 [ %n, %entry ], [ %ir.1, %while.end ]
  %l.0 = phi i32 [ %add, %entry ], [ %l.1, %while.end ]
  %cmp = icmp sgt i32 %l.0, 1
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %dec = add nsw i32 %l.0, -1
  %idxprom = sext i32 %dec to i64
  %arrayidx = getelementptr inbounds double, double* %ra, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  br label %if.end10

if.else:
  %idxprom1 = sext i32 %ir.0 to i64
  %arrayidx2 = getelementptr inbounds double, double* %ra, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %2 = load double, double* %arrayidx3, align 8
  store double %2, double* %arrayidx2, align 8
  %dec6 = add nsw i32 %ir.0, -1
  %cmp7 = icmp eq i32 %dec6, 1
  br i1 %cmp7, label %if.then8, label %if.end10

if.then8:
  store double %1, double* %arrayidx3, align 8
  ret void

if.end10:
  %ir.1 = phi i32 [ %ir.0, %if.then ], [ %dec6, %if.else ]
  %l.1 = phi i32 [ %dec, %if.then ], [ %l.0, %if.else ]
  %rra.0 = phi double [ %0, %if.then ], [ %1, %if.else ]
  %add31 = add nsw i32 %ir.1, 1
  br label %while.cond.outer

while.cond.outer:
  %j.0.ph.in = phi i32 [ %l.1, %if.end10 ], [ %j.1, %if.then24 ]
  %j.0.ph = shl i32 %j.0.ph.in, 1
  br label %while.cond

while.cond:
  %j.0 = phi i32 [ %add31, %if.end20 ], [ %j.0.ph, %while.cond.outer ]
  %cmp11 = icmp sgt i32 %j.0, %ir.1
  br i1 %cmp11, label %while.end, label %while.body

while.body:
  %cmp12 = icmp slt i32 %j.0, %ir.1
  br i1 %cmp12, label %land.lhs.true, label %if.end20

land.lhs.true:
  %idxprom13 = sext i32 %j.0 to i64
  %arrayidx14 = getelementptr inbounds double, double* %ra, i64 %idxprom13
  %3 = load double, double* %arrayidx14, align 8
  %add15 = add nsw i32 %j.0, 1
  %idxprom16 = sext i32 %add15 to i64
  %arrayidx17 = getelementptr inbounds double, double* %ra, i64 %idxprom16
  %4 = load double, double* %arrayidx17, align 8
  %cmp18 = fcmp olt double %3, %4
  br i1 %cmp18, label %if.then19, label %if.end20

if.then19:
  br label %if.end20

if.end20:
  %j.1 = phi i32 [ %add15, %if.then19 ], [ %j.0, %land.lhs.true ], [ %j.0, %while.body ]
  %idxprom21 = sext i32 %j.1 to i64
  %arrayidx22 = getelementptr inbounds double, double* %ra, i64 %idxprom21
  %5 = load double, double* %arrayidx22, align 8
  %cmp23 = fcmp olt double %rra.0, %5
  br i1 %cmp23, label %if.then24, label %while.cond

if.then24:
  %idxprom27 = sext i32 %j.0.ph.in to i64
  %arrayidx28 = getelementptr inbounds double, double* %ra, i64 %idxprom27
  store double %5, double* %arrayidx28, align 8
  br label %while.cond.outer

while.end:
  %idxprom33 = sext i32 %j.0.ph.in to i64
  %arrayidx34 = getelementptr inbounds double, double* %ra, i64 %idxprom33
  store double %rra.0, double* %arrayidx34, align 8
  br label %for.cond
}

declare void @cold_function() cold

define i32 @test_cold_calls(i32* %a) {
; Test that edges to blocks post-dominated by cold calls are
; marked as not expected to be taken.  They should be laid out
; at the bottom.
; CHECK-LABEL: test_cold_calls:
; CHECK: %entry
; CHECK: %else
; CHECK: %exit
; CHECK: %then

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then, label %else

then:
  call void @cold_function()
  br label %exit

else:
  %gep2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %gep2
  br label %exit

exit:
  %ret = phi i32 [ %val1, %then ], [ %val2, %else ]
  ret i32 %ret
}

; Make sure we put landingpads out of the way.
declare i32 @pers(...)

declare i32 @foo();

declare i32 @bar();

define i32 @test_lp(i32 %a) personality i32 (...)* @pers {
; CHECK-LABEL: test_lp:
; CHECK: %entry
; CHECK: %hot
; CHECK: %then
; CHECK: %cold
; CHECK: %coldlp
; CHECK: %hotlp
; CHECK: %lpret
entry:
  %0 = icmp sgt i32 %a, 1
  br i1 %0, label %hot, label %cold, !prof !4

hot:
  %1 = invoke i32 @foo()
          to label %then unwind label %hotlp

cold:
  %2 = invoke i32 @bar()
          to label %then unwind label %coldlp

then:
  %3 = phi i32 [ %1, %hot ], [ %2, %cold ]
  ret i32 %3

hotlp:
  %4 = landingpad { i8*, i32 }
          cleanup
  br label %lpret

coldlp:
  %5 = landingpad { i8*, i32 }
          cleanup
  br label %lpret

lpret:
  %6 = phi i32 [-1, %hotlp], [-2, %coldlp]
  %7 = add i32 %6, 42
  ret i32 %7
}

!4 = !{!"branch_weights", i32 65536, i32 0}

; Make sure that ehpad are scheduled from the least probable one
; to the most probable one. See selectBestCandidateBlock as to why.
declare void @clean();

define void @test_flow_unwind() personality i32 (...)* @pers {
; CHECK-LABEL: test_flow_unwind:
; CHECK: %entry
; CHECK: %then
; CHECK: %exit
; CHECK: %innerlp
; CHECK: %outerlp
; CHECK: %outercleanup
entry:
  %0 = invoke i32 @foo()
          to label %then unwind label %outerlp

then:
  %1 = invoke i32 @bar()
          to label %exit unwind label %innerlp

exit:
  ret void

innerlp:
  %2 = landingpad { i8*, i32 }
          cleanup
  br label %innercleanup

outerlp:
  %3 = landingpad { i8*, i32 }
          cleanup
  br label %outercleanup

outercleanup:
  %4 = phi { i8*, i32 } [%2, %innercleanup], [%3, %outerlp]
  call void @clean()
  resume { i8*, i32 } %4

innercleanup:
  call void @clean()
  br label %outercleanup
}

declare void @hot_function()

define void @test_hot_branch(i32* %a) {
; Test that a hot branch that has a probability a little larger than 80% will
; break CFG constrains when doing block placement.
; CHECK-LABEL: test_hot_branch:
; CHECK: %entry
; CHECK: %then
; CHECK: %exit
; CHECK: %else

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then, label %else, !prof !5

then:
  call void @hot_function()
  br label %exit

else:
  call void @cold_function()
  br label %exit

exit:
  call void @hot_function()
  ret void
}

define void @test_hot_branch_profile(i32* %a) !prof !6 {
; Test that a hot branch that has a probability a little larger than 50% will
; break CFG constrains when doing block placement when profile is available.
; CHECK-LABEL: test_hot_branch_profile:
; CHECK: %entry
; CHECK: %then
; CHECK: %exit
; CHECK: %else

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then, label %else, !prof !7

then:
  call void @hot_function()
  br label %exit

else:
  call void @cold_function()
  br label %exit

exit:
  call void @hot_function()
  ret void
}

define void @test_hot_branch_triangle_profile(i32* %a) !prof !6 {
; Test that a hot branch that has a probability a little larger than 80% will
; break triangle shaped CFG constrains when doing block placement if profile
; is present.
; CHECK-LABEL: test_hot_branch_triangle_profile:
; CHECK: %entry
; CHECK: %exit
; CHECK: %then

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %exit, label %then, !prof !5

then:
  call void @hot_function()
  br label %exit

exit:
  call void @hot_function()
  ret void
}

define void @test_hot_branch_triangle_profile_topology(i32* %a) !prof !6 {
; Test that a hot branch that has a probability between 50% and 66% will not
; break triangle shaped CFG constrains when doing block placement if profile
; is present.
; CHECK-LABEL: test_hot_branch_triangle_profile_topology:
; CHECK: %entry
; CHECK: %then
; CHECK: %exit

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %exit, label %then, !prof !7

then:
  call void @hot_function()
  br label %exit

exit:
  call void @hot_function()
  ret void
}

declare void @a()
declare void @b()

define void @test_forked_hot_diamond(i32* %a) {
; Test that a hot-branch with probability > 80% followed by a 50/50 branch
; will not place the cold predecessor if the probability for the fallthrough
; remains above 80%
; CHECK-LABEL: test_forked_hot_diamond
; CHECK: %entry
; CHECK: %then
; CHECK: %fork1
; CHECK: %else
; CHECK: %fork2
; CHECK: %exit
entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then, label %else, !prof !5

then:
  call void @hot_function()
  %gep2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %gep2
  %cond2 = icmp ugt i32 %val2, 2
  br i1 %cond2, label %fork1, label %fork2, !prof !8

else:
  call void @cold_function()
  %gep3 = getelementptr i32, i32* %a, i32 3
  %val3 = load i32, i32* %gep3
  %cond3 = icmp ugt i32 %val3, 3
  br i1 %cond3, label %fork1, label %fork2, !prof !8

fork1:
  call void @a()
  br label %exit

fork2:
  call void @b()
  br label %exit

exit:
  call void @hot_function()
  ret void
}

define void @test_forked_hot_diamond_gets_cold(i32* %a) {
; Test that a hot-branch with probability > 80% followed by a 50/50 branch
; will place the cold predecessor if the probability for the fallthrough
; falls below 80%
; The probability for both branches is 85%. For then2 vs else1
; this results in a compounded probability of 83%.
; Neither then2->fork1 nor then2->fork2 has a large enough relative
; probability to break the CFG.
; Relative probs:
; then2 -> fork1 vs else1 -> fork1 = 71%
; then2 -> fork2 vs else2 -> fork2 = 74%
; CHECK-LABEL: test_forked_hot_diamond_gets_cold
; CHECK: %entry
; CHECK: %then1
; CHECK: %then2
; CHECK: %else1
; CHECK: %fork1
; CHECK: %else2
; CHECK: %fork2
; CHECK: %exit
entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then1, label %else1, !prof !9

then1:
  call void @hot_function()
  %gep2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %gep2
  %cond2 = icmp ugt i32 %val2, 2
  br i1 %cond2, label %then2, label %else2, !prof !9

else1:
  call void @cold_function()
  br label %fork1

then2:
  call void @hot_function()
  %gep3 = getelementptr i32, i32* %a, i32 3
  %val3 = load i32, i32* %gep2
  %cond3 = icmp ugt i32 %val2, 3
  br i1 %cond3, label %fork1, label %fork2, !prof !8

else2:
  call void @cold_function()
  br label %fork2

fork1:
  call void @a()
  br label %exit

fork2:
  call void @b()
  br label %exit

exit:
  call void @hot_function()
  ret void
}

define void @test_forked_hot_diamond_stays_hot(i32* %a) {
; Test that a hot-branch with probability > 88.88% (1:8) followed by a 50/50
; branch will not place the cold predecessor as the probability for the
; fallthrough stays above 80%
; (1:8) followed by (1:1) is still (1:4)
; Here we use 90% probability because two in a row
; have a 89 % probability vs the original branch.
; CHECK-LABEL: test_forked_hot_diamond_stays_hot
; CHECK: %entry
; CHECK: %then1
; CHECK: %then2
; CHECK: %fork1
; CHECK: %else1
; CHECK: %else2
; CHECK: %fork2
; CHECK: %exit
entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then1, label %else1, !prof !10

then1:
  call void @hot_function()
  %gep2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %gep2
  %cond2 = icmp ugt i32 %val2, 2
  br i1 %cond2, label %then2, label %else2, !prof !10

else1:
  call void @cold_function()
  br label %fork1

then2:
  call void @hot_function()
  %gep3 = getelementptr i32, i32* %a, i32 3
  %val3 = load i32, i32* %gep2
  %cond3 = icmp ugt i32 %val2, 3
  br i1 %cond3, label %fork1, label %fork2, !prof !8

else2:
  call void @cold_function()
  br label %fork2

fork1:
  call void @a()
  br label %exit

fork2:
  call void @b()
  br label %exit

exit:
  call void @hot_function()
  ret void
}

; Because %endif has a higher frequency than %if, the calculations show we
; shouldn't tail-duplicate %endif so that we can place it after %if. We were
; previously undercounting the cost by ignoring execution frequency that didn't
; come from the %if->%endif path.
; CHECK-LABEL: higher_frequency_succ_tail_dup
; CHECK: %entry
; CHECK: %elseif
; CHECK: %else
; CHECK: %endif
; CHECK: %then
; CHECK: %ret
define void @higher_frequency_succ_tail_dup(i1 %a, i1 %b, i1 %c) {
entry:
  br label %if
if:                                               ; preds = %entry
  call void @effect(i32 0)
  br i1 %a, label %elseif, label %endif, !prof !11 ; even

elseif:                                           ; preds = %if
  call void @effect(i32 1)
  br i1 %b, label %else, label %endif, !prof !11 ; even

else:                                             ; preds = %elseif
  call void @effect(i32 2)
  br label %endif

endif:                                            ; preds = %if, %elseif, %else
  br i1 %c, label %then, label %ret, !prof !12 ; 5 to 3

then:                                             ; preds = %endif
  call void @effect(i32 3)
  br label %ret

ret:                                              ; preds = %endif, %then
  ret void
}

define i32 @not_rotate_if_extra_branch(i32 %count) {
; Test checks that there is no loop rotation
; if it introduces extra branch.
; Specifically in this case because best exit is .header
; but it has fallthrough to .middle block and last block in
; loop chain .slow does not have afallthrough to .header.
; CHECK-LABEL: not_rotate_if_extra_branch
; CHECK: %.entry
; CHECK: %.header
; CHECK: %.middle
; CHECK: %.backedge
; CHECK: %.slow
; CHECK: %.bailout
; CHECK: %.stop
.entry:
  %sum.0 = shl nsw i32 %count, 1
  br label %.header

.header:
  %i = phi i32 [ %i.1, %.backedge ], [ 0, %.entry ]
  %sum = phi i32 [ %sum.1, %.backedge ], [ %sum.0, %.entry ]
  %is_exc = icmp sgt i32 %i, 9000000
  br i1 %is_exc, label %.bailout, label %.middle, !prof !13

.bailout:
  %sum.2 = add nsw i32 %count, 1
  br label %.stop

.middle:
  %pr.1 = and i32 %i, 1023
  %pr.2 = icmp eq i32 %pr.1, 0
  br i1 %pr.2, label %.slow, label %.backedge, !prof !14

.slow:
  tail call void @effect(i32 %sum)
  br label %.backedge

.backedge:
  %sum.1 = add nsw i32 %i, %sum
  %i.1 = add nsw i32 %i, 1
  %end = icmp slt i32 %i.1, %count
  br i1 %end, label %.header, label %.stop, !prof !15

.stop:
  %sum.phi = phi i32 [ %sum.1, %.backedge ], [ %sum.2, %.bailout ]
  ret i32 %sum.phi
}

define i32 @not_rotate_if_extra_branch_regression(i32 %count, i32 %init) {
; This is a regression test against patch avoid loop rotation if
; it introduce an extra btanch.
; CHECK-LABEL: not_rotate_if_extra_branch_regression
; CHECK: %.entry
; CHECK: %.first_backedge
; CHECK: %.second_header
; CHECK: %.slow
.entry:
  %sum.0 = shl nsw i32 %count, 1
  br label %.first_header

.first_header:
  %i = phi i32 [ %i.1, %.first_backedge ], [ 0, %.entry ]
  %is_bo1 = icmp sgt i32 %i, 9000000
  br i1 %is_bo1, label %.bailout, label %.first_backedge, !prof !14

.first_backedge:
  %i.1 = add nsw i32 %i, 1
  %end = icmp slt i32 %i.1, %count
  br i1 %end, label %.first_header, label %.second_header, !prof !13

.second_header:
  %j = phi i32 [ %j.1, %.second_backedge ], [ %init, %.first_backedge ]
  %end.2 = icmp sgt i32 %j, %count
  br i1 %end.2, label %.stop, label %.second_middle, !prof !14

.second_middle:
  %is_slow = icmp sgt i32 %j, 9000000
  br i1 %is_slow, label %.slow, label %.second_backedge, !prof !14

.slow:
  tail call void @effect(i32 %j)
  br label %.second_backedge

.second_backedge:
  %j.1 = add nsw i32 %j, 1
  %end.3 = icmp slt i32 %j, 10000000
  br i1 %end.3, label %.second_header, label %.stop, !prof !13

.stop:
  %res = add nsw i32 %j, %i.1
  ret i32 %res

.bailout:
  ret i32 0
}

declare void @effect(i32)

!5 = !{!"branch_weights", i32 84, i32 16}
!6 = !{!"function_entry_count", i32 10}
!7 = !{!"branch_weights", i32 60, i32 40}
!8 = !{!"branch_weights", i32 5001, i32 4999}
!9 = !{!"branch_weights", i32 85, i32 15}
!10 = !{!"branch_weights", i32 90, i32 10}
!11 = !{!"branch_weights", i32 1, i32 1}
!12 = !{!"branch_weights", i32 5, i32 3}
!13 = !{!"branch_weights", i32 1, i32 1}
!14 = !{!"branch_weights", i32 1, i32 1023}
!15 = !{!"branch_weights", i32 4095, i32 1}
