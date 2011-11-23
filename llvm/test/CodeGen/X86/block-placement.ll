; RUN: llc -march=x86 -enable-block-placement < %s | FileCheck %s

declare void @error(i32 %i, i32 %a, i32 %b)

define i32 @test_ifchains(i32 %i, i32* %a, i32 %b) {
; Test a chain of ifs, where the block guarded by the if is error handling code
; that is not expected to run.
; CHECK: test_ifchains:
; CHECK: %entry
; CHECK: %else1
; CHECK: %else2
; CHECK: %else3
; CHECK: %else4
; CHECK: %exit
; CHECK: %then1
; CHECK: %then2
; CHECK: %then3
; CHECK: %then4
; CHECK: %then5

entry:
  %gep1 = getelementptr i32* %a, i32 1
  %val1 = load i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then1, label %else1, !prof !0

then1:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else1

else1:
  %gep2 = getelementptr i32* %a, i32 2
  %val2 = load i32* %gep2
  %cond2 = icmp ugt i32 %val2, 2
  br i1 %cond2, label %then2, label %else2, !prof !0

then2:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else2

else2:
  %gep3 = getelementptr i32* %a, i32 3
  %val3 = load i32* %gep3
  %cond3 = icmp ugt i32 %val3, 3
  br i1 %cond3, label %then3, label %else3, !prof !0

then3:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else3

else3:
  %gep4 = getelementptr i32* %a, i32 4
  %val4 = load i32* %gep4
  %cond4 = icmp ugt i32 %val4, 4
  br i1 %cond4, label %then4, label %else4, !prof !0

then4:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else4

else4:
  %gep5 = getelementptr i32* %a, i32 3
  %val5 = load i32* %gep5
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
; CHECK: test_loop_cold_blocks:
; CHECK: %entry
; CHECK: %body1
; CHECK: %body2
; CHECK: %body3
; CHECK: %unlikely1
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
  %arrayidx = getelementptr inbounds i32* %a, i32 %iv
  %0 = load i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body1

exit:
  ret i32 %sum
}

!0 = metadata !{metadata !"branch_weights", i32 4, i32 64}

define i32 @test_loop_early_exits(i32 %i, i32* %a) {
; Check that we sink early exit blocks out of loop bodies.
; CHECK: test_loop_early_exits:
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
  %arrayidx = getelementptr inbounds i32* %a, i32 %iv
  %0 = load i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body1

exit:
  ret i32 %sum
}

define i32 @test_loop_align(i32 %i, i32* %a) {
; Check that we provide basic loop body alignment with the block placement
; pass.
; CHECK: test_loop_align:
; CHECK: %entry
; CHECK: .align [[ALIGN:[0-9]+]],
; CHECK-NEXT: %body
; CHECK: %exit

entry:
  br label %body

body:
  %iv = phi i32 [ 0, %entry ], [ %next, %body ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body ]
  %arrayidx = getelementptr inbounds i32* %a, i32 %iv
  %0 = load i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body

exit:
  ret i32 %sum
}

define i32 @test_nested_loop_align(i32 %i, i32* %a, i32* %b) {
; Check that we provide nested loop body alignment.
; CHECK: test_nested_loop_align:
; CHECK: %entry
; CHECK: .align [[ALIGN]],
; CHECK-NEXT: %loop.body.1
; CHECK: .align [[ALIGN]],
; CHECK-NEXT: %inner.loop.body
; CHECK-NOT: .align
; CHECK: %exit

entry:
  br label %loop.body.1

loop.body.1:
  %iv = phi i32 [ 0, %entry ], [ %next, %loop.body.2 ]
  %arrayidx = getelementptr inbounds i32* %a, i32 %iv
  %bidx = load i32* %arrayidx
  br label %inner.loop.body

inner.loop.body:
  %inner.iv = phi i32 [ 0, %loop.body.1 ], [ %inner.next, %inner.loop.body ]
  %base = phi i32 [ 0, %loop.body.1 ], [ %sum, %inner.loop.body ]
  %scaled_idx = mul i32 %bidx, %iv
  %inner.arrayidx = getelementptr inbounds i32* %b, i32 %scaled_idx
  %0 = load i32* %inner.arrayidx
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
; CHECK: unnatural_cfg1
; CHECK: %entry
; CHECK: %loop.body1
; CHECK: %loop.body2
; CHECK: %loop.body3

entry:
  br label %loop.header

loop.header:
  br label %loop.body1

loop.body1:
  br i1 undef, label %loop.body3, label %loop.body2

loop.body2:
  %ptr = load i32** undef, align 4
  br label %loop.body3

loop.body3:
  %myptr = phi i32* [ %ptr2, %loop.body5 ], [ %ptr, %loop.body2 ], [ undef, %loop.body1 ]
  %bcmyptr = bitcast i32* %myptr to i32*
  %val = load i32* %bcmyptr, align 4
  %comp = icmp eq i32 %val, 48
  br i1 %comp, label %loop.body4, label %loop.body5

loop.body4:
  br i1 undef, label %loop.header, label %loop.body5

loop.body5:
  %ptr2 = load i32** undef, align 4
  br label %loop.body3
}

define void @unnatural_cfg2() {
; Test that we can handle a loop with a nested natural loop *and* an unnatural
; loop. This was reduced from a crash on block placement when run over
; single-source GCC.
; CHECK: unnatural_cfg2
; CHECK: %entry
; CHECK: %loop.header
; CHECK: %loop.body1
; CHECK: %loop.body2
; CHECK: %loop.body3
; CHECK: %loop.inner1.begin
; The end block is folded with %loop.body3...
; CHECK-NOT: %loop.inner1.end
; CHECK: %loop.body4
; CHECK: %loop.inner2.begin
; The loop.inner2.end block is folded
; CHECK: %bail

entry:
  br label %loop.header

loop.header:
  %comp0 = icmp eq i32* undef, null
  br i1 %comp0, label %bail, label %loop.body1

loop.body1:
  %val0 = load i32** undef, align 4
  br i1 undef, label %loop.body2, label %loop.inner1.begin

loop.body2:
  br i1 undef, label %loop.body4, label %loop.body3

loop.body3:
  %ptr1 = getelementptr inbounds i32* %val0, i32 0
  %castptr1 = bitcast i32* %ptr1 to i32**
  %val1 = load i32** %castptr1, align 4
  br label %loop.inner1.begin

loop.inner1.begin:
  %valphi = phi i32* [ %val2, %loop.inner1.end ], [ %val1, %loop.body3 ], [ %val0, %loop.body1 ]
  %castval = bitcast i32* %valphi to i32*
  %comp1 = icmp eq i32 undef, 48
  br i1 %comp1, label %loop.inner1.end, label %loop.body4

loop.inner1.end:
  %ptr2 = getelementptr inbounds i32* %valphi, i32 0
  %castptr2 = bitcast i32* %ptr2 to i32**
  %val2 = load i32** %castptr2, align 4
  br label %loop.inner1.begin

loop.body4.dead:
  br label %loop.body4

loop.body4:
  %comp2 = icmp ult i32 undef, 3
  br i1 %comp2, label %loop.inner2.begin, label %loop.end

loop.inner2.begin:
  br i1 false, label %loop.end, label %loop.inner2.end

loop.inner2.end:
  %comp3 = icmp eq i32 undef, 1769472
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

define void @fpcmp_unanalyzable_branch(i1 %cond) {
; This function's CFG contains an unanalyzable branch that is likely to be
; split due to having a different high-probability predecessor.
; CHECK: fpcmp_unanalyzable_branch
; CHECK: %entry
; CHECK: %exit
; CHECK-NOT: %if.then
; CHECK-NOT: %if.end
; CHECK-NOT: jne
; CHECK-NOT: jnp
; CHECK: jne
; CHECK-NEXT: jnp
; CHECK-NEXT: %if.then

entry:
; Note that this branch must be strongly biased toward
; 'entry.if.then_crit_edge' to ensure that we would try to form a chain for
; 'entry' -> 'entry.if.then_crit_edge' -> 'if.then'. It is the last edge in that
; chain which would violate the unanalyzable branch in 'exit', but we won't even
; try this trick unless 'if.then' is believed to almost always be reached from
; 'entry.if.then_crit_edge'.
  br i1 %cond, label %entry.if.then_crit_edge, label %lor.lhs.false, !prof !1

entry.if.then_crit_edge:
  %.pre14 = load i8* undef, align 1, !tbaa !0
  br label %if.then

lor.lhs.false:
  br i1 undef, label %if.end, label %exit

exit:
  %cmp.i = fcmp une double 0.000000e+00, undef
  br i1 %cmp.i, label %if.then, label %if.end

if.then:
  %0 = phi i8 [ %.pre14, %entry.if.then_crit_edge ], [ undef, %exit ]
  %1 = and i8 %0, 1
  store i8 %1, i8* undef, align 4, !tbaa !0
  br label %if.end

if.end:
  ret void
}

!1 = metadata !{metadata !"branch_weights", i32 1000, i32 1}

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

!2 = metadata !{metadata !"branch_weights", i32 3, i32 1}

declare i32 @__gxx_personality_v0(...)

define void @test_eh_lpad_successor() {
; Some times the landing pad ends up as the first successor of an invoke block.
; When this happens, a strange result used to fall out of updateTerminators: we
; didn't correctly locate the fallthrough successor, assuming blindly that the
; first one was the fallthrough successor. As a result, we would add an
; erroneous jump to the landing pad thinking *that* was the default successor.
; CHECK: test_eh_lpad_successor
; CHECK: %entry
; CHECK-NOT: jmp
; CHECK: %loop

entry:
  invoke i32 @f() to label %preheader unwind label %lpad

preheader:
  br label %loop

lpad:
  %lpad.val = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  resume { i8*, i32 } %lpad.val

loop:
  br label %loop
}

declare void @fake_throw() noreturn

define void @test_eh_throw() {
; For blocks containing a 'throw' (or similar functionality), we have
; a no-return invoke. In this case, only EH successors will exist, and
; fallthrough simply won't occur. Make sure we don't crash trying to update
; terminators for such constructs.
;
; CHECK: test_eh_throw
; CHECK: %entry
; CHECK: %cleanup

entry:
  invoke void @fake_throw() to label %continue unwind label %cleanup

continue:
  unreachable

cleanup:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  unreachable
}

define void @test_unnatural_cfg_backwards_inner_loop() {
; Test that when we encounter an unnatural CFG structure after having formed
; a chain for an inner loop which happened to be laid out backwards we don't
; attempt to merge onto the wrong end of the inner loop just because we find it
; first. This was reduced from a crasher in GCC's single source.
;
; CHECK: test_unnatural_cfg_backwards_inner_loop
; CHECK: %entry
; CHECK: %body
; CHECK: %loop1
; CHECK: %loop2b
; CHECK: %loop2a

entry:
  br i1 undef, label %loop2a, label %body

body:
  br label %loop2a

loop1:
  %next.load = load i32** undef
  br i1 %comp.a, label %loop2a, label %loop2b

loop2a:
  %var = phi i32* [ null, %entry ], [ null, %body ], [ %next.phi, %loop1 ]
  %next.var = phi i32* [ null, %entry ], [ undef, %body ], [ %next.load, %loop1 ]
  %comp.a = icmp eq i32* %var, null
  br label %loop3

loop2b:
  %gep = getelementptr inbounds i32* %var.phi, i32 0
  %next.ptr = bitcast i32* %gep to i32**
  store i32* %next.phi, i32** %next.ptr
  br label %loop3

loop3:
  %var.phi = phi i32* [ %next.phi, %loop2b ], [ %var, %loop2a ]
  %next.phi = phi i32* [ %next.load, %loop2b ], [ %next.var, %loop2a ]
  br label %loop1
}
