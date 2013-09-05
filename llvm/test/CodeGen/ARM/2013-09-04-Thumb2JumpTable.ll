; RUN: llc < %s -march=thumb -mattr=+thumb2 |  FileCheck %s

define i32 @foo(i32 %n, i32* nocapture %inp) #0 {
; CHECK: foo
; CHECK-NOT: mov pc, lr
.split:
  %0 = icmp sgt i32 %n, 1
  %1 = add nsw i32 %n, -1
  %loop_guard = icmp sgt i32 %1, 0
  %or.cond = and i1 %0, %loop_guard
  br i1 %or.cond, label %stmt.preheader, label %loop_exit

stmt.preheader:                            ; preds = %.split
  %adjust_ub = add i32 %n, -2
  %scevgep6.gep = getelementptr i32* %inp, i32 1
  %2 = icmp sgt i32 %adjust_ub, 0
  %adjust_ub.op = add i32 %n, -1
  %3 = select i1 %2, i32 %adjust_ub.op, i32 1
  %xtraiter = and i32 %3, 3
  switch i32 %xtraiter, label %stmt.unr [
    i32 0, label %stmt.
    i32 1, label %stmt.unr30
    i32 2, label %stmt.unr16
    i32 3, label %stmt.unr8
  ]

stmt.unr:                                  ; preds = %stmt.preheader
  %scevgep6.inc.unr = getelementptr i32* %inp, i32 2
  br label %stmt.unr8

stmt.unr8:                                 ; preds = %stmt.preheader, %stmt.unr
  %imax.03.reg2mem.0.unr = phi i32 [ 1, %stmt.unr ], [ 0, %stmt.preheader ]
  %scevgep6.phi.unr = phi i32* [ %scevgep6.inc.unr, %stmt.unr ], [ %scevgep6.gep, %stmt.preheader ]
  %scevgep6.inc.unr15 = getelementptr i32* %scevgep6.phi.unr, i32 1
  br label %stmt.unr16

stmt.unr16:                                ; preds = %stmt.preheader, %stmt.unr8
  %imax.03.reg2mem.0.unr17 = phi i32 [ 0, %stmt.unr8 ], [ 0, %stmt.preheader ]
  %selv.lcssa.reg2mem.1.unr18 = phi i32 [ 0, %stmt.unr8 ], [ undef, %stmt.preheader ]
  %scevgep6.phi.unr19 = phi i32* [ %scevgep6.inc.unr15, %stmt.unr8 ], [ %scevgep6.gep, %stmt.preheader ]
  %indvar.unr20 = phi i32 [ 1, %stmt.unr8 ], [ 0, %stmt.preheader ]
  %scevgep6.inc.unr27 = getelementptr i32* %scevgep6.phi.unr19, i32 1
  br label %stmt.unr30

stmt.unr30:                                ; preds = %stmt.preheader, %stmt.unr16
  %imax.03.reg2mem.0.unr31 = phi i32 [ 1, %stmt.unr16 ], [ 0, %stmt.preheader ]
  %selv.lcssa.reg2mem.1.unr32 = phi i32 [ 0, %stmt.unr16 ], [ undef, %stmt.preheader ]
  %scevgep6.phi.unr33 = phi i32* [ %scevgep6.inc.unr27, %stmt.unr16 ], [ %scevgep6.gep, %stmt.preheader ]
  %indvar.unr34 = phi i32 [ 0, %stmt.unr16 ], [ 1, %stmt.preheader ]
  %_p_scalar_.unr36 = load i32* %scevgep6.phi.unr33, align 4
  %p_.unr37 = icmp sgt i32 %_p_scalar_.unr36, %imax.03.reg2mem.0.unr31
  %scevgep6.inc.unr41 = getelementptr i32* %scevgep6.phi.unr33, i32 1
  %4 = icmp ugt i32 %3, 4
  br i1 %4, label %stmt., label %loop_exit


loop_exit:                                  ; preds = %stmt.unr30, %stmt., %.split
  %itemp.0.lcssa.reg2mem.0 = phi i32 [ undef, %.split ], [ 1, %stmt.unr30 ], [0, %stmt. ]
  ret i32 %itemp.0.lcssa.reg2mem.0

stmt.:                                      ; preds = %stmt.preheader, %stmt.unr30, %stmt.
  %imax.03.reg2mem.0 = phi i32 [ %p_selv2.3, %stmt. ], [ 1, %stmt.unr30 ], [ 0, %stmt.preheader ]
  %selv.lcssa.reg2mem.1 = phi i32 [ 0, %stmt. ], [ 1, %stmt.unr30 ], [ undef, %stmt.preheader ]
  %scevgep6.phi = phi i32* [ %scevgep6.inc.3, %stmt. ], [ %scevgep6.inc.unr41, %stmt.unr30 ], [ %scevgep6.gep, %stmt.preheader ]
  %indvar = phi i32 [ %scevgep.sum.3, %stmt. ], [ 1, %stmt.unr30 ], [ 0, %stmt.preheader ]
  %scevgep.sum = add i32 %indvar, 1
  %_p_scalar_ = load i32* %scevgep6.phi, align 4
  %p_ = icmp sgt i32 %_p_scalar_, %imax.03.reg2mem.0
  %p_selv = select i1 %p_, i32 %scevgep.sum, i32 %selv.lcssa.reg2mem.1
  %scevgep.sum.3 = add i32 %indvar, 4
  %p_selv2.3 = select i1 %p_, i32 %_p_scalar_, i32 %p_selv
  %scevgep6.inc.3 = getelementptr i32* %scevgep6.phi, i32 4
  %loop_cond.4 = icmp slt i32 %scevgep.sum.3, %adjust_ub
  br i1 %loop_cond.4, label %stmt., label %loop_exit
}

