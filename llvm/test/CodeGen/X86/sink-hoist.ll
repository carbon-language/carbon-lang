; RUN: llc < %s -march=x86-64 -asm-verbose=false -mtriple=x86_64-unknown-linux-gnu -post-RA-scheduler=true | FileCheck %s

; Currently, floating-point selects are lowered to CFG triangles.
; This means that one side of the select is always unconditionally
; evaluated, however with MachineSink we can sink the other side so
; that it's conditionally evaluated.

; CHECK: foo:
; CHECK-NEXT: testb $1, %dil
; CHECK-NEXT: je
; CHECK-NEXT: divsd
; CHECK-NEXT: ret
; CHECK:      divsd

define double @foo(double %x, double %y, i1 %c) nounwind {
  %a = fdiv double %x, 3.2
  %b = fdiv double %y, 3.3
  %z = select i1 %c, double %a, double %b
  ret double %z
}

; Make sure the critical edge is broken so the divsd is sunken below
; the conditional branch.
; rdar://8454886

; CHECK: split:
; CHECK-NEXT: testb $1, %dil
; CHECK-NEXT: je
; CHECK-NEXT: divsd
; CHECK-NEXT: ret
; CHECK:      movaps
; CHECK-NEXT: ret
define double @split(double %x, double %y, i1 %c) nounwind {
  %a = fdiv double %x, 3.2
  %z = select i1 %c, double %a, double %y
  ret double %z
}


; Hoist floating-point constant-pool loads out of loops.

; CHECK: bar:
; CHECK: movsd
; CHECK: align
define void @bar(double* nocapture %p, i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 0
  br i1 %0, label %bb, label %return

bb:
  %i.03 = phi i64 [ 0, %entry ], [ %3, %bb ]
  %scevgep = getelementptr double* %p, i64 %i.03
  %1 = load double* %scevgep, align 8
  %2 = fdiv double 3.200000e+00, %1
  store double %2, double* %scevgep, align 8
  %3 = add nsw i64 %i.03, 1
  %exitcond = icmp eq i64 %3, %n
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Sink instructions with dead EFLAGS defs.

; FIXME: Unfail the zzz test if we can correctly mark pregs with the kill flag.
; 
; See <rdar://problem/8030636>. This test isn't valid after we made machine
; sinking more conservative about sinking instructions that define a preg into a
; block when we don't know if the preg is killed within the current block.


; FIXMEHECK: zzz:
; FIXMEHECK:      je
; FIXMEHECK-NEXT: orb

; define zeroext i8 @zzz(i8 zeroext %a, i8 zeroext %b) nounwind readnone {
; entry:
;   %tmp = zext i8 %a to i32                        ; <i32> [#uses=1]
;   %tmp2 = icmp eq i8 %a, 0                    ; <i1> [#uses=1]
;   %tmp3 = or i8 %b, -128                          ; <i8> [#uses=1]
;   %tmp4 = and i8 %b, 127                          ; <i8> [#uses=1]
;   %b_addr.0 = select i1 %tmp2, i8 %tmp4, i8 %tmp3 ; <i8> [#uses=1]
;   ret i8 %b_addr.0
; }

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone

; CodeGen should use the correct register class when extracting
; a load from a zero-extending load for hoisting.

; CHECK: default_get_pch_validity:
; CHECK: movl cl_options_count(%rip), %ecx

@cl_options_count = external constant i32         ; <i32*> [#uses=2]

define void @default_get_pch_validity() nounwind {
entry:
  %tmp4 = load i32* @cl_options_count, align 4    ; <i32> [#uses=1]
  %tmp5 = icmp eq i32 %tmp4, 0                    ; <i1> [#uses=1]
  br i1 %tmp5, label %bb6, label %bb2

bb2:                                              ; preds = %bb2, %entry
  %i.019 = phi i64 [ 0, %entry ], [ %tmp25, %bb2 ] ; <i64> [#uses=1]
  %tmp25 = add i64 %i.019, 1                      ; <i64> [#uses=2]
  %tmp11 = load i32* @cl_options_count, align 4   ; <i32> [#uses=1]
  %tmp12 = zext i32 %tmp11 to i64                 ; <i64> [#uses=1]
  %tmp13 = icmp ugt i64 %tmp12, %tmp25            ; <i1> [#uses=1]
  br i1 %tmp13, label %bb2, label %bb6

bb6:                                              ; preds = %bb2, %entry
  ret void
}
