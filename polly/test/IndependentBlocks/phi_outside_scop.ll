; RUN: opt %loadPolly -basicaa -polly-independent -S < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-independent -polly-codegen-scev -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @phi_nodes_outside() {
entry:
  br label %for.i.1

for.i.1:
  %i.1 = phi i32 [ %i.1.next, %for.i.1 ], [ 0, %entry ]
  %i.1.next = add nsw i32 %i.1, 1
  br i1 false, label %for.i.1 , label %for.i.2.preheader

for.i.2.preheader:
  br label %for.i.2

for.i.2:
; The value of %i.1.next is used outside of the scop in a PHI node.
  %i.2 = phi i32 [ %i.2.next , %for.i.2 ], [ %i.1.next, %for.i.2.preheader ]
  %i.2.next = add nsw i32 %i.2, 1
  fence seq_cst
  br i1 false, label %for.i.2, label %cleanup

cleanup:
  ret void
}

; CHECK:  store i32 %i.1.next, i32* %i.1.next.s2a

; CHECK: for.i.2.preheader:
; CHECK:    %i.1.next.loadoutside = load i32* %i.1.next.s2a

; CHECK: for.i.2:
; CHECK:    %i.2 = phi i32 [ %i.2.next, %for.i.2 ], [ %i.1.next.loadoutside, %for.i.2.preheader ]

