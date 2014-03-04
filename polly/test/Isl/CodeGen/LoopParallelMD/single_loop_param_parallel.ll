; RUN: opt %loadPolly -polly-codegen-isl -S < %s | FileCheck %s -check-prefix=SEQUENTIAL
; RUN: opt %loadPolly -polly-codegen-isl -S -polly-codegen-scev < %s | FileCheck %s -check-prefix=SEQUENTIAL-SCEV
; RUN: opt %loadPolly -polly-codegen-isl -polly-ast-detect-parallel -S < %s | FileCheck %s -check-prefix=PARALLEL
; RUN: opt %loadPolly -polly-codegen-isl -polly-ast-detect-parallel -S -polly-codegen-scev < %s | FileCheck %s -check-prefix=PARALLEL-SCEV
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

; This is a trivially parallel loop. We just use it to ensure that we actually
; emit the right information.
;
; for (i = 0; i < n; i++)
;   A[i] = 1;
;
@A = common global [1024 x i32] zeroinitializer
define void @test-one(i64 %n) {
start:
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %exitcond = icmp ne i64 %i, %n
  br i1 %exitcond, label %loop.body, label %ret

loop.body:
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %i
  store i32 1, i32* %scevgep
  br label %loop.backedge

loop.backedge:
  %i.next = add nsw i64 %i, 1
  br label %loop.header

ret:
  fence seq_cst
  ret void
}

; SEQUENTIAL: @test-one
; SEQUENTIAL-NOT: !llvm.mem.parallel_loop_access
; SEQUENTIAL-NOT: !llvm.loop !0
; SEQUENTIAL-SCEV: @test-one
; SEQUENTIAL-SCEV-NOT: !llvm.mem.parallel_loop_access
; SEQUENTIAL-SCEV-NOT: !llvm.loop

; PARALLEL: @test-one
; PARALLEL: store i32 1, i32* %p_scevgep, !llvm.mem.parallel_loop_access !0
; PARALLEL:  br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit, !llvm.loop !0
; PARALLEL-SCEV: @test-one
; PARALLEL-SCEV: store i32 1, i32* %scevgep1, !llvm.mem.parallel_loop_access !0
; PARALLEL-SCEV:  br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit, !llvm.loop !0

; This loop has memory dependences that require at least a simple dependence
; analysis to detect the parallelism.
;
; for (i = 0; i < n; i++)
;   A[2 * i] = A[2 * i + 1];
;
define void @test-two(i64 %n) {
start:
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %exitcond = icmp ne i64 %i, %n
  br i1 %exitcond, label %loop.body, label %ret

loop.body:
  %loadoffset1 = mul nsw i64 %i, 2
  %loadoffset2 = add nsw i64 %loadoffset1, 1
  %scevgepload = getelementptr [1024 x i32]* @A, i64 0, i64 %loadoffset2
  %val = load i32* %scevgepload
  %storeoffset = mul i64 %i, 2
  %scevgepstore = getelementptr [1024 x i32]* @A, i64 0, i64 %storeoffset
  store i32 %val, i32* %scevgepstore
  br label %loop.backedge

loop.backedge:
  %i.next = add nsw i64 %i, 1
  br label %loop.header

ret:
  fence seq_cst
  ret void
}

; SEQUENTIAL: @test-two
; SEQUENTIAL-NOT: !llvm.mem.parallel_loop_access
; SEQUENTIAL-NOT: !llvm.loop !0
; SEQUENTIAL-SCEV: @test-two
; SEQUENTIAL-SCEV-NOT: !llvm.mem.parallel_loop_access
; SEQUENTIAL-SCEV-NOT: !llvm.loop

; PARALLEL: @test-two
; PARALLEL: %val_p_scalar_ = load i32* %p_scevgepload, !llvm.mem.parallel_loop_access !1
; PARALLEL: store i32 %val_p_scalar_, i32* %p_scevgepstore, !llvm.mem.parallel_loop_access !1
; PARALLEL: br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit, !llvm.loop !1
; PARALLEL-SCEV: @test-two
; PARALLEL-SCEV: %val_p_scalar_ = load i32* %scevgep, !llvm.mem.parallel_loop_access !1
; PARALLEL-SCEV: store i32 %val_p_scalar_, i32* %scevgep1, !llvm.mem.parallel_loop_access !1
; PARALLEL-SCEV:  br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit, !llvm.loop !1
