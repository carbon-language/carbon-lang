; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl  | FileCheck %s

; ModuleID = 'foo.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.cttz.i64(i64, i1) #0

; Function Attrs: nounwind
define void @foo(float* noalias %aFOO, float %b, i32 %a) {
allocas:
  %full_mask_memory.i57 = alloca <8 x float>
  %return_value_memory.i60 = alloca i1
  %cmp.i = icmp eq i32 %a, 65535
  br i1 %cmp.i, label %all_on, label %some_on

all_on:
  %mask0 = load <8 x float>, <8 x float>* %full_mask_memory.i57
  %v0.i.i.i70 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask0) #0
  %allon.i.i76 = icmp eq i32 %v0.i.i.i70, 65535
  br i1 %allon.i.i76, label %check_neighbors.i.i121, label %domixed.i.i100

domixed.i.i100: 
  br label %check_neighbors.i.i121

check_neighbors.i.i121: 
  %v1.i5.i.i116 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask0) #0
  %alleq.i.i120 = icmp eq i32 %v1.i5.i.i116, 65535
  br i1 %alleq.i.i120, label %all_equal.i.i123, label %not_all_equal.i.i124

; CHECK: kxnorw  %k0, %k0, %k0
; CHECK: kshiftrw        $15, %k0, %k0
; CHECK: jmp
; CHECK: kxorw   %k0, %k0, %k0

all_equal.i.i123:
  br label %reduce_equal___vyi.exit128

not_all_equal.i.i124:        
  br label %reduce_equal___vyi.exit128

reduce_equal___vyi.exit128:
  %calltmp2.i125 = phi i1 [ true, %all_equal.i.i123 ], [ false, %not_all_equal.i.i124 ]
  store i1 %calltmp2.i125, i1* %return_value_memory.i60
  %return_value.i126 = load i1, i1* %return_value_memory.i60
  %. = select i1 %return_value.i126, i32 1, i32 0
  %select_to_float = sitofp i32 %. to float
  ret void

some_on:
  ret void
}

