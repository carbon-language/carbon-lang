; RUN: opt -passes=instcombine -S < %s | FileCheck %s
target datalayout = "e-m:o-p:40:64:64:32-i64:64-f80:128-n8:16:32:64-S128"

; check that memory builtins can be handled.
define i64 @objsize1_custom_idx(i64 %sz) {
entry:
  %ptr = call i8* @malloc(i64 %sz)
  %ptr2 = getelementptr inbounds i8, i8* %ptr, i32 2
  %calc_size = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr2, i1 false, i1 true, i1 true)
  ret i64 %calc_size
}

%struct.V = type { [10 x i8], i32, [10 x i8] }

define i32 @objsize2_custom_idx() #0 {
entry:
  %var = alloca %struct.V, align 4
  %0 = bitcast %struct.V* %var to i8*
  call void @llvm.lifetime.start.p0i8(i64 28, i8* %0) #3
  %buf1 = getelementptr inbounds %struct.V, %struct.V* %var, i32 0, i32 0
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf1, i64 0, i64 1
  %1 = call i64 @llvm.objectsize.i64.p0i8(i8* %arrayidx, i1 false, i1 false, i1 false)
  %conv = trunc i64 %1 to i32
  call void @llvm.lifetime.end.p0i8(i64 28, i8* %0) #3
  ret i32 %conv
; CHECK: ret i32 27
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1
declare i8* @malloc(i64)
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1, i1)
