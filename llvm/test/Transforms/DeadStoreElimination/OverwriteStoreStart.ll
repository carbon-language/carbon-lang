; RUN: opt < %s -basicaa -dse -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct.vec2 = type { <4 x i32>, <4 x i32> }
%struct.vec2plusi = type { <4 x i32>, <4 x i32>, i32 }

@glob1 = global %struct.vec2 zeroinitializer, align 16
@glob2 = global %struct.vec2plusi zeroinitializer, align 16

define void @write4to8(i32* nocapture %p) nounwind uwtable ssp {
; CHECK: @write4to8
entry:
  %arrayidx0 = getelementptr inbounds i32* %p, i64 1
  %p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* %{{[0-9]+}}, i8 0, i64 24, i32 4, i1 false)
  call void @llvm.memset.p0i8.i64(i8* %p3, i8 0, i64 28, i32 4, i1 false)
  %arrayidx1 = getelementptr inbounds i32* %p, i64 1
  store i32 1, i32* %arrayidx1, align 4
  ret void
}

define void @write4to12(i32* nocapture %p) nounwind uwtable ssp {
; CHECK: @write4to12
entry:
%arrayidx0 = getelementptr inbounds i32* %p, i64 1
%p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* %{{[0-9]+}}, i8 0, i64 20, i32 4, i1 false)
call void @llvm.memset.p0i8.i64(i8* %p3, i8 0, i64 28, i32 4, i1 false)
%arrayidx1 = bitcast i32* %arrayidx0 to i64*
store i64 1, i64* %arrayidx1, align 4
ret void
}

define void @write4to8_2(i32* nocapture %p) nounwind uwtable ssp {
; CHECK: @write4to8_2
entry:
%arrayidx0 = getelementptr inbounds i32* %p, i64 1
%p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* %{{[0-9]+}}, i8 0, i64 24, i32 4, i1 false)
call void @llvm.memset.p0i8.i64(i8* %p3, i8 0, i64 28, i32 4, i1 false)
%arrayidx1 = bitcast i32* %p to i64*
store i64 1, i64* %arrayidx1, align 4
ret void
}

define void @dontwrite4to6(i32* nocapture %p) nounwind uwtable ssp {
; CHECK: @dontwrite4to6
entry:
%arrayidx0 = getelementptr inbounds i32* %p, i64 1
%p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* %p3, i8 0, i64 28, i32 4, i1 false)
call void @llvm.memset.p0i8.i64(i8* %p3, i8 0, i64 28, i32 4, i1 false)
%arrayidx1 = bitcast i32* %arrayidx0 to i16*
store i16 1, i16* %arrayidx1, align 4
ret void
}

define void @write4to8_neg_gep(i32* nocapture %p) nounwind uwtable ssp {
; CHECK: @write4to8_neg_gep
entry:
%arrayidx0 = getelementptr inbounds i32* %p, i64 -1
%p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* %{{[0-9]+}}, i8 0, i64 24, i32 4, i1 false)
call void @llvm.memset.p0i8.i64(i8* %p3, i8 0, i64 28, i32 4, i1 false)
%neg2 = getelementptr inbounds i32* %p, i64 -2
%arrayidx1 = bitcast i32* %neg2 to i64*
store i64 1, i64* %arrayidx1, align 4
ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
