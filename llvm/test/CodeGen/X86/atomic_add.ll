; RUN: llc < %s -march=x86-64 | FileCheck %s

; rdar://7103704

define void @sub1(i32* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: sub1:
; CHECK: subl
	%0 = tail call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %p, i32 %v)		; <i32> [#uses=0]
	ret void
}

define void @inc4(i64* nocapture %p) nounwind ssp {
entry:
; CHECK: inc4:
; CHECK: incq
	%0 = tail call i64 @llvm.atomic.load.add.i64.p0i64(i64* %p, i64 1)		; <i64> [#uses=0]
	ret void
}

declare i64 @llvm.atomic.load.add.i64.p0i64(i64* nocapture, i64) nounwind

define void @add8(i64* nocapture %p) nounwind ssp {
entry:
; CHECK: add8:
; CHECK: addq $2
	%0 = tail call i64 @llvm.atomic.load.add.i64.p0i64(i64* %p, i64 2)		; <i64> [#uses=0]
	ret void
}

define void @add4(i64* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: add4:
; CHECK: addq
	%0 = sext i32 %v to i64		; <i64> [#uses=1]
	%1 = tail call i64 @llvm.atomic.load.add.i64.p0i64(i64* %p, i64 %0)		; <i64> [#uses=0]
	ret void
}

define void @inc3(i8* nocapture %p) nounwind ssp {
entry:
; CHECK: inc3:
; CHECK: incb
	%0 = tail call i8 @llvm.atomic.load.add.i8.p0i8(i8* %p, i8 1)		; <i8> [#uses=0]
	ret void
}

declare i8 @llvm.atomic.load.add.i8.p0i8(i8* nocapture, i8) nounwind

define void @add7(i8* nocapture %p) nounwind ssp {
entry:
; CHECK: add7:
; CHECK: addb $2
	%0 = tail call i8 @llvm.atomic.load.add.i8.p0i8(i8* %p, i8 2)		; <i8> [#uses=0]
	ret void
}

define void @add3(i8* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: add3:
; CHECK: addb
	%0 = trunc i32 %v to i8		; <i8> [#uses=1]
	%1 = tail call i8 @llvm.atomic.load.add.i8.p0i8(i8* %p, i8 %0)		; <i8> [#uses=0]
	ret void
}

define void @inc2(i16* nocapture %p) nounwind ssp {
entry:
; CHECK: inc2:
; CHECK: incw
	%0 = tail call i16 @llvm.atomic.load.add.i16.p0i16(i16* %p, i16 1)		; <i16> [#uses=0]
	ret void
}

declare i16 @llvm.atomic.load.add.i16.p0i16(i16* nocapture, i16) nounwind

define void @add6(i16* nocapture %p) nounwind ssp {
entry:
; CHECK: add6:
; CHECK: addw $2
	%0 = tail call i16 @llvm.atomic.load.add.i16.p0i16(i16* %p, i16 2)		; <i16> [#uses=0]
	ret void
}

define void @add2(i16* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: add2:
; CHECK: addw
	%0 = trunc i32 %v to i16		; <i16> [#uses=1]
	%1 = tail call i16 @llvm.atomic.load.add.i16.p0i16(i16* %p, i16 %0)		; <i16> [#uses=0]
	ret void
}

define void @inc1(i32* nocapture %p) nounwind ssp {
entry:
; CHECK: inc1:
; CHECK: incl
	%0 = tail call i32 @llvm.atomic.load.add.i32.p0i32(i32* %p, i32 1)		; <i32> [#uses=0]
	ret void
}

declare i32 @llvm.atomic.load.add.i32.p0i32(i32* nocapture, i32) nounwind

define void @add5(i32* nocapture %p) nounwind ssp {
entry:
; CHECK: add5:
; CHECK: addl $2
	%0 = tail call i32 @llvm.atomic.load.add.i32.p0i32(i32* %p, i32 2)		; <i32> [#uses=0]
	ret void
}

define void @add1(i32* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: add1:
; CHECK: addl
	%0 = tail call i32 @llvm.atomic.load.add.i32.p0i32(i32* %p, i32 %v)		; <i32> [#uses=0]
	ret void
}

define void @dec4(i64* nocapture %p) nounwind ssp {
entry:
; CHECK: dec4:
; CHECK: decq
	%0 = tail call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %p, i64 1)		; <i64> [#uses=0]
	ret void
}

declare i64 @llvm.atomic.load.sub.i64.p0i64(i64* nocapture, i64) nounwind

define void @sub8(i64* nocapture %p) nounwind ssp {
entry:
; CHECK: sub8:
; CHECK: subq $2
	%0 = tail call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %p, i64 2)		; <i64> [#uses=0]
	ret void
}

define void @sub4(i64* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: sub4:
; CHECK: subq
	%0 = sext i32 %v to i64		; <i64> [#uses=1]
	%1 = tail call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %p, i64 %0)		; <i64> [#uses=0]
	ret void
}

define void @dec3(i8* nocapture %p) nounwind ssp {
entry:
; CHECK: dec3:
; CHECK: decb
	%0 = tail call i8 @llvm.atomic.load.sub.i8.p0i8(i8* %p, i8 1)		; <i8> [#uses=0]
	ret void
}

declare i8 @llvm.atomic.load.sub.i8.p0i8(i8* nocapture, i8) nounwind

define void @sub7(i8* nocapture %p) nounwind ssp {
entry:
; CHECK: sub7:
; CHECK: subb $2
	%0 = tail call i8 @llvm.atomic.load.sub.i8.p0i8(i8* %p, i8 2)		; <i8> [#uses=0]
	ret void
}

define void @sub3(i8* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: sub3:
; CHECK: subb
	%0 = trunc i32 %v to i8		; <i8> [#uses=1]
	%1 = tail call i8 @llvm.atomic.load.sub.i8.p0i8(i8* %p, i8 %0)		; <i8> [#uses=0]
	ret void
}

define void @dec2(i16* nocapture %p) nounwind ssp {
entry:
; CHECK: dec2:
; CHECK: decw
	%0 = tail call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %p, i16 1)		; <i16> [#uses=0]
	ret void
}

declare i16 @llvm.atomic.load.sub.i16.p0i16(i16* nocapture, i16) nounwind

define void @sub6(i16* nocapture %p) nounwind ssp {
entry:
; CHECK: sub6:
; CHECK: subw $2
	%0 = tail call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %p, i16 2)		; <i16> [#uses=0]
	ret void
}

define void @sub2(i16* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK: sub2:
; CHECK: subw
	%0 = trunc i32 %v to i16		; <i16> [#uses=1]
	%1 = tail call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %p, i16 %0)		; <i16> [#uses=0]
	ret void
}

define void @dec1(i32* nocapture %p) nounwind ssp {
entry:
; CHECK: dec1:
; CHECK: decl
	%0 = tail call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %p, i32 1)		; <i32> [#uses=0]
	ret void
}

declare i32 @llvm.atomic.load.sub.i32.p0i32(i32* nocapture, i32) nounwind

define void @sub5(i32* nocapture %p) nounwind ssp {
entry:
; CHECK: sub5:
; CHECK: subl $2
	%0 = tail call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %p, i32 2)		; <i32> [#uses=0]
	ret void
}
