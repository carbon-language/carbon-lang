; RUN: llc < %s -march=x86-64 | FileCheck %s

@var = external global i64		; <i64*> [#uses=1]

define i32 @main() nounwind {
entry:
; CHECK: main:
; CHECK: lock
; CHECK: decq
	tail call i64 @llvm.atomic.load.sub.i64.p0i64( i64* @var, i64 1 )		; <i64>:0 [#uses=0]
	unreachable
}

declare i64 @llvm.atomic.load.sub.i64.p0i64(i64*, i64) nounwind
