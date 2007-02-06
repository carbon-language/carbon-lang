; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm -enable-thumb &&
; RUN: llvm-as < %s | llc -march=arm -enable-thumb | not grep "ldr sp"

	%struct.state = type { i32, %struct.info*, float**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i8* }
	%struct.info = type { i32, i32, i32, i32, i32, i32, i32, i8* }

define void @f1(%struct.state* %v) {
	%tmp6 = load i32* null
	%tmp8 = alloca float, i32 %tmp6
	store i32 1, i32* null
	br i1 false, label %bb123.preheader, label %return

bb123.preheader:
	br i1 false, label %bb43, label %return

bb43:
	call fastcc void @f2( float* %tmp8, float* null, i32 0 )
	%tmp70 = load i32* null
	%tmp85 = getelementptr float* %tmp8, i32 0
	call fastcc void @f3( float* null, float* null, float* %tmp85, i32 %tmp70 )
	ret void

return:
	ret void
}

declare fastcc void @f2(float*, float*, i32)

declare fastcc void @f3(float*, float*, float*, i32)
