; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep "ldrb.*7"

	%struct.A = type { i8, i8, i8, i8, i16, i8, i8, %struct.B** }
	%struct.B = type { float, float, i32, i32, i32, [0 x i8] }

implementation   ; Functions:

define i32 @f1(%struct.A* %d) {
entry:
	%tmp2 = getelementptr %struct.A* %d, i32 0, i32 4
	%tmp23 = bitcast i16* %tmp2 to i32*
	%tmp4 = load i32* %tmp23
	%tmp512 = lshr i32 %tmp4, 24
	%tmp56 = trunc i32 %tmp512 to i8
	icmp eq i8 %tmp56, 0
	br i1 %0, label %UnifiedReturnBlock, label %conArue

conArue:
	%tmp8 = tail call i32 @f( %struct.A* %d )
	ret i32 %tmp8

UnifiedReturnBlock:
	ret i32 0
}

declare i32 @f(%struct.A*)
