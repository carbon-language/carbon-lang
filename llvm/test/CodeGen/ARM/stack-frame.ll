; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -march=arm | grep add | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=thumb
; RUN: llvm-as < %s | llc -march=thumb | grep add | wc -l | grep 1

define void @f1() {
	%c = alloca i8, align 1
	ret void
}

define i32 @f2() {
	ret i32 1
}


