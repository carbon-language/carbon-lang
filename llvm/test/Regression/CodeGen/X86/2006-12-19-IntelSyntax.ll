; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -x86-asm-syntax=intel
; PR1061

target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

implementation   ; Functions:

void %bar(uint %n) {
entry:
	switch uint %n, label %bb12 [
		 uint 1, label %bb
		 uint 2, label %bb6
		 uint 4, label %bb7
		 uint 5, label %bb8
		 uint 6, label %bb10
		 uint 7, label %bb1
		 uint 8, label %bb3
		 uint 9, label %bb4
		 uint 10, label %bb9
		 uint 11, label %bb2
		 uint 12, label %bb5
		 uint 13, label %bb11
	]

bb:		; preds = %entry
	call void (...)* %foo1( )
	ret void

bb1:		; preds = %entry
	call void (...)* %foo2( )
	ret void

bb2:		; preds = %entry
	call void (...)* %foo6( )
	ret void

bb3:		; preds = %entry
	call void (...)* %foo3( )
	ret void

bb4:		; preds = %entry
	call void (...)* %foo4( )
	ret void

bb5:		; preds = %entry
	call void (...)* %foo5( )
	ret void

bb6:		; preds = %entry
	call void (...)* %foo1( )
	ret void

bb7:		; preds = %entry
	call void (...)* %foo2( )
	ret void

bb8:		; preds = %entry
	call void (...)* %foo6( )
	ret void

bb9:		; preds = %entry
	call void (...)* %foo3( )
	ret void

bb10:		; preds = %entry
	call void (...)* %foo4( )
	ret void

bb11:		; preds = %entry
	call void (...)* %foo5( )
	ret void

bb12:		; preds = %entry
	call void (...)* %foo6( )
	ret void
}

declare void %foo1(...)

declare void %foo2(...)

declare void %foo6(...)

declare void %foo3(...)

declare void %foo4(...)

declare void %foo5(...)
