; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep foos+5
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep foos+1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep foos+9
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep bara+19
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep bara+4

; make sure we compute the correct offset for a packed structure

;Note: codegen for this could change rendering the above checks wrong

; ModuleID = 'foo.c'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
	%struct.anon = type <{ sbyte, int, int, int }>
%foos = external global %struct.anon
%bara = weak global [4 x <{ int, sbyte }>] zeroinitializer

implementation   ; Functions:

int %foo() {
entry:
	%tmp = load int*  getelementptr (%struct.anon* %foos, int 0, uint 1)
	%tmp3 = load int* getelementptr (%struct.anon* %foos, int 0, uint 2)
	%tmp6 = load int* getelementptr (%struct.anon* %foos, int 0, uint 3)
	%tmp4 = add int %tmp3, %tmp
	%tmp7 = add int %tmp4, %tmp6
	ret int %tmp7
}

sbyte %bar() {
entry:
	%tmp = load sbyte* getelementptr([4 x <{ int, sbyte }>]* %bara, int 0, int 0, uint 1 )
	%tmp4 = load sbyte* getelementptr ([4 x <{ int, sbyte }>]* %bara, int 0, int 3, uint 1)
	%tmp5 = add sbyte %tmp4, %tmp
	ret sbyte %tmp5
}
