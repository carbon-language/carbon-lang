; RUN: llvm-as -f < %s > %t1.bc
; RUN: llvm-as -f < `dirname %s`/2004-05-07-TypeResolution2.ll > %t2.bc
; RUN: llvm-link -f -o %t3.bc %t1.bc %t2.bc

target endian = little
target pointersize = 32

  %myint = type opaque
	%struct2 = type { %struct1 }

	%struct1 = type { int, void (%struct2*)*,  %myint *, int (uint *)* }


%driver1 = global %struct1 zeroinitializer		; <%struct1*> [#uses=1]

%m1 = external global [1 x sbyte] * 	; <%struct.task_struct**> [#uses=0]
;%m1 = external global uint  	; <%struct.task_struct**> [#uses=0]
%str1 = constant [1 x ubyte] zeroinitializer
%str2 = constant [2 x ubyte] zeroinitializer
%str3 = constant [3 x ubyte] zeroinitializer
%str4 = constant [4 x ubyte] zeroinitializer
%str5 = constant [5 x ubyte] zeroinitializer
%str6 = constant [6 x ubyte] zeroinitializer
%str7 = constant [7 x ubyte] zeroinitializer
%str8 = constant [8 x ubyte] zeroinitializer
%str9 = constant [9 x ubyte] zeroinitializer
%stra = constant [10 x ubyte] zeroinitializer
%strb = constant [11 x ubyte] zeroinitializer
%strc = constant [12 x ubyte] zeroinitializer
%strd = constant [13 x ubyte] zeroinitializer
%stre = constant [14 x ubyte] zeroinitializer
%strf = constant [15 x ubyte] zeroinitializer
%strg = constant [16 x ubyte] zeroinitializer
%strh = constant [17 x ubyte] zeroinitializer

implementation   ; Functions:

declare void %func(%struct2*)

void %tty_init() {
entry:
	volatile store void (%struct2*)* %func, void (%struct2 *)** getelementptr (%struct1*  %driver1, uint 0, uint 1)
	ret void
}
