; RUN: llvm-as < %s | opt -sccp -adce -simplifycfg | llvm-dis | not grep br

; No matter how hard you try, sqrt(1.0) is always 1.0.  This allows the
; optimizer to delete this loop.

declare double %sqrt(double)

double %test(uint %param) {
entry:
	br label %Loop

Loop:
	%I2 = phi uint [ 0, %entry ], [ %I3, %Loop ]
	%V  = phi double [ 1.0, %entry], [ %V2, %Loop ]

	%V2 = call double %sqrt(double %V)

	%I3 = add uint %I2, 1
	%tmp.7 = setne uint %I3, %param
	br bool %tmp.7, label %Loop, label %Exit

Exit:
	ret double %V
}
