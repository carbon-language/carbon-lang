; This function should have exactly one call to fixdfdi, no more!

; RUN: llvm-as < %s | llc -march=ppc32 -mattr=-64bit | grep 'bl .*fixdfdi' | wc -l | grep 1

double %test2(double %tmp.7705) {
        %mem_tmp.2.0.in = cast double %tmp.7705 to long                ; <long> [#uses=1]
        %mem_tmp.2.0 = cast long %mem_tmp.2.0.in to double
	ret double %mem_tmp.2.0
}
