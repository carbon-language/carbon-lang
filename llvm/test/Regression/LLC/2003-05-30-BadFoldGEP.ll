;; Date:     May 28, 2003.
;; From:     test/Programs/External/SPEC/CINT2000/254.gap.llvm.bc
;; Function: int %OpenOutput(sbyte* %filename.1)
;;
;; Error:    A sequence of GEPs is folded incorrectly by llc during selection
;;	     causing an assertion about a dynamic casting error.
;;	     This code sequence was produced (correctly) by preselection
;;	     from a nested pair of ConstantExpr getelementptrs.
;;	     The code below is the output of preselection.
;;	     The original ConstantExprs are included in a comment.
;;
;; Cause:    FoldGetElemChain() was inserting an extra leading 0 even though
;;	     the first instruction in the sequence contributes no indices.
;;	     The next instruction contributes a leading non-zero so another
;;	     zero should not be added before it!
;;


%FileType = type { int, [256 x sbyte], int, int, int, int }
%OutputFiles = uninitialized global [16 x %FileType]
%Output = internal global %FileType* null


implementation; Functions:

internal int %OpenOutput(sbyte* %filename.1) {
entry:
	%tmp.0 = load %FileType** %Output      
        %tmp.4 = getelementptr %FileType* %tmp.0, long 1

	;;------ Original instruction in 254.gap.llvm.bc:
	;; %tmp.10 = seteq { int, [256 x sbyte], int, int, int, int }* %tmp.4, getelementptr ([16 x { int, [256 x sbyte], int, int, int, int }]* getelementptr ([16 x { int, [256 x sbyte], int, int, int, int }]* %OutputFiles, long 1), long 0, long 0)

	;;------ Code sequence produced by preselection phase for above instr:
	;; This code sequence is folded incorrectly by llc during selection
	;; causing an assertion about a dynamic casting error.
        %addrOfGlobal = getelementptr [16 x %FileType]* %OutputFiles, long 0
        %constantGEP = getelementptr [16 x %FileType]* %addrOfGlobal, long 1
        %constantGEP = getelementptr [16 x %FileType]* %constantGEP, long 0, long 0
	%tmp.10 = seteq %FileType* %tmp.4, %constantGEP
        br bool %tmp.10, label %return, label %endif.0

endif.0:
	ret int 0

return:
	ret int 1
}
