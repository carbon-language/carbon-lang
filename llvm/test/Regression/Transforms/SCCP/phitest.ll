; RUN: llvm-as < %s | opt -sccp -dce -simplifycfg | llvm-dis | not grep br

int %test(int %param) {
entry:
	%tmp.1 = setne int %param, 0
	br bool %tmp.1, label %endif.0, label %else

else:
	br label %endif.0

endif.0:
	%a.0 = phi int [ 2, %else ], [ 3, %entry ]
	%b.0 = phi int [ 3, %else ], [ 2, %entry ]
	%tmp.5 = add int %a.0, %b.0
	%tmp.7 = setne int %tmp.5, 5
	br bool %tmp.7, label %UnifiedReturnBlock, label %endif.1

endif.1:
	ret int 0

UnifiedReturnBlock:
	ret int 2
}
