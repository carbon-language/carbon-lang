; RUN: llvm-as < %s | opt -simplifycfg -instcombine -simplifycfg | llvm-dis | not grep call

declare void %bar()

void %test(int %X, int %Y) {
entry:
        %tmp.2 = setne int %X, %Y
        br bool %tmp.2, label %shortcirc_next, label %UnifiedReturnBlock

shortcirc_next:
        %tmp.3 = setne int %X, %Y
        br bool %tmp.3, label %UnifiedReturnBlock, label %then

then:
        call void %bar( )
        ret void

UnifiedReturnBlock:             ; preds = %entry, %shortcirc_next
	ret void
}

