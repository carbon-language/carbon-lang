; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep call

declare void %bar()

void %test(int %X, int %Y) {
entry:
        %tmp.2 = setlt int %X, %Y               ; <bool> [#uses=2]
        br bool %tmp.2, label %shortcirc_next, label %UnifiedReturnBlock

shortcirc_next:         ; preds = %entry
        br bool %tmp.2, label %UnifiedReturnBlock, label %then

then:           ; preds = %shortcirc_next
        call void %bar( )
        ret void

UnifiedReturnBlock:             ; preds = %entry, %shortcirc_next
	ret void
}

