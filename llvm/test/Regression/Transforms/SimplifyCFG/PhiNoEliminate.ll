; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep select

;; The PHI node in this example should not be turned into a select, as we are
;; not able to ifcvt the entire block.  As such, converting to a select just 
;; introduces inefficiency without saving copies.

int %bar(bool %C) {
entry:
        br bool %C, label %then, label %endif

then:
        %tmp.3 = call int %qux()
        br label %endif

endif:
	%R = phi int [123, %entry], [12312, %then]
	;; stuff to disable tail duplication
        call int %qux()
        call int %qux()
        call int %qux()
        call int %qux()
        call int %qux()
        call int %qux()
        call int %qux()
        ret int %R
}

declare int %qux()

