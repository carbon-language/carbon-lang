; Test CFG simplify removal of branch instructions...
;
; RUN: if as < %s | opt -simplifycfg | dis | grep br
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi


void "test1"() {
	br label %BB1
BB1:
	ret void
}

void "test2"() {
	ret void
BB1:
	ret void
}

void "test3"(bool %T) {
	br bool %T, label %BB1, label %BB1
BB1:
	ret void
}


