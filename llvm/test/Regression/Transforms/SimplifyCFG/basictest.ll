; Test CFG simplify removal of branch instructions...
;
; RUN: as < %s | opt -simplifycfg | dis | not grep br


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


