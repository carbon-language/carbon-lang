; RUN: as < %s | opt -simplifycfg | dis | not grep switch

int %test1() {   ; Test normal folding
	switch uint 5, label %Default [
		uint 0, label %Foo
		uint 1, label %Bar
		uint 2, label %Baz
		uint 5, label %TheDest
	]
Default:ret int -1
Foo:	ret int -2
Bar:	ret int -3
Baz: 	ret int -4
TheDest:ret int 1234
}

int %test2() {   ; Test folding to default dest
	switch uint 3, label %Default [
		uint 0, label %Foo
		uint 1, label %Bar
		uint 2, label %Baz
		uint 5, label %TheDest
	]
Default:ret int 1234
Foo:	ret int -2
Bar:	ret int -5
Baz: 	ret int -6
TheDest:ret int -8
}

int %test3(bool %C) {   ; Test folding all to same dest
	br bool %C, label %Start, label %TheDest
Start:
	switch uint 3, label %TheDest [
		uint 0, label %TheDest
		uint 1, label %TheDest
		uint 2, label %TheDest
		uint 5, label %TheDest
	]
TheDest: ret int 1234
}

int %test4(uint %C) {   ; Test folding switch -> branch
	switch uint %C, label %L1 [
		uint 0, label %L2
	]
L1:	ret int 0
L2:	ret int 1
}
