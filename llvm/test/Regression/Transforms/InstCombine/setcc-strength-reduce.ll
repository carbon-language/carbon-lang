; This test ensures that "strength reduction" of conditional expressions are
; working.  Basically this boils down to converting setlt,gt,le,ge instructions
; into equivalent setne,eq instructions.
;

; RUN: as < %s | opt -instcombine | dis | grep -v seteq | grep -v setne | not grep set

bool "test1"(uint %A) {
	%B = setge uint %A, 1   ; setne %A, 0
	ret bool %B
}

bool "test2"(uint %A) {
	%B = setgt uint %A, 0   ; setne %A, 0
	ret bool %B
}

bool "test3"(sbyte %A) {
	%B = setge sbyte %A, -127   ; setne %A, -128
	ret bool %B
}

bool %test4(sbyte %A) {
	%B = setle sbyte %A, 126  ; setne %A, 127
	ret bool %B
}

bool %test5(sbyte %A) {
	%B = setlt sbyte %A, 127 ; setne %A, 127
	ret bool %B
}
