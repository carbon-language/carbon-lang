; Ensure constant propogation of remainder instructions is working correctly.

; RUN: if as < %s | opt -constprop -die | dis | grep rem
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test1"() {
	%R = rem int 4, 3
	ret int %R
}

int "test2"() {
	%R = rem int 123, -23
	ret int %R
}

float "test3"() {
	%R = rem float 12.45, 123.213
	ret float %R
}

double "test4"() {
	%R = rem double 312.20213123, 12.3333412
	ret double %R
}
