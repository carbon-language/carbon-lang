; Ensure constant propogation of 'not' instructions is working correctly.

; RUN: if as < %s | opt -constprop | dis | grep not
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test1"() {
	%R = not int 4
	ret int %R
}

int "test2"() {
	%R = not int -23
	ret int %R
}

bool "test3"() {
	%R = not bool true
	ret bool %R
}

