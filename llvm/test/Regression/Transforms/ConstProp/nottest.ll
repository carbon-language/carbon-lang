; Ensure constant propogation of 'not' instructions is working correctly.

; RUN: if as < %s | opt -constprop -die | dis | grep xor
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test1"() {
	%R = xor int 4, -1
	ret int %R
}

int "test2"() {
	%R = xor int -23, -1
	ret int %R
}

bool "test3"() {
	%R = xor bool true, true
	ret bool %R
}

