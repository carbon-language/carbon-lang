; Ensure constant propogation of remainder instructions is working correctly.

; RUN: if as < %s | opt -constprop | dis | grep rem
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

