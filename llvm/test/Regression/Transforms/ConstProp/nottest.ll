; Ensure constant propogation of 'not' instructions is working correctly.

; RUN: llvm-as < %s | opt -constprop -die | llvm-dis | not grep xor

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

