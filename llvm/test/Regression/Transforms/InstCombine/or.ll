; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine -dce | dis | grep or
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A) {
	%B = or int %A, 0
	ret int %B
}

int "test2"(int %A) {
	%B = or int %A, -1
	ret int %B
}

bool "test3"(bool %A) {
	%B = or bool %A, false
	ret bool %B
}

bool "test4"(bool %A) {
	%B = or bool %A, true
	ret bool %B
}

bool "test5"(bool %A) {
	%B = xor bool %A, false
	ret bool %B
}

int "test5"(int %A) {
	%B = xor int %A, 0
	ret int %B
}


