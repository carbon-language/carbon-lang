; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine | dis | grep set
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%X = uninitialized global int

bool "test1"(int %A) {
	%B = seteq int %A, %A
	%C = seteq int* %X, null   ; Never true
	%D = and bool %B, %C
	ret bool %D
}

bool "test2"(int %A) {
	%B = setne int %A, %A
	%C = setne int* %X, null   ; Never false
	%D = or bool %B, %C
	ret bool %D
}

bool "test3"(int %A) {
	%B = setlt int %A, %A
	ret bool %B
}

bool "test4"(int %A) {
	%B = setgt int %A, %A
	ret bool %B
}

bool "test5"(int %A) {
	%B = setle int %A, %A
	ret bool %B
}

bool "test6"(int %A) {
	%B = setge int %A, %A
	ret bool %B
}

bool "test7"(uint %A) {
	%B = setge uint %A, 0  ; true
	ret bool %B
}

bool "test8"(uint %A) {
	%B = setlt uint %A, 0  ; false
	ret bool %B
}
