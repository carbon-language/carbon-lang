; SetCC on boolean values was not implemented!

; RUN: if as < %s | opt -constprop -die | dis | grep 'set'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

bool "test1"() {
	%A = setle bool true, false
	%B = setge bool true, false
	%C = setlt bool false, true
	%D = setgt bool true, false
	%E = seteq bool false, false
	%F = setne bool false, true
	%G = and bool %A, %B
	%H = and bool %C, %D
	%I = and bool %E, %F
	%J = and bool %G, %H
	%K = and bool %I, %J
	ret bool %K
}

