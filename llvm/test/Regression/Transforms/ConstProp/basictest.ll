; This is a basic sanity check for constant propogation.  The add instruction 
; should be eliminated.

; RUN: if as < %s | opt -constprop -die | dis | grep add
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(bool %B)
begin
	br bool %B, label %BB1, label %BB2
BB1:
	%Val = add int 0, 0
	br label %BB3
BB2:
	br label %BB3
BB3:
	%Ret = phi int [%Val, %BB1], [1, %BB2]
	ret int %Ret
end
