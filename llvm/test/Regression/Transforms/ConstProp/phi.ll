; This is a basic sanity check for constant propogation.  The add instruction 
; should be eliminated.

; RUN: if as < %s | opt -constprop -die | dis | grep phi
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(bool %B)
begin
BB0:
	br bool %B, label %BB1, label %BB3
BB1:
	br label %BB3
BB3:
	%Ret = phi int [1, %BB0], [1, %BB1]
	ret int %Ret
end
