; Test merging of blocks with phi nodes.
;
; RUN: if as < %s | opt -simplifycfg | dis | grep 'N:'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
;

int %test(bool %a) {
Q:
	br bool %a, label %N, label %M
N:
	br label %M
M:
	; It's ok to merge N and M because the incoming values for W are the 
        ; same for both cases...
	%W = phi int [2, %N], [2, %Q]
	%R = add int %W, 1
	ret int %R
}

