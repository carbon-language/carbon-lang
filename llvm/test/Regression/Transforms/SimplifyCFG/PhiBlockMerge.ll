; Test merging of blocks that only have PHI nodes in them
;
; RUN: as < %s | opt -simplifycfg | dis | not grep 'N:'
;

int %test(bool %a, bool %b) {
        br bool %a, label %M, label %O

O:
	br bool %b, label %N, label %Q
Q:
	br label %N
N:
	%Wp = phi int [0, %O], [1, %Q]
	; This block should be foldable into M
	br label %M

M:
	%W = phi int [%Wp, %N], [2, %0]
	%R = add int %W, 1
	ret int %R
}

