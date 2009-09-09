; RUN: llc < %s -march=ppc32
; RUN: llc < %s -march=ppc64

define {i64, float} @bar(i64 %a, float %b) {
        %y = add i64 %a, 7
        %z = fadd float %b, 7.0
	ret i64 %y, float %z
}

define i64 @foo() {
	%M = call {i64, float} @bar(i64 21, float 21.0)
        %N = getresult {i64, float} %M, 0
        %O = getresult {i64, float} %M, 1
        %P = fptosi float %O to i64
        %Q = add i64 %P, %N
	ret i64 %Q
}
