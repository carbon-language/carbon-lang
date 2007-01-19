; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep movmi &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep moveq &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep movgt &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep movge &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep movne &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fcmped | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fcmpes | wc -l | grep 6

int %f1(float %a) {
entry:
	%tmp = setlt float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f2(float %a) {
entry:
	%tmp = seteq float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f3(float %a) {
entry:
	%tmp = setgt float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f4(float %a) {
entry:
	%tmp = setge float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f5(float %a) {
entry:
	%tmp = setle float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f6(float %a) {
entry:
	%tmp = setne float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %g1(double %a) {
entry:
	%tmp = setlt double %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}
