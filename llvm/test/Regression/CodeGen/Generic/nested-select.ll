; RUN: llvm-as < %s | llc -o /dev/null -f

; Test that select of a select works

int %ic_test(double %p.0.2.0.val, double %p.0.2.1.val, double %p.0.2.2.val, %typedef.tree* %t) {
        %result.1.0 = cast bool false to int            ; <int> [#uses=1]
        %tmp.55 = setge double 0.000000e+00, 1.000000e+00               ; <bool> [#uses=1]
        %tmp.66 = div double 0.000000e+00, 0.000000e+00         ; <double> [#uses=1]
	br label %N
N:
        %result.1.1 = select bool %tmp.55, int 0, int %result.1.0               ; <int> [#uses=1]
        %tmp.75 = setge double %tmp.66, 1.000000e+00            ; <bool> [#uses=1]
        %retval1 = select bool %tmp.75, int 0, int %result.1.1          ; <int> [#uses=1]
        ret int %retval1
}

