; RUN: llvm-as < %s | llc -fast -march=x86 -mcpu=i386 | not grep 'movl %eax, %edx'

int %foo(int %t, int %C) {
entry:
        br label %cond_true

cond_true:              ; preds = %cond_true, %entry
        %t_addr.0.0 = phi int [ %t, %entry ], [ %tmp7, %cond_true ]             ; <int> [#uses=2]
        %tmp7 = add int %t_addr.0.0, 1  ; <int> [#uses=1]
        %tmp = setgt int %C, 39         ; <bool> [#uses=1]
        br bool %tmp, label %bb12, label %cond_true

bb12:           ; preds = %cond_true
        ret int %t_addr.0.0
}
