; RUN: llvm-as < %s | opt -cee -constprop -instcombine -dce | llvm-dis | not grep 'REMOVE'

int %test_case_values_should_propagate(int %A) {
    switch int %A, label %D [
        int 40, label %C1
        int 41, label %C2
        int 42, label %C3
    ]
C1:
    %REMOVE1 = add int %A, 2     ; Should be 42.
    ret int %REMOVE1
C2:
    %REMOVE2 = add int %A, 3     ; Should be 44.
    ret int %REMOVE2
C3:
    %REMOVE3 = add int %A, 4     ; Should be 46.
    ret int %REMOVE3
D:
    ret int 10
}
