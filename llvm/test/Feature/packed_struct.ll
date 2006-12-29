; RUN: llvm-upgrade < %s | llvm-as | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll &&
; RUN: not grep cast %t2.ll &&
; RUN: grep "<{" %t2.ll

%struct.anon = type <{ sbyte, int, int, int }>
%foos = external global %struct.anon 
%bara = external global [2 x <{ int, sbyte }>]

;initializers should work for packed and non-packed the same way
%E1 = global <{sbyte, int, int}> {sbyte 1, int 2, int 3}
%E2 = global {sbyte, int, int} {sbyte 4, int 5, int 6}

implementation   ; Functions:

int %main() 
{
        %tmp = load int*  getelementptr (%struct.anon* %foos, int 0, uint 1)            ; <int> [#uses=1]
        %tmp3 = load int* getelementptr (%struct.anon* %foos, int 0, uint 2)            ; <int> [#uses=1]
        %tmp6 = load int* getelementptr (%struct.anon* %foos, int 0, uint 3)            ; <int> [#uses=1]
        %tmp4 = add int %tmp3, %tmp             ; <int> [#uses=1]
        %tmp7 = add int %tmp4, %tmp6            ; <int> [#uses=1]
        ret int %tmp7
}

int %bar() {
entry:
        %tmp = load int* getelementptr([2 x <{ int, sbyte }>]* %bara, int 0, int 0, uint 0 )            ; <int> [#uses=1]
        %tmp4 = load int* getelementptr ([2 x <{ int, sbyte }>]* %bara, int 0, int 1, uint 0)           ; <int> [#uses=1]
        %tmp5 = add int %tmp4, %tmp             ; <int> [#uses=1]
        ret int %tmp5
}
