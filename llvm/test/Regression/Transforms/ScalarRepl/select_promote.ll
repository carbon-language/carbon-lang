; Test promotion of loads that use the result of a select instruction.

; RUN: llvm-as < %s | opt -mem2reg | llvm-dis | not grep alloca

int %main() {
        %mem_tmp.0 = alloca int         ; <int*> [#uses=3]
        %mem_tmp.1 = alloca int         ; <int*> [#uses=3]
        store int 0, int* %mem_tmp.0
        store int 1, int* %mem_tmp.1
        %tmp.1.i = load int* %mem_tmp.1         ; <int> [#uses=1]
        %tmp.3.i = load int* %mem_tmp.0         ; <int> [#uses=1]
        %tmp.4.i = setle int %tmp.1.i, %tmp.3.i         ; <bool> [#uses=1]
        %mem_tmp.i.0 = select bool %tmp.4.i, int* %mem_tmp.1, int* %mem_tmp.0           ; <int*> [#uses=1]
        %tmp.3 = load int* %mem_tmp.i.0         ; <int> [#uses=1]
        ret int %tmp.3
}

