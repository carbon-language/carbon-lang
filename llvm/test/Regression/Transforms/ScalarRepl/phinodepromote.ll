; RUN: llvm-as < %s | opt -mem2reg | llvm-dis | not grep alloca
;
; This tests to see if mem2reg can promote alloca instructions whose addresses
; are used by PHI nodes that are immediately loaded.  The LLVM C++ front-end
; often generates code that looks like this (when it codegen's ?: exprs as 
; lvalues), so handling this simple extension is quite useful.
;
; This testcase is what the following program looks like when it reaches
; mem2reg:
;
; template<typename T>
; const T& max(const T& a1, const T& a2) { return a1 < a2 ? a1 : a2; }
; int main() { return max(0, 1); }
;

int %main() {
entry:
        %mem_tmp.0 = alloca int
        %mem_tmp.1 = alloca int
        store int 0, int* %mem_tmp.0
        store int 1, int* %mem_tmp.1
        %tmp.1.i = load int* %mem_tmp.1
        %tmp.3.i = load int* %mem_tmp.0
        %tmp.4.i = setle int %tmp.1.i, %tmp.3.i
        br bool %tmp.4.i, label %cond_true.i, label %cond_continue.i

cond_true.i:
        br label %cond_continue.i

cond_continue.i:
        %mem_tmp.i.0 = phi int* [ %mem_tmp.1, %cond_true.i ], [ %mem_tmp.0, %entry ]
        %tmp.3 = load int* %mem_tmp.i.0
        ret int %tmp.3
}

