; RUN: llvm-as < %s | llc -march=x86 -relocation-model=static | grep 'movl _last' | wc -l | grep 2

%block = external global ubyte*		; <ubyte**> [#uses=1]
%last = external global int		; <int*> [#uses=3]

implementation   ; Functions:

bool %loadAndRLEsource_no_exit_2E_1_label_2E_0(int %tmp.21.reload, int %tmp.8) {
newFuncRoot:
	br label %label.0

label.0.no_exit.1_crit_edge.exitStub:		; preds = %label.0
	ret bool true

codeRepl5.exitStub:		; preds = %label.0
	ret bool false

label.0:		; preds = %newFuncRoot
	%tmp.35 = load int* %last		; <int> [#uses=1]
	%inc.1 = add int %tmp.35, 1		; <int> [#uses=2]
	store int %inc.1, int* %last
	%tmp.36 = load ubyte** %block		; <ubyte*> [#uses=1]
	%tmp.38 = getelementptr ubyte* %tmp.36, int %inc.1		; <ubyte*> [#uses=1]
	%tmp.40 = cast int %tmp.21.reload to ubyte		; <ubyte> [#uses=1]
	store ubyte %tmp.40, ubyte* %tmp.38
	%tmp.910 = load int* %last		; <int> [#uses=1]
	%tmp.1111 = setlt int %tmp.910, %tmp.8		; <bool> [#uses=1]
	%tmp.1412 = setne int %tmp.21.reload, 257		; <bool> [#uses=1]
	%tmp.1613 = and bool %tmp.1111, %tmp.1412		; <bool> [#uses=1]
	br bool %tmp.1613, label %label.0.no_exit.1_crit_edge.exitStub, label %codeRepl5.exitStub
}
