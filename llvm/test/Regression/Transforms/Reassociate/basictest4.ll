; RUN: llvm-as < %s | opt -reassociate -gcse -instcombine | llvm-dis | not grep add

%a = weak global int 0
%b = weak global int 0
%c = weak global int 0
%d = weak global int 0

implementation

int %foo() {
	%tmp.0 = load int* %a
	%tmp.1 = load int* %b
	%tmp.2 = add int %tmp.0, %tmp.1   ; (a+b)
	%tmp.4 = load int* %c
	%tmp.5 = add int %tmp.2, %tmp.4   ; (a+b)+c
	%tmp.8 = add int %tmp.0, %tmp.4   ; (a+c)
	%tmp.11 = add int %tmp.8, %tmp.1  ; (a+c)+b
	%RV = xor int %tmp.5, %tmp.11     ; X ^ X = 0
	ret int %RV
}
