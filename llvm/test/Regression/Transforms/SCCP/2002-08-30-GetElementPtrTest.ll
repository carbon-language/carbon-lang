; RUN: if as < %s | opt -sccp | dis | grep '%X'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%G = uninitialized global [40x int]

implementation

int* %test() {
	%X = getelementptr [40x int]* %G, uint 0, uint 0
	ret int* %X
}
