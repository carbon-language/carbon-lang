; RUN: if as < %s | opt -funcresolve -disable-output 2>&1 | grep WARNING
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
%__popcount_tab = external constant [0 x ubyte]
%__popcount_tab = constant [4 x ubyte] c"\00\01\01\02"

void %test() {
	getelementptr [0 x ubyte]* %__popcount_tab, long 0, long 2
	getelementptr [4 x ubyte]* %__popcount_tab, long 0, long 2
	ret void
}

