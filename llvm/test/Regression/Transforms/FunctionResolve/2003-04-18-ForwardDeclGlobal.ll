; RUN: as < %s | opt -funcresolve -disable-output 2>&1 | not grep WARNING

%__popcount_tab = external constant [0 x ubyte]
%__popcount_tab = constant [4 x ubyte] c"\00\01\01\02"

declare void %foo(ubyte *)

void %test() {
	getelementptr [0 x ubyte]* %__popcount_tab, long 0, long 2
	getelementptr [4 x ubyte]* %__popcount_tab, long 0, long 2
	call void %foo(ubyte * getelementptr ([0 x ubyte]* %__popcount_tab, long 0, long 2))
	ret void
}

