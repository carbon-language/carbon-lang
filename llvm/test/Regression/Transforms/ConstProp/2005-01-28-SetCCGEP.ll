; RUN: llvm-as < %s | opt -constprop | llvm-dis | not grep 'ret bool false'

%b = external global [2 x {  }] 

implementation

bool %f() {
	; tmp.2 -> true, not false.
	%tmp.2 = seteq {  }* getelementptr ([2 x {  }]* %b, int 0, int 0), 
                             getelementptr ([2 x {  }]* %b, int 0, int 1)
	ret bool %tmp.2
}
