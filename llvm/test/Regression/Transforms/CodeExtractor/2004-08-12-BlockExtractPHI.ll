; RUN: llvm-as < %s | opt -extract-blocks -disable-output

implementation

void %l102_yyparse() {
no_exit.0.i:
        br bool false, label %yylex.entry, label %yylex.entry

yylex.entry:
	%tmp.1027 = phi int  [ 0, %no_exit.0.i ], [ 0, %no_exit.0.i ]
	ret void
}
