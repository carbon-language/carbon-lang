; RUN: llvm-as < %s | opt -extract-blocks -disable-output

implementation

void %test1() {
no_exit.0.i:
        br bool false, label %yylex.entry, label %yylex.entry

yylex.entry:
	%tmp.1027 = phi int  [ 0, %no_exit.0.i ], [ 0, %no_exit.0.i ]
	ret void
}

void %test2() {
no_exit.0.i:
        switch uint 0, label %yylex.entry [
            uint 0, label %yylex.entry
            uint 1, label %foo
        ]

yylex.entry:
        %tmp.1027 = phi int  [ 0, %no_exit.0.i ], [ 0, %no_exit.0.i ]
        ret void
foo:
        ret void
}

