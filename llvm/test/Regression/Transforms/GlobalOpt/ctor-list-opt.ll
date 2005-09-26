; RUN: llvm-as < %s | opt -globalopt -disable-output &&
; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep CTOR

%llvm.global_ctors = appending global [6 x { int, void ()* }] [ 
  { int, void ()* } { int 65535, void ()* %CTOR1 },
  { int, void ()* } { int 65535, void ()* %CTOR1 },
  { int, void ()* } { int 65535, void ()* %CTOR2 },
  { int, void ()* } { int 65535, void ()* %CTOR3 },
  { int, void ()* } { int 65535, void ()* %CTOR4 },
  { int, void ()* } { int 2147483647, void ()* null }
]

%G = global int 0
%G2 = global int 0
%G3 = global int -123

%CTORGV = internal global bool false    ;; Should become constant after eval

implementation

internal void %CTOR1() {   ;; noop ctor, remove.
        ret void
}

internal void %CTOR2() {   ;; evaluate the store
	%A = add int 1, 23
	store int %A, int* %G
	store bool true, bool* %CTORGV
        ret void
}

internal void %CTOR3() {
	%X = or bool true, false
	br label %Cont
Cont:
	br bool %X, label %S, label %T
S:
	store int 24, int* %G2
	ret void
T:
	ret void
}

internal void %CTOR4() {
	%X = load int* %G3
	%Y = add int %X, 123
	store int %Y, int* %G3
	ret void
}

bool %accessor() {
	%V = load bool* %CTORGV   ;; constant true
	ret bool %V
}
