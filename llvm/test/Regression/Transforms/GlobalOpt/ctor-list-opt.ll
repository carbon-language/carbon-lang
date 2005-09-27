; RUN: llvm-as < %s | opt -globalopt -disable-output &&
; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep CTOR

%llvm.global_ctors = appending global [10 x { int, void ()* }] [ 
  { int, void ()* } { int 65535, void ()* %CTOR1 },
  { int, void ()* } { int 65535, void ()* %CTOR1 },
  { int, void ()* } { int 65535, void ()* %CTOR2 },
  { int, void ()* } { int 65535, void ()* %CTOR3 },
  { int, void ()* } { int 65535, void ()* %CTOR4 },
  { int, void ()* } { int 65535, void ()* %CTOR5 },
  { int, void ()* } { int 65535, void ()* %CTOR6 },
  { int, void ()* } { int 65535, void ()* %CTOR7 },
  { int, void ()* } { int 65535, void ()* %CTOR8 },
  { int, void ()* } { int 2147483647, void ()* null }
]

%G = global int 0
%G2 = global int 0
%G3 = global int -123
%X = global {int, [2 x int]} { int 0, [2 x int] [ int 17, int 21] }
%Y = global int -1
%Z = global int 123
%D = global double 0.0

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

internal void %CTOR5() {
	%X.2p = getelementptr {int,[2 x int]}* %X, int 0, uint 1, int 0
	%X.2 = load int* %X.2p
	%X.1p = getelementptr {int,[2 x int]}* %X, int 0, uint 0
	store int %X.2, int* %X.1p
	store int 42, int* %X.2p
	ret void
}

internal void %CTOR6() {
	%A = alloca int
	%y = load int* %Y
	store int %y, int* %A
	%Av = load int* %A
	%Av1 = add int %Av, 1
	store int %Av1, int* %Y
	ret void
}

internal void %CTOR7() {
	call void %setto(int* %Z, int 0)
	ret void
}

void %setto(int* %P, int %V) {
	store int %V, int* %P
	ret void
}

declare double %cos(double)

internal void %CTOR8() {
	%X = call double %cos(double 1.0)
	store double %X, double* %D
	ret void
}
bool %accessor() {
	%V = load bool* %CTORGV   ;; constant true
	ret bool %V
}
