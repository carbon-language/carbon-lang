; RUN: llvm-as < %s | opt -globalopt -disable-output &&
; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep CTOR

%llvm.global_ctors = appending global [4 x { int, void ()* }] [ 
  { int, void ()* } { int 65535, void ()* %CTOR1 },
  { int, void ()* } { int 65535, void ()* %CTOR1 },
  { int, void ()* } { int 65535, void ()* %CTOR2 },
  { int, void ()* } { int 2147483647, void ()* null }
]

%G = global int 0

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

bool %accessor() {
	%V = load bool* %CTORGV   ;; constant true
	ret bool %V
}
