; RUN: llvm-as < %s | opt -globalopt -disable-output &&
; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep CTOR

%llvm.global_ctors = appending global [2 x { int, void ()* }] [ 
  { int, void ()* } { int 65535, void ()* %CTOR1 },
  { int, void ()* } { int 65535, void ()* %CTOR1 }
]

implementation

internal void %CTOR1() {   ;; noop ctor, remove.
        ret void
}

