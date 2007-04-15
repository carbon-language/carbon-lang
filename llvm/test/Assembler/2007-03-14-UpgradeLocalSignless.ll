; PR1256
; RUN: llvm-upgrade < %s | grep {call void @f( i32 .tmp )}
; RUN: llvm-upgrade < %s | grep {call void @g( i8 .tmp\.upgrd\.2 )}

target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

implementation   ; Functions:

void %_Z4func() {
entry:
        %tmp = add int 0, 0
        %tmp = add uint 1, 1
        %tmp = add ubyte 1, 2
        %tmp = add sbyte 2, 3
        call void %f (int %tmp)
        call void %g (ubyte %tmp)
        ret void
}

declare void %f(int)
declare void %g(ubyte)
