; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha

target datalayout = "e-p:64:64"
target endian = little
target pointersize = 64
target triple = "alphaev67-unknown-linux-gnu"
        %struct.va_list = type { sbyte*, int, int }

implementation   ; Functions:

void %yyerror(int, ...) {
entry:
        call void %llvm.va_start( %struct.va_list* null )
        ret void
}

declare void %llvm.va_start(%struct.va_list*)

