
; RUN: llvm-as < %s | llc -f -o /dev/null
target triple = "powerpc-apple-darwin9.5"
        %llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }*, i8*, i8* }
@llvm.dbg.variable24 = external constant %llvm.dbg.variable.type                ; <%llvm.dbg.variable.type*> [#uses=1]

declare void @llvm.dbg.declare({ }*, { }*) nounwind

define i32 @isascii(i32 %_c) nounwind {
entry:
        call void @llvm.dbg.declare({ }* null, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable24 to { }*))
        unreachable
}


