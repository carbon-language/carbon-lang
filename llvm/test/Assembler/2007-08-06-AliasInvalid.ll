; RUN: llvm-as < %s > /dev/null |& grep {Invalid type for reference to global}
; PR1577

@anInt = global i32 1 alias i32 @anAlias
define i32 @main() {
   ret i32 0 
}
