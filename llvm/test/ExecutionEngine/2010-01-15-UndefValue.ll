; RUN: llvm-as %s -o %t.bc
; RUN: lli -force-interpreter=true %t.bc

define i32 @main() {
       %a = add i32 0, undef
       %b = fadd float 0.0, undef
       %c = fadd double 0.0, undef
       ret i32 0
}
