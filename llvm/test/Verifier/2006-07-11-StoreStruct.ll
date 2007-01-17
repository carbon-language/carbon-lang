; RUN: not llvm-as %s -o /dev/null -f
; PR826

        %struct_4 = type { int }

implementation   ; Functions:

void %test() {
        store %struct_4 zeroinitializer, %struct_4* null
        unreachable
}
