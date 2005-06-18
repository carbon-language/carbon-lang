; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

implementation   ; Functions:
declare void %llvm.va_start(sbyte**)
declare void %llvm.va_copy(sbyte**, sbyte*)
declare void %llvm.va_end(sbyte**)

int %test(int %X, ...) {
        ; Allocate two va_list items.  On this target, va_list is of type sbyte*
        %ap = alloca sbyte*             ; <sbyte**> [#uses=4]
        %aq = alloca sbyte*             ; <sbyte**> [#uses=2]

        ; Initialize variable argument processing
        call void %llvm.va_start(sbyte** %ap)

        ; Read a single integer argument
        %tmp = vaarg sbyte** %ap, int           ; <int> [#uses=1]

        ; Demonstrate usage of llvm.va_copy and llvm_va_end
        %apv = load sbyte** %ap         ; <sbyte*> [#uses=1]
        call void %llvm.va_copy(sbyte** %aq, sbyte* %apv)
        call void %llvm.va_end(sbyte** %aq)

        ; Stop processing of arguments.
        call void %llvm.va_end(sbyte** %ap)
        ret int %tmp

}
