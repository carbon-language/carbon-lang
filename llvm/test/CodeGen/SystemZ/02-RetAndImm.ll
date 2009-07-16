; RUN: llvm-as < %s | llc
define i64 @foo(i64 %a, i64 %b) {
entry:
    %c = and i64 %a, 1
    ret i64 %c
}

; FIXME: SystemZ has 4 and reg-imm instructions depending on imm,
; we need to support them someday.