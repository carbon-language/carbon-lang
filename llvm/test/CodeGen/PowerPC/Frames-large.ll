; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 > %t
; RUN: not grep {stw r31, 20(r1)} %t
; RUN: grep {lis r0, -1} %t
; RUN: grep {ori r0, r0, 32704} %t
; RUN: grep {stwux r1, r1, r0} %t
; RUN: grep {lwz r1, 0(r1)} %t
; RUN: not grep {lwz r31, 20(r1)} %t
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim > %t
; RUN: grep {stw r31, 20(r1)} %t
; RUN: grep {lis r0, -1} %t
; RUN: grep {ori r0, r0, 32704} %t
; RUN: grep {stwux r1, r1, r0} %t
; RUN: grep {lwz r1, 0(r1)} %t
; RUN: grep {lwz r31, 20(r1)} %t
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 > %t
; RUN: not grep {std r31, 40(r1)} %t
; RUN: grep {lis r0, -1} %t
; RUN: grep {ori r0, r0, 32656} %t
; RUN: grep {stdux r1, r1, r0} %t
; RUN: grep {ld r1, 0(r1)} %t
; RUN: not grep {ld r31, 40(r1)} %t
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim > %t
; RUN: grep {std r31, 40(r1)} %t
; RUN: grep {lis r0, -1} %t
; RUN: grep {ori r0, r0, 32656} %t
; RUN: grep {stdux r1, r1, r0} %t
; RUN: grep {ld r1, 0(r1)} %t
; RUN: grep {ld r31, 40(r1)} %t

define i32* @f1() {
        %tmp = alloca i32, i32 8191             ; <i32*> [#uses=1]
        ret i32* %tmp
}

