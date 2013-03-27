target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-freebsd10.0"
; RUN: llc -O0 < %s -march=ppc64 | FileCheck -check-prefix=OPT0 %s
; RUN: llc -O1 < %s -march=ppc64 | FileCheck -check-prefix=OPT1 %s

@a = thread_local global i32 0, align 4

;OPT0:          localexec:
;OPT1:          localexec:
define i32 @localexec() nounwind {
entry:
;OPT0:          addis [[REG1:[0-9]+]], 13, a@tprel@ha
;OPT0-NEXT:     li [[REG2:[0-9]+]], 42
;OPT0-NEXT:     addi [[REG1]], [[REG1]], a@tprel@l
;OPT0:          stw [[REG2]], 0([[REG1]])
;OPT1:          addis [[REG1:[0-9]+]], 13, a@tprel@ha
;OPT1-NEXT:     li [[REG2:[0-9]+]], 42
;OPT1-NEXT:     stw [[REG2]], a@tprel@l([[REG1]])
  store i32 42, i32* @a, align 4
  ret i32 0
}
