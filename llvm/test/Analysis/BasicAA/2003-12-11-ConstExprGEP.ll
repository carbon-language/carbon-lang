; This testcase consists of alias relations which should be completely
; resolvable by basicaa, but require analysis of getelementptr constant exprs.

; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -aa-eval -print-may-aliases -disable-output |& not grep May:

%T = type { uint, [10 x ubyte] }

%G = external global %T

void %test() {
  %D = getelementptr %T* %G, long 0, uint 0
  %E = getelementptr %T* %G, long 0, uint 1, long 5
  %F = getelementptr uint* getelementptr (%T* %G, long 0, uint 0), long 0
  %X = getelementptr [10 x ubyte]* getelementptr (%T* %G, long 0, uint 1), long 0, long 5

  ret void
}
