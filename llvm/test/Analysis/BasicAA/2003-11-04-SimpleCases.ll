; This testcase consists of alias relations which should be completely
; resolvable by basicaa.

; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -aa-eval -print-may-aliases -disable-output |& not grep May:

%T = type { uint, [10 x ubyte] }

void %test(%T* %P) {
  %A = getelementptr %T* %P, long 0
  %B = getelementptr %T* %P, long 0, uint 0
  %C = getelementptr %T* %P, long 0, uint 1
  %D = getelementptr %T* %P, long 0, uint 1, long 0
  %E = getelementptr %T* %P, long 0, uint 1, long 5
  ret void
}
