; This testcase consists of alias relations which should be completely
; resolvable by basicaa.

; RUN: llvm-as < %s | opt -aa-eval -print-may-aliases 2>&1 -disable-output | not grep May:

%T = type { uint, [10 x ubyte] }

void %test(%T* %P) {
  %A = getelementptr %T* %P, long 0
  %B = getelementptr %T* %P, long 0, ubyte 0
  %C = getelementptr %T* %P, long 0, ubyte 1
  %D = getelementptr %T* %P, long 0, ubyte 1, long 0
  %E = getelementptr %T* %P, long 0, ubyte 1, long 5
  ret void
}
