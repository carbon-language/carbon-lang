; This testcase, which was distilled from a HUGE function, causes problems
; because both the source and the destination of the %Y cast are converted
; to a new type, which causes massive problems.

; RUN: llvm-as < %s | opt -raise -raise-start-inst=W

int **%test(sbyte **%S) {
BB0:
   br label %Loop

Loop:
   %X = phi sbyte* [null , %BB0], [%Z, %Loop]

   %Y = cast sbyte *%X to sbyte **
   %Z = load sbyte** %Y
   br bool true, label %Loop, label %Out

Out:
  %W = cast sbyte** %Y to int**
  ret int** %W
}
