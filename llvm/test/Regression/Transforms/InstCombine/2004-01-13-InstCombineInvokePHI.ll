; Test for a problem afflicting several C++ programs in the testsuite.  The 
; instcombine pass is trying to get rid of the cast in the invoke instruction, 
; inserting a cast of the return value after the PHI instruction, but which is
; used by the PHI instruction.  This is bad: because of the semantics of the
; invoke instruction, we really cannot perform this transformation at all at
; least without splitting the critical edge.
;
; RUN: llvm-as < %s | opt -instcombine -disable-output

declare sbyte* %test()

int %foo() {
entry:
  br bool true, label %cont, label %call
call:
  %P = invoke int*()* cast (sbyte*()* %test to int*()*)()
       to label %cont except label %N
cont:
  %P2 = phi int* [%P, %call], [null, %entry]
  %V = load int* %P2
  ret int %V
N:
  ret int 0
}
