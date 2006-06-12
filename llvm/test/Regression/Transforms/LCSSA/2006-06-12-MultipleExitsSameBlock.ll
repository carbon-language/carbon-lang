; RUN: llvm-as < %s | opt -lcssa | llvm-dis | grep "%X.1.lcssa" &&
; RUN: llvm-as < %s | opt -lcssa | llvm-dis | not grep "%X.1.lcssa1"

declare bool %c1()
declare bool %c2()

int %foo() {
entry:
  br label %loop_begin

loop_begin:
  br bool true, label %loop_body.1, label %loop_exit2

loop_body.1:
  %X.1 = add int 0, 1
  %rel.1 = call bool %c1()
  br bool %rel.1, label %loop_exit, label %loop_body.2
  
loop_body.2:
  %rel.2 = call bool %c2()
  br bool %rel.2, label %loop_exit, label %loop_begin

loop_exit:
  ret int %X.1
  
loop_exit2:
  ret int 1
}
