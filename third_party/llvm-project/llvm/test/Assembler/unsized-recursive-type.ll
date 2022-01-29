; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: base element of getelementptr must be sized

%myTy = type { %myTy }
define void @foo(%myTy* %p){
  %0 = getelementptr %myTy, %myTy* %p, i32 0
  ret void
}
