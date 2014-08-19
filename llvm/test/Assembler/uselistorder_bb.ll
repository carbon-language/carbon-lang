; RUN: llvm-as < %s -disable-output 2>&1 | FileCheck %s -allow-empty
; CHECK-NOT: error
; CHECK-NOT: warning
; RUN: verify-uselistorder < %s

@ba1 = constant i8* blockaddress (@bafunc1, %bb)
@ba2 = constant i8* getelementptr (i8* blockaddress (@bafunc2, %bb), i61 0)
@ba3 = constant i8* getelementptr (i8* blockaddress (@bafunc2, %bb), i61 0)

define i8* @babefore() {
  ret i8* getelementptr (i8* blockaddress (@bafunc2, %bb), i61 0)
bb1:
  ret i8* blockaddress (@bafunc1, %bb)
bb2:
  ret i8* blockaddress (@bafunc3, %bb)
}
define void @bafunc1() {
  br label %bb
bb:
  unreachable
}
define void @bafunc2() {
  br label %bb
bb:
  unreachable
}
define void @bafunc3() {
  br label %bb
bb:
  unreachable
}
define i8* @baafter() {
  ret i8* blockaddress (@bafunc2, %bb)
bb1:
  ret i8* blockaddress (@bafunc1, %bb)
bb2:
  ret i8* blockaddress (@bafunc3, %bb)
}

uselistorder_bb @bafunc1, %bb, { 1, 0 }
uselistorder_bb @bafunc2, %bb, { 1, 0 }
uselistorder_bb @bafunc3, %bb, { 1, 0 }
