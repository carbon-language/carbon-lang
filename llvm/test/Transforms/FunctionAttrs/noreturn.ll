; RUN: opt < %s -functionattrs -instcombine -S | FileCheck %s

define void @endless_loop() noreturn nounwind readnone ssp uwtable {
entry:
  br label %while.body

while.body:
  br label %while.body
}
;CHECK-LABEL: @main(
;CHECK: endless_loop
;CHECK: ret
define i32 @main() noreturn nounwind ssp uwtable {
entry:
  tail call void @endless_loop()
  unreachable
}

