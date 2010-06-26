; If there are two specializations of a function, make sure each callsite
; calls the right one.
;
; RN: opt -S -partialspecialization %s | FileCheck %s
; RUN: true
declare void @callback1()
declare void @callback2()

define internal void @UseCallback(void()* %pCallback) {
  call void %pCallback()
  ret void
}

define void @foo(void()* %pNonConstCallback)
{
Entry:
; CHECK: Entry
; CHECK-NEXT: call void @callback1()
; CHECK-NEXT: call void @callback1()
; CHECK-NEXT: call void @callback2()
; CHECK-NEXT: call void %pNonConstCallback()
; CHECK-NEXT: call void @callback1()
; CHECK-NEXT: call void @callback2()
; CHECK-NEXT: call void @callback2()
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback2)
  call void @UseCallback(void()* %pNonConstCallback)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback2)
  call void @UseCallback(void()* @callback2)
  ret void
}
