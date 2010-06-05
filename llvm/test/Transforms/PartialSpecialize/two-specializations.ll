; If there are two specializations of a function, make sure each callsite
; calls the right one.
;
; RUN: opt -S -partialspecialization %s | FileCheck %s
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
; CHECK-NEXT: call void @UseCallback1()
; CHECK-NEXT: call void @UseCallback1()
; CHECK-NEXT: call void @UseCallback2()
; CHECK-NEXT: call void @UseCallback(void ()* %pNonConstCallback)
; CHECK-NEXT: call void @UseCallback1()
; CHECK-NEXT: call void @UseCallback2()
; CHECK-NEXT: call void @UseCallback2()
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback2)
  call void @UseCallback(void()* %pNonConstCallback)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback2)
  call void @UseCallback(void()* @callback2)
  ret void
}
