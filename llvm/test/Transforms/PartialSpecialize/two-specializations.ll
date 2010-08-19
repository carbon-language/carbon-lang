; If there are two specializations of a function, make sure each callsite
; calls the right one.
;
; RUN: opt -S -partialspecialization -disable-inlining %s | opt -S -inline | FileCheck %s -check-prefix=CORRECT
; RUN: opt -S -partialspecialization -disable-inlining %s | FileCheck %s 
declare void @callback1()
declare void @callback2()

define internal void @UseCallback(void()* %pCallback) {
  call void %pCallback()
  ret void
}

define void @foo(void()* %pNonConstCallback)
{
Entry:
; CORRECT: Entry
; CORRECT-NEXT: call void @callback1()
; CORRECT-NEXT: call void @callback1()
; CORRECT-NEXT: call void @callback2()
; CORRECT-NEXT: call void %pNonConstCallback()
; CORRECT-NEXT: call void @callback1()
; CORRECT-NEXT: call void @callback2()
; CORRECT-NEXT: call void @callback2()
; CHECK: Entry
; CHECK-NOT: call void @UseCallback(void ()* @callback1)
; CHECK-NOT: call void @UseCallback(void ()* @callback2)
; CHECK: ret void
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback2)
  call void @UseCallback(void()* %pNonConstCallback)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback2)
  call void @UseCallback(void()* @callback2)
  ret void
}
