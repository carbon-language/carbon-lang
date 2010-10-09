; If there are not enough callsites for a particular specialization to
; justify its existence, the specialization shouldn't be created.
;
; RUN: opt -S -partialspecialization -disable-inlining %s | FileCheck %s 
declare void @callback1()
declare void @callback2()

declare void @othercall()

define internal void @UseCallback(void()* %pCallback) {
  call void %pCallback()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  call void @othercall()
  ret void
}

define void @foo(void()* %pNonConstCallback)
{
Entry:
; CHECK: Entry
; CHECK-NOT: call void @UseCallback(void ()* @callback1)
; CHECK: call void @UseCallback(void ()* @callback2)
; CHECK-NEXT: call void @UseCallback(void ()* @callback2)
; CHECK-NEXT: ret void
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback1)
  call void @UseCallback(void()* @callback2)
  call void @UseCallback(void()* @callback2)

  ret void
}
