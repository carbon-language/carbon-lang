; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare i8* @llvm.frameallocate(i32)
declare i8* @llvm.framerecover(i8*, i8*)

define internal void @f() {
  call i8* @llvm.frameallocate(i32 4)
  call i8* @llvm.frameallocate(i32 4)
  ret void
}
; CHECK: multiple calls to llvm.frameallocate in one function

define internal void @f_a(i32 %n) {
  call i8* @llvm.frameallocate(i32 %n)
  ret void
}
; CHECK: llvm.frameallocate argument must be constant integer size

define internal void @g() {
entry:
  br label %not_entry
not_entry:
  call i8* @llvm.frameallocate(i32 4)
  ret void
}
; CHECK: llvm.frameallocate used outside of entry block

define internal void @h() {
  call i8* @llvm.framerecover(i8* null, i8* null)
  ret void
}
; CHECK: llvm.framerecover first argument must be function defined in this module

@global = constant i8 0

declare void @declaration()

define internal void @i() {
  call i8* @llvm.framerecover(i8* @global, i8* null)
  ret void
}
; CHECK: llvm.framerecover first argument must be function defined in this module

define internal void @j() {
  call i8* @llvm.framerecover(i8* bitcast(void()* @declaration to i8*), i8* null)
  ret void
}
; CHECK: llvm.framerecover first argument must be function defined in this module
