; RUN: llc -mtriple=i686-windows-msvc -O1 < %s | FileCheck %s
; RUN: llc -mtriple=i686-windows-msvc -O0 < %s | FileCheck %s

; The MSVC family of x86 calling conventions makes tail calls really tricky.
; Tests of all the various combinations should live here.

declare i32 @cdecl_i32()
declare void @cdecl_void()

; Don't allow tail calling these cdecl functions, because we need to clear the
; incoming stack arguments for these argument-clearing conventions.

define x86_thiscallcc void @thiscall_cdecl_notail(i32 %a, i32 %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: thiscall_cdecl_notail: # @thiscall_cdecl_notail
; CHECK: calll _cdecl_void
; CHECK: retl $8
define x86_stdcallcc void @stdcall_cdecl_notail(i32 %a, i32 %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: _stdcall_cdecl_notail@12: # @stdcall_cdecl_notail
; CHECK: calll _cdecl_void
; CHECK: retl $12
define x86_vectorcallcc void @vectorcall_cdecl_notail(i32 inreg %a, i32 inreg %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: vectorcall_cdecl_notail@@12: # @vectorcall_cdecl_notail
; CHECK: calll _cdecl_void
; CHECK: retl $4
define x86_fastcallcc void @fastcall_cdecl_notail(i32 inreg %a, i32 inreg %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: @fastcall_cdecl_notail@12: # @fastcall_cdecl_notail
; CHECK: calll _cdecl_void
; CHECK: retl $4
