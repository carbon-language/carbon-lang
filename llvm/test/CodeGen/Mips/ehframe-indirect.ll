; RUN: llc -mtriple=mipsel-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-linux-android < %s | FileCheck %s

define i32 @main() {
; CHECK: .cfi_startproc
; CHECK: .cfi_personality 128, DW.ref.__gxx_personality_v0

entry:
  invoke void @foo() to label %cont unwind label %lpad
; CHECK: foo
; CHECK: jalr

lpad:
  %0 = landingpad { i8*, i32 } personality i8*
    bitcast (i32 (...)* @__gxx_personality_v0 to i8*) catch i8* null
  ret i32 0

cont:
  ret i32 0
}
; CHECK: .cfi_endproc

declare i32 @__gxx_personality_v0(...)

declare void @foo()

; CHECK: .hidden DW.ref.__gxx_personality_v0
; CHECK: .weak DW.ref.__gxx_personality_v0
; CHECK: .section .data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
; CHECK: .align 2
; CHECK: .type DW.ref.__gxx_personality_v0,@object
; CHECK: .size DW.ref.__gxx_personality_v0, 4
; CHECK: DW.ref.__gxx_personality_v0:
; CHECK: .4byte __gxx_personality_v0
