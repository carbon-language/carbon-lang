; RUN: llc -O0 -mtriple=i386-pc-win32 -filetype=asm -o - %s | FileCheck %s

!0 = !{i32 6, !"Linker Options", !{!{!"/DEFAULTLIB:msvcrt.lib"}, !{!"/DEFAULTLIB:msvcrt.lib", !"/DEFAULTLIB:secur32.lib"}, !{!"/DEFAULTLIB:\22C:\5Cpath to\5Casan_rt.lib\22"}, !{!"\22/with spaces\22"}}}

!llvm.module.flags = !{ !0 }

define dllexport void @foo() {
  ret void
}

; CHECK: .section        .drectve,"yn"
; CHECK: .ascii   " /DEFAULTLIB:msvcrt.lib"
; CHECK: .ascii   " /DEFAULTLIB:msvcrt.lib"
; CHECK: .ascii   " /DEFAULTLIB:secur32.lib"
; CHECK: .ascii   " /DEFAULTLIB:\"C:\\path to\\asan_rt.lib\""
; CHECK: .ascii   " \"/with spaces\""
; CHECK: .ascii   " /EXPORT:_foo"
