; RUN: llc -O0 -mtriple=i386-pc-win32 -filetype=asm -o - %s | FileCheck %s

!0 = metadata !{ i32 6, metadata !"Linker Options",
   metadata !{
      metadata !{ metadata !"/DEFAULTLIB:msvcrt.lib" },
      metadata !{ metadata !"/DEFAULTLIB:msvcrt.lib",
                  metadata !"/DEFAULTLIB:secur32.lib" },
      metadata !{ metadata !"/DEFAULTLIB:C:\5Cpath to\5Casan_rt.lib" },
      metadata !{ metadata !"/with spaces" } } }

!llvm.module.flags = !{ !0 }

define dllexport void @foo() {
  ret void
}

; CHECK: .section        .drectve,"yn"
; CHECK: .ascii   " /DEFAULTLIB:msvcrt.lib"
; CHECK: .ascii   " /DEFAULTLIB:msvcrt.lib"
; CHECK: .ascii   " /DEFAULTLIB:secur32.lib"
; CHECK: .ascii   " \"/DEFAULTLIB:C:\\path to\\asan_rt.lib\""
; CHECK: .ascii   " \"/with spaces\""
; CHECK: .ascii   " /EXPORT:_foo"
