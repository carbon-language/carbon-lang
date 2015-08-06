; RUN: llvm-as -o %T/lto-linker-opts.obj %s
; RUN: env LIB=%S/Inputs lld-link /out:%T/lto-linker-opts.exe /entry:main /subsystem:console %T/lto-linker-opts.obj

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

!llvm.module.flags = !{!0}

!0 = !{i32 6, !"Linker Options", !1}
!1 = !{!2}
!2 = !{!"/DEFAULTLIB:ret42.lib"}
