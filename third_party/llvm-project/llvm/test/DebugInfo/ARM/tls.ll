; RUN: llc -O0 -filetype=asm -mtriple=armv7-linux-gnuehabi < %s \
; RUN:     | FileCheck %s
; RUN: llc -O0 -filetype=asm -mtriple=armv7-linux-gnuehabi -emulated-tls < %s \
; RUN:     | FileCheck %s --check-prefix=EMU

; Generated with clang with source
; __thread int x;

source_filename = "test/DebugInfo/ARM/tls.ll"

@x = thread_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}
; 6 byte of data
; CHECK: .byte 6 @ DW_AT_location
; DW_OP_const4u
; CHECK: .byte 12
; The debug relocation of the address of the tls variable
; CHECK: .long x(tlsldo)

; TODO: Add expected output for -emulated-tls tests.
; EMU-NOT: .long x(tlsldo)

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "tls.c", directory: "/tmp")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 "}

