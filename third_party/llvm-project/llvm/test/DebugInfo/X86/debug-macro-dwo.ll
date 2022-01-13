; This test checks emission of .debug_macro.dwo section when
; -gdwarf-5 -gsplit-dwarf -fdebug-macro is specified.

; RUN: llc -dwarf-version=5 -O0 -filetype=obj \
; RUN: -split-dwarf-file=foo.dwo < %s | llvm-dwarfdump -debug-macro -debug-info -debug-line -v - | FileCheck %s

; CHECK-LABEL:  .debug_info contents:
; CHECK: DW_AT_macros [DW_FORM_sec_offset] (0x00000000)

; CHECK-LABEL:  .debug_macro.dwo contents:
; CHECK-NEXT: 0x00000000:
; CHECK-NEXT: macro header: version = 0x0005, flags = 0x02, format = DWARF32, debug_line_offset = 0x00000000
; CHECK-NEXT: DW_MACRO_start_file - lineno: 0 filenum: 0
; CHECK-NEXT:   DW_MACRO_start_file - lineno: 1 filenum: 1
; CHECK-NEXT:     DW_MACRO_define_strx - lineno: 1 macro: FOO 5
; CHECK-NEXT:   DW_MACRO_end_file
; CHECK-NEXT:   DW_MACRO_start_file - lineno: 2 filenum: 2
; CHECK-NEXT:     DW_MACRO_undef_strx - lineno: 14 macro: YEA
; CHECK-NEXT:   DW_MACRO_end_file
; CHECK-NEXT:   DW_MACRO_undef_strx - lineno: 14 macro: YEA
; CHECK-NEXT: DW_MACRO_end_file

; CHECK-LABEL: .debug_line.dwo contents:
; CHECK: file_names[  0]:
; CHECK:            name: "test.c"
; CHECK: file_names[  1]:
; CHECK:            name: "foo.h"
; CHECK: file_names[  2]:
; CHECK:            name: "bar.h"



; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p200:32:32-p201:32:32-p202:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "test.dwo", emissionKind: FullDebug, enums: !2, macros: !3, globals: !18, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/", checksumkind: CSK_MD5, checksum: "ef6a7032e0c7ceeef614583f2c00dc80")
!2 = !{}
!3 = !{!4}
!4 = !DIMacroFile(file: !1, nodes: !5)
!5 = !{!6, !10, !13}
!6 = !DIMacroFile(line: 1, file: !7, nodes: !8)
!7 = !DIFile(filename: "./foo.h", directory: "/home/", checksumkind: CSK_MD5, checksum: "0f0cd0e15b44f49d3944992c8dc28661")
!8 = !{!9}
!9 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "FOO", value: "5")
!10 = !DIMacroFile(line: 2, file: !11, nodes: !12)
!11 = !DIFile(filename: "./bar.h", directory: "/home/", checksumkind: CSK_MD5, checksum: "bf4b34c263eaaa1d7085c18243b8d100")
!12 = !{!13}
!13 = !DIMacro(type: DW_MACINFO_undef, line: 14, name: "YEA")
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 11.0.0"}
!18 = !{!19}
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "i", scope: !0, file: !21, line: 1, type: !22, isLocal: true, isDefinition: true)
!21 = !DIFile(filename: "./not_used_by_macro.h", directory: "/home/", checksumkind: CSK_MD5, checksum: "cf4b34c263eaaa1d7085c18243b8d101")
!22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
