; MD5 checksums provided by IR should be passed through to asm.
; They'll be emitted to an object file only for DWARF 5 or later.

; REQUIRES: object-emission
; RUN: %llc_dwarf -dwarf-version 4 -filetype=asm -o - %s | FileCheck %s --check-prefix=ASM-4
; RUN: %llc_dwarf -dwarf-version 4 -filetype=asm -dwarf-directory=0 < %s | FileCheck %s --check-prefix=ASM-4-GAS
; RUN: %llc_dwarf -dwarf-version 5 -filetype=asm -o - %s | FileCheck %s --check-prefix=ASM-5
; RUN: %llc_dwarf -dwarf-version 4 -filetype=obj -o %t4.o %s
; RUN: llvm-dwarfdump -debug-line %t4.o | FileCheck %s --check-prefix=OBJ
; RUN: %llc_dwarf -dwarf-version 5 -filetype=obj -o %t5.o %s
; RUN: llvm-dwarfdump -debug-line %t5.o | FileCheck %s --check-prefixes=OBJ,OBJ-5

; ASM-4-GAS: .file 1 "/scratch{{/|\\\\}}t1.h"
; ASM-4-GAS: .file 2 "/scratch{{/|\\\\}}t2.h"

; ASM-4-NOT: .file 0
; ASM-4: .file 1 "/scratch" "t1.h"
; ASM-4-NOT:  md5
; ASM-4: .file 2 "/scratch" "t2.h"
; ASM-4-NOT:  md5
; ASM-5: .file 0 "/scratch" "t.c" md5 0x00000000000000000000000000000000
; ASM-5: .file 1 "t1.h" md5 0x11111111111111111111111111111111
; ASM-5: .file 2 "t2.h" md5 0x22222222222222222222222222222222

; OBJ-5: file_names[ 0]:
; OBJ-5-NEXT: name: "t.c"
; OBJ-5-NEXT: dir_index: 0
; OBJ-5-NEXT: md5_checksum: 00000000000000000000000000000000
; OBJ: file_names[ 1]:
; OBJ-NEXT: name: "t1.h"
; OBJ-NEXT: dir_index: 0
; OBJ-5-NEXT: md5_checksum: 11111111111111111111111111111111
; OBJ: file_names[ 2]:
; OBJ-NEXT: name: "t2.h"
; OBJ-NEXT: dir_index: 0
; OBJ-5-NEXT: md5_checksum: 22222222222222222222222222222222

; ModuleID = 't.c'
source_filename = "t.c"

@t1 = global i32 1, align 4, !dbg !0
@t2 = global i32 0, align 4, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "t1", scope: !2, file: !10, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 322159)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.c", directory: "/scratch", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "t2", scope: !2, file: !8, line: 1, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "t2.h", directory: "/scratch", checksumkind: CSK_MD5, checksum: "22222222222222222222222222222222")
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIFile(filename: "t1.h", directory: "/scratch", checksumkind: CSK_MD5, checksum: "11111111111111111111111111111111")
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 7.0.0 (trunk 322159)"}
