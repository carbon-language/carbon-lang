; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -verify-machineinstrs -filetype=asm | FileCheck %s -check-prefix=ARM-linux
; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -filetype=obj

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

define void @test_basic() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
	ret void

; ARM-linux:      test_basic:

; ARM-linux:      push    {r4, r5}
; ARM-linux:      .cfi_def_cfa_offset 8
; ARM-linux:      .cfi_offset r5, -4
; ARM-linux:      .cfi_offset r4, -8
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: mov     r5, sp
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB0_2

; ARM-linux:      mov     r4, #48
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux:      .cfi_def_cfa_offset 12
; ARM-linux:      .cfi_offset lr, -12
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux:      .cfi_def_cfa_offset 0
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}
; ARM-linux:      .cfi_def_cfa_offset 0
; ARM-linux       .cfi_same_value r4
; ARM-linux       .cfi_same_value r5
}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !MDFile(filename: "var.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !MDSubprogram(name: "test_basic", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !1, scope: !5, type: !6, function: void ()* @test_basic, variables: !2)
!5 = !MDFile(filename: "var.c", directory: "/tmp")
!6 = !MDSubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.5 "}
!12 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "count", line: 5, arg: 1, scope: !4, file: !5, type: !8)
!13 = !MDLocation(line: 5, scope: !4)
!14 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "vl", line: 6, scope: !4, file: !5, type: !15)
!15 = !MDDerivedType(tag: DW_TAG_typedef, name: "va_list", line: 30, file: !16, baseType: !17)
!16 = !MDFile(filename: "/linux-x86_64-high/gcc_4.7.2/dbg/llvm/bin/../lib/clang/3.5/include/stdarg.h", directory: "/tmp")
!17 = !MDDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", line: 6, file: !1, baseType: !18)
!18 = !MDCompositeType(tag: DW_TAG_structure_type, name: "__va_list", line: 6, size: 32, align: 32, file: !1, elements: !19)
!19 = !{!20}
!20 = !MDDerivedType(tag: DW_TAG_member, name: "__ap", line: 6, size: 32, align: 32, file: !1, scope: !18, baseType: !21)
!21 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, baseType: null)
!22 = !MDLocation(line: 6, scope: !4)
!23 = !MDLocation(line: 7, scope: !4)
!24 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "test_basic", line: 8, scope: !4, file: !5, type: !8)
!25 = !MDLocation(line: 8, scope: !4)
!26 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "i", line: 9, scope: !27, file: !5, type: !8)
!27 = distinct !MDLexicalBlock(line: 9, column: 0, file: !1, scope: !4)
!28 = !MDLocation(line: 9, scope: !27)
!29 = !MDLocation(line: 10, scope: !30)
!30 = distinct !MDLexicalBlock(line: 9, column: 0, file: !1, scope: !27)
!31 = !MDLocation(line: 11, scope: !30)
!32 = !MDLocation(line: 12, scope: !4)
!33 = !MDLocation(line: 13, scope: !4)

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

attributes #0 = { "split-stack" }
