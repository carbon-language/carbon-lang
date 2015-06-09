; REQUIRES: object-emission
; RUN: %llc_dwarf -mtriple x86_64-apple-darwin14.0.0-elf -filetype=obj %s -o %t
; RUN: %llc_dwarf -mtriple x86_64-apple-darwin14.0.0-elf -O0 -filetype=obj %s -o %t0
; RUN: llvm-dwarfdump -debug-dump=loc %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-dump=loc %t0 | FileCheck -check-prefix CHECK-O0 %s

; CHECK: 0x00000000: Beginning address offset: 0x0000000000000000
; CHECK:                Ending address offset: 0x[[END:[0-9a-f]+]]
; CHECK:                 Location description:
; CHECK-NOT:                                   75 00 55
; CHECK-SAME:                                  55
; CHECK: 0x00000023: Beginning address offset: 0x0000000000000000
; CHECK:                Ending address offset: 0x{{.*}}[[END]]
; CHECK:                 Location description: 75 08 9f
; CHECK: 0x00000048: Beginning address offset: 0x0000000000000000
; CHECK:                Ending address offset: 0x{{.*}}[[END]]
; CHECK:                 Location description: 75 10 9f
; CHECK: 0x0000006d: Beginning address offset: 0x0000000000000000
; CHECK:                Ending address offset: 0x{{.*}}[[END]]
; CHECK:                 Location description: 75 18


; CHECK-O0: 0x00000000: Beginning address offset: 0x0000000000000000
; CHECK-O0:                Ending address offset: 0x000000000000001b
; CHECK-O0:                 Location description: 55
; CHECK-O0:             Beginning address offset: 0x000000000000001b
; CHECK-O0:                Ending address offset: 0x0000000000000024
; CHECK-O0:                 Location description: 54
; CHECK-O0:             Beginning address offset: 0x0000000000000024
; CHECK-O0:                Ending address offset: 0x0000000000000025
; CHECK-O0:                 Location description: 77 78 23 00
; CHECK-O0: 0x0000004c: Beginning address offset: 0x0000000000000000
; CHECK-O0:                Ending address offset: 0x000000000000001b
; CHECK-O0:                 Location description: 75 08 9f
; CHECK-O0:             Beginning address offset: 0x000000000000001b
; CHECK-O0:                Ending address offset: 0x0000000000000024
; CHECK-O0:                 Location description: 74 08 9f
; CHECK-O0:             Beginning address offset: 0x0000000000000024
; CHECK-O0:                Ending address offset: 0x0000000000000025
; CHECK-O0:                 Location description: 77 78 23 08
; CHECK-O0: 0x0000009c: Beginning address offset: 0x0000000000000000
; CHECK-O0:                Ending address offset: 0x000000000000001b
; CHECK-O0:                 Location description: 75 10 9f
; CHECK-O0:             Beginning address offset: 0x000000000000001b
; CHECK-O0:                Ending address offset: 0x0000000000000024
; CHECK-O0:                 Location description: 74 10 9f
; CHECK-O0:             Beginning address offset: 0x0000000000000024
; CHECK-O0:                Ending address offset: 0x0000000000000025
; CHECK-O0:                 Location description: 77 78 23 08 23 08
; CHECK-O0: 0x000000ee: Beginning address offset: 0x0000000000000000
; CHECK-O0:                Ending address offset: 0x000000000000001b
; CHECK-O0:                 Location description: 75 18
; CHECK-O0:             Beginning address offset: 0x000000000000001b
; CHECK-O0:                Ending address offset: 0x0000000000000024
; CHECK-O0:                 Location description: 74 18
; CHECK-O0:             Beginning address offset: 0x0000000000000024
; CHECK-O0:                Ending address offset: 0x0000000000000025
; CHECK-O0:                 Location description: 77 78 23 10 23 08 06

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

define float @foo(float* %args, float *%args2)
{
    call void @llvm.dbg.value(metadata float* %args, i64 0, metadata !11, metadata !12), !dbg !19
    call void @llvm.dbg.value(metadata float* %args, i64 0, metadata !13, metadata !14), !dbg !19
    call void @llvm.dbg.value(metadata float* %args, i64 0, metadata !15, metadata !16), !dbg !19
    call void @llvm.dbg.value(metadata float* %args, i64 0, metadata !17, metadata !18), !dbg !19
    %a = load float, float* %args, !dbg !19
    %bptr = getelementptr float, float* %args, i32 1, !dbg !19
    %b = load float, float* %bptr, !dbg !19
    %cptr = getelementptr float, float* %args, i32 2, !dbg !19
    %c = load float, float* %cptr, !dbg !19
    %dptr = getelementptr float, float* %args, i32 3, !dbg !19
    %d = load float, float* %dptr, !dbg !19
    %ret1 = fadd float %a, %b, !dbg !19
    %ret2 = fadd float %c, %d, !dbg !19
    call void @llvm.dbg.value(metadata float* %args2, i64 0, metadata !11, metadata !12), !dbg !19
    call void @llvm.dbg.value(metadata float* %args2, i64 0, metadata !13, metadata !14), !dbg !19
    call void @llvm.dbg.value(metadata float* %args2, i64 0, metadata !15, metadata !16), !dbg !19
    call void @llvm.dbg.value(metadata float* %args2, i64 0, metadata !17, metadata !18), !dbg !19
    %ret  = fsub float %ret1, %ret2, !dbg !19
    ret float %ret, !dbg !19
}

attributes #0 = { nounwind readnone }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}

!2 = !DICompileUnit(language: DW_LANG_C89, file: !3, producer: "byHand", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !4, retainedTypes: !4, subprograms: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "expressions", directory: ".")
!4 = !{}
!5 = !{!6}
!6 = !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !3, type: !7, isLocal: false, isDefinition: true, isOptimized: true, function: float (float*, float*)* @foo, variables: !4)
!7 = !DISubroutineType(types: !8)
!8 = !{!10, !10}
!9 = !DIBasicType(name: "float", size: 4, align: 4, encoding: DW_ATE_float)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!11 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "a", arg: 1, scope: !6, file: !3, line: 1, type: !10)
!12 = !DIExpression(DW_OP_plus, 0)
!13 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "b", arg: 2, scope: !6, file: !3, line: 1, type: !10)
!14 = !DIExpression(DW_OP_plus, 8)
!15 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "c", arg: 3, scope: !6, file: !3, line: 1, type: !10)
!16 = !DIExpression(DW_OP_plus, 8, DW_OP_plus, 8)
!17 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "d", arg: 4, scope: !6, file: !3, line: 1, type: !9)
!18 = !DIExpression(DW_OP_plus, 16, DW_OP_plus, 8, DW_OP_deref)
!19 = !DILocation(line: 1, scope: !6)
