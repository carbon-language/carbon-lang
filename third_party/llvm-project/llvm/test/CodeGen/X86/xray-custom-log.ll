; RUN: llc -mtriple=x86_64 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -relocation-model=pic < %s | FileCheck %s --check-prefix=PIC

; RUN: llc -mtriple=x86_64 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s --check-prefix=DBG

define i32 @customevent() nounwind "function-instrument"="xray-always" !dbg !1 {
    %eventptr = alloca i8
    %eventsize = alloca i32
    store i32 3, i32* %eventsize
    %val = load i32, i32* %eventsize
    call void @llvm.xray.customevent(i8* %eventptr, i32 %val), !dbg !8
    ; CHECK-LABEL: Lxray_event_sled_0:
    ; CHECK:       .byte 0xeb, 0x0f
    ; CHECK-NEXT:  pushq %rdi
    ; CHECK-NEXT:  pushq %rsi
    ; CHECK-NEXT:  movq %rcx, %rdi
    ; CHECK-NEXT:  movq %rax, %rsi
    ; CHECK-NEXT:  callq __xray_CustomEvent
    ; CHECK-NEXT:  popq %rsi
    ; CHECK-NEXT:  popq %rdi

    ; PIC-LABEL: Lxray_event_sled_0:
    ; PIC:       .byte 0xeb, 0x0f
    ; PIC-NEXT:  pushq %rdi
    ; PIC-NEXT:  pushq %rsi
    ; PIC-NEXT:  movq %rcx, %rdi
    ; PIC-NEXT:  movq %rax, %rsi
    ; PIC-NEXT:  callq __xray_CustomEvent@PLT
    ; PIC-NEXT:  popq %rsi
    ; PIC-NEXT:  popq %rdi
    ret i32 0
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .quad {{.*}}xray_event_sled_0

define i32 @typedevent() nounwind "function-instrument"="xray-always" !dbg !2 {
    %eventptr = alloca i8
    %eventsize = alloca i32
    %eventtype = alloca i16
    store i16 6, i16* %eventtype
    %type = load i16, i16* %eventtype
    store i32 3, i32* %eventsize
    %val = load i32, i32* %eventsize
    call void @llvm.xray.typedevent(i16 %type, i8* %eventptr, i32 %val), !dbg !9
    ; CHECK-LABEL: Lxray_typed_event_sled_0:
    ; CHECK:       .byte 0xeb, 0x14
    ; CHECK-NEXT:  pushq %rdi
    ; CHECK-NEXT:  pushq %rsi
    ; CHECK-NEXT:  pushq %rdx
    ; CHECK-NEXT:  movq %rdx, %rdi
    ; CHECK-NEXT:  movq %rcx, %rsi
    ; CHECK-NEXT:  movq %rax, %rdx
    ; CHECK-NEXT:  callq __xray_TypedEvent
    ; CHECK-NEXT:  popq %rdx
    ; CHECK-NEXT:  popq %rsi
    ; CHECK-NEXT:  popq %rdi

    ; PIC-LABEL: Lxray_typed_event_sled_0:
    ; PIC:       .byte 0xeb, 0x14
    ; PIC-NEXT:  pushq %rdi
    ; PIC-NEXT:  pushq %rsi
    ; PIC-NEXT:  pushq %rdx
    ; PIC-NEXT:  movq %rdx, %rdi
    ; PIC-NEXT:  movq %rcx, %rsi
    ; PIC-NEXT:  movq %rax, %rdx
    ; PIC-NEXT:  callq __xray_TypedEvent@PLT
    ; PIC-NEXT:  popq %rdx
    ; PIC-NEXT:  popq %rsi
    ; PIC-NEXT:  popq %rdi
    ret i32 0
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start1:
; CHECK:       .quad {{.*}}xray_typed_event_sled_0

declare void @llvm.xray.customevent(i8*, i32)
declare void @llvm.xray.typedevent(i16, i8*, i32)

;; Construct call site entries for PATCHABLE_EVENT_CALL.
; DBG:     DW_TAG_subprogram
; DBG:       DW_TAG_call_site
; DBG-NEXT:    DW_AT_call_target (DW_OP_reg{{.*}})
; DBG-NEXT:    DW_AT_call_return_pc

; DBG:     DW_TAG_subprogram
; DBG:       DW_TAG_call_site
; DBG-NEXT:    DW_AT_call_target (DW_OP_reg{{.*}})
; DBG-NEXT:    DW_AT_call_return_pc

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!10, !11}

!1 = distinct !DISubprogram(name: "customevent", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7)
!2 = distinct !DISubprogram(name: "typedevent", scope: !3, file: !3, line: 3, type: !4, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7)
!3 = !DIFile(filename: "a.c", directory: "/tmp")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!8 = !DILocation(line: 2, column: 3, scope: !1)
!9 = !DILocation(line: 4, column: 3, scope: !2)

!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
