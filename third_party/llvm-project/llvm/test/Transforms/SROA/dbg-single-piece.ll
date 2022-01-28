; RUN: opt -sroa %s -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%foo = type { [8 x i8], [8 x i8] }

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0
define void @_ZL18findInsertLocationPN4llvm17MachineBasicBlockENS_9SlotIndexERNS_13LiveIntervalsE() {
entry:
  %retval = alloca %foo, align 8
  call void @llvm.dbg.declare(metadata %foo* %retval, metadata !1, metadata !7), !dbg !8
; Checks that SROA still inserts a bit_piece expression, even if it produces only one piece
; (as long as that piece is smaller than the whole thing)
; CHECK-NOT: call void @llvm.dbg.value
; CHECK: call void @llvm.dbg.value(metadata %foo* undef, {{.*}}, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg
; CHECK-NOT: call void @llvm.dbg.value
  %0 = bitcast %foo* %retval to i8*
  %1 = getelementptr inbounds i8, i8* %0, i64 8
  %2 = bitcast i8* %1 to %foo**
  store %foo* undef, %foo** %2, align 8
  ret void
}

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocalVariable(name: "I", scope: !2, file: !3, line: 947, type: !4)
!2 = distinct !DISubprogram(name: "findInsertLocation", linkageName: "_ZL18findInsertLocationPN4llvm17MachineBasicBlockENS_9SlotIndexERNS_13LiveIntervalsE", scope: !3, file: !3, line: 937, isLocal: true, isDefinition: true, scopeLine: 938, flags: DIFlagPrototyped, isOptimized: true, unit: !9)
!3 = !DIFile(filename: "none", directory: ".")
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "bundle_iterator<llvm::MachineInstr, llvm::ilist_iterator<llvm::MachineInstr> >", scope: !5, file: !3, line: 163, size: 128, align: 64, elements: !6, templateParams: !6, identifier: "_ZTSN4llvm17MachineBasicBlock15bundle_iteratorINS_12MachineInstrENS_14ilist_iteratorIS2_EEEE")
!5 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "MachineBasicBlock", file: !3, line: 68, size: 1408, align: 64, identifier: "_ZTSN4llvm17MachineBasicBlockE")
!6 = !{}
!7 = !DIExpression()
!8 = !DILocation(line: 947, column: 35, scope: !2)
!9 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3)
