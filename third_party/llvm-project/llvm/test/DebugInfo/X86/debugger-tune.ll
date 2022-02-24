; Verify target-based defaults for "debugger tuning," and the ability to
; override defaults.
; We use the DW_AT_APPLE_optimized attribute and the DW_OP_form_tls_address
; vs. DW_OP_GNU_push_tls_address opcodes to distinguish the debuggers.

; Verify defaults for various targets.
; RUN: llc -mtriple=x86_64-scei-ps4 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=SCE %s
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=LLDB %s
; RUN: llc -mtriple=x86_64-pc-freebsd -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=GDB %s
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=GDB %s

; We can override defaults.
; RUN: llc -mtriple=x86_64-scei-ps4 -filetype=obj -debugger-tune=gdb < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=GDB %s
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -debugger-tune=lldb < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=LLDB %s
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj -debugger-tune=sce < %s | llvm-dwarfdump -debug-info - | FileCheck --check-prefix=SCE %s

; GDB-NOT: DW_AT_APPLE_optimized
; GDB-NOT: DW_OP_form_tls_address

; LLDB: DW_AT_APPLE_optimized
; LLDB: DW_OP_form_tls_address

; SCE-NOT: DW_AT_APPLE_optimized
; SCE-NOT: DW_OP_GNU_push_tls_address

@var = thread_local global i32 0, align 4, !dbg !0

; Function Attrs: norecurse nounwind readnone uwtable
define void @_Z3funv() local_unnamed_addr #0 !dbg !11 {
  ret void, !dbg !14
}

; Function Attrs: norecurse uwtable
define weak_odr hidden i32* @_ZTW3var() local_unnamed_addr #1 {
  ret i32* @var
}

attributes #0 = { norecurse nounwind readnone uwtable }
attributes #1 = { norecurse uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 322268) (llvm/trunk 322267)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "debugger-tune.cpp", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 7.0.0 (trunk 322268) (llvm/trunk 322267)"}
!11 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 2, type: !12, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 2, column: 13, scope: !11)

