; RUN: llc < %s -mcpu=generic -mtriple=i386-apple-darwin -no-integrated-as

@g1 = global double 0.000000e+00, align 8
@g2 = global i32 0, align 4

define void @_Z16fpuop_arithmeticjj(i32, i32) {
entry:
  switch i32 undef, label %sw.bb.i1921 [
  ]

sw.bb261:                                         ; preds = %entry, %entry
  unreachable

sw.bb.i1921:                                      ; preds = %if.end504
  switch i32 undef, label %if.end511 [
    i32 1, label %sw.bb27.i
  ]

sw.bb27.i:                                        ; preds = %sw.bb.i1921
  %conv.i.i1923 = fpext float undef to x86_fp80
  br label %if.end511

if.end511:                                        ; preds = %sw.bb27.i, %sw.bb13.i
  %src.sroa.0.0.src.sroa.0.0.2280 = phi x86_fp80 [ %conv.i.i1923, %sw.bb27.i ], [ undef, %sw.bb.i1921 ]
  switch i32 undef, label %sw.bb992 [
    i32 3, label %sw.bb735
    i32 18, label %if.end41.i2210
  ]

sw.bb735:                                         ; preds = %if.end511
  %2 = call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 %src.sroa.0.0.src.sroa.0.0.2280)
  unreachable

if.end41.i2210:                                   ; preds = %if.end511
  call void @llvm.dbg.value(metadata x86_fp80 %src.sroa.0.0.src.sroa.0.0.2280, i64 0, metadata !20, metadata !MDExpression())
  unreachable

sw.bb992:                                         ; preds = %if.end511
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25}
!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (http://llvm.org/git/clang 8444ae7cfeaefae031f8fedf0d1435ca3b14d90b) (http://llvm.org/git/llvm 886f0101a7d176543b831f5efb74c03427244a55)", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !21, imports: !2)
!1 = !MDFile(filename: "fpu_ieee.cpp", directory: "x87stackifier")
!2 = !{}
!3 = !{!4}
!4 = !MDSubprogram(name: "fpuop_arithmetic", linkageName: "_Z16fpuop_arithmeticjj", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 13, file: !5, scope: !6, type: !7, function: void (i32, i32)* @_Z16fpuop_arithmeticjj, variables: !10)
!5 = !MDFile(filename: "f1.cpp", directory: "x87stackifier")
!6 = !MDFile(filename: "f1.cpp", directory: "x87stackifier")
!7 = !MDSubroutineType(types: !8)
!8 = !{null, !9, !9}
!9 = !MDBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!10 = !{!11, !12, !13, !18, !20}
!11 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "", line: 11, arg: 1, scope: !4, file: !6, type: !9)
!12 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "", line: 11, arg: 2, scope: !4, file: !6, type: !9)
!13 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "x", line: 14, scope: !4, file: !6, type: !14)
!14 = !MDDerivedType(tag: DW_TAG_typedef, name: "fpu_extended", line: 3, file: !5, baseType: !15)
!15 = !MDDerivedType(tag: DW_TAG_typedef, name: "fpu_register", line: 2, file: !5, baseType: !16)
!16 = !MDDerivedType(tag: DW_TAG_typedef, name: "uae_f64", line: 1, file: !5, baseType: !17)
!17 = !MDBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!18 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "a", line: 15, scope: !4, file: !6, type: !19)
!19 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!20 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "value", line: 16, scope: !4, file: !6, type: !14)
!21 = !{!22, !23}
!22 = !MDGlobalVariable(name: "g1", line: 5, isLocal: false, isDefinition: true, scope: null, file: !6, type: !14, variable: double* @g1)
!23 = !MDGlobalVariable(name: "g2", line: 6, isLocal: false, isDefinition: true, scope: null, file: !6, type: !19, variable: i32* @g2)
!24 = !{i32 2, !"Dwarf Version", i32 2}
!25 = !{i32 2, !"Debug Info Version", i32 3}
