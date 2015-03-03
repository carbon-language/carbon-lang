; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; RUN: llc -filetype=obj %s -regalloc=basic -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; Test that the type for the formal parameter "var" makes it into the debug info.
; rdar://8950491

;CHECK: DW_TAG_formal_parameter
;CHECK-NEXT: DW_AT_location
;CHECK-NEXT: DW_AT_name {{.*}} "var"
;CHECK-NEXT: DW_AT_decl_file
;CHECK-NEXT: DW_AT_decl_line
;CHECK-NEXT: DW_AT_type

@dfm = external global i32, align 4

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @foo(i32 %dev, i64 %cmd, i8* %data, i32 %data2) nounwind optsize ssp {
entry:
  call void @llvm.dbg.value(metadata i32 %dev, i64 0, metadata !12, metadata !MDExpression()), !dbg !13
  %tmp.i = load i32, i32* @dfm, align 4, !dbg !14
  %cmp.i = icmp eq i32 %tmp.i, 0, !dbg !14
  br i1 %cmp.i, label %if.else, label %if.end.i, !dbg !14

if.end.i:                                         ; preds = %entry
  switch i64 %cmd, label %if.then [
    i64 2147772420, label %bb.i
    i64 536897538, label %bb116.i
  ], !dbg !22

bb.i:                                             ; preds = %if.end.i
  unreachable

bb116.i:                                          ; preds = %if.end.i
  unreachable

if.then:                                          ; preds = %if.end.i
  ret i32 undef, !dbg !23

if.else:                                          ; preds = %entry
  ret i32 0
}

declare hidden fastcc i32 @bar(i32, i32* nocapture) nounwind optsize ssp
declare hidden fastcc i32 @bar2(i32) nounwind optsize ssp
declare hidden fastcc i32 @bar3(i32) nounwind optsize ssp
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!29}

!0 = !MDSubprogram(name: "foo", line: 19510, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 19510, file: !26, scope: !1, type: !3, function: i32 (i32, i64, i8*, i32)* @foo)
!1 = !MDFile(filename: "/tmp/f.c", directory: "/tmp")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 124753)", isOptimized: true, emissionKind: 0, file: !27, enums: !28, retainedTypes: !28, subprograms: !24, imports:  null)
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !MDSubprogram(name: "bar3", line: 14827, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !26, scope: !1, type: !3, function: i32 (i32)* @bar3)
!7 = !MDSubprogram(name: "bar2", line: 15397, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !26, scope: !1, type: !3, function: i32 (i32)* @bar2)
!8 = !MDSubprogram(name: "bar", line: 12382, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !26, scope: !1, type: !9, function: i32 (i32, i32*)* @bar)
!9 = !MDSubroutineType(types: !10)
!10 = !{!11}
!11 = !MDBasicType(tag: DW_TAG_base_type, name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!12 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "var", line: 19509, arg: 0, scope: !0, file: !1, type: !5)
!13 = !MDLocation(line: 19509, column: 20, scope: !0)
!14 = !MDLocation(line: 18091, column: 2, scope: !15, inlinedAt: !17)
!15 = distinct !MDLexicalBlock(line: 18086, column: 1, file: !26, scope: !16)
!16 = !MDSubprogram(name: "foo_bar", line: 18086, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !26, scope: !1, type: !3)
!17 = !MDLocation(line: 19514, column: 2, scope: !18)
!18 = distinct !MDLexicalBlock(line: 19510, column: 1, file: !26, scope: !0)
!22 = !MDLocation(line: 18094, column: 2, scope: !15, inlinedAt: !17)
!23 = !MDLocation(line: 19524, column: 1, scope: !18)
!24 = !{!0, !6, !7, !8, !16}
!25 = !MDFile(filename: "f.i", directory: "/tmp")
!26 = !MDFile(filename: "/tmp/f.c", directory: "/tmp")
!27 = !MDFile(filename: "f.i", directory: "/tmp")
!28 = !{i32 0}
!29 = !{i32 1, !"Debug Info Version", i32 3}
