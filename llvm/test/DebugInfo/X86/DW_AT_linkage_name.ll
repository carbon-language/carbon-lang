; RUN: llc -mtriple=x86_64-apple-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
;
; struct A {
;   A(int i);
;   ~A();
; };
;
; A::~A() {}
;
; void foo() {
;   A a(1);
; }
;
; rdar://problem/16362674
;
; Test that we do not emit a linkage name for the declaration of a destructor.
; Test that we do emit a linkage name for a specific instance of it.

; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}} "~A"
; CHECK-NOT: DW_AT_MIPS_linkage_name
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name {{.*}} "_ZN1AD2Ev"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "~A"


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.A = type { i8 }

; Function Attrs: nounwind ssp uwtable
define void @_ZN1AD2Ev(%struct.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !26, metadata !MDExpression()), !dbg !28
  %this1 = load %struct.A*, %struct.A** %this.addr
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_ZN1AD1Ev(%struct.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !30, metadata !MDExpression()), !dbg !31
  %this1 = load %struct.A*, %struct.A** %this.addr
  call void @_ZN1AD2Ev(%struct.A* %this1), !dbg !32
  ret void, !dbg !33
}

; Function Attrs: ssp uwtable
define void @_Z3foov() #2 {
entry:
  %a = alloca %struct.A, align 1
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !34, metadata !MDExpression()), !dbg !35
  call void @_ZN1AC1Ei(%struct.A* %a, i32 1), !dbg !35
  call void @_ZN1AD1Ev(%struct.A* %a), !dbg !36
  ret void, !dbg !36
}

declare void @_ZN1AC1Ei(%struct.A*, i32)

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.ident = !{!25}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !3, subprograms: !16, globals: !2, imports: !2)
!1 = !MDFile(filename: "linkage-name.cpp", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !MDCompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS1A")
!5 = !{!6, !12}
!6 = !MDSubprogram(name: "A", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !"_ZTS1A", type: !7, variables: !11)
!7 = !MDSubroutineType(types: !8)
!8 = !{null, !9, !10}
!9 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1A")
!10 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{i32 786468}
!12 = !MDSubprogram(name: "~A", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !"_ZTS1A", type: !13, variables: !15)
!13 = !MDSubroutineType(types: !14)
!14 = !{null, !9}
!15 = !{i32 786468}
!16 = !{!17, !18, !19}
!17 = !MDSubprogram(name: "~A", linkageName: "_ZN1AD2Ev", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !1, scope: !"_ZTS1A", type: !13, function: void (%struct.A*)* @_ZN1AD2Ev, declaration: !12, variables: !2)
!18 = !MDSubprogram(name: "~A", linkageName: "_ZN1AD1Ev", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !1, scope: !"_ZTS1A", type: !13, function: void (%struct.A*)* @_ZN1AD1Ev, declaration: !12, variables: !2)
!19 = !MDSubprogram(name: "foo", linkageName: "_Z3foov", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 10, file: !1, scope: !20, type: !21, function: void ()* @_Z3foov, variables: !2)
!20 = !MDFile(filename: "linkage-name.cpp", directory: "")
!21 = !MDSubroutineType(types: !22)
!22 = !{null}
!23 = !{i32 2, !"Dwarf Version", i32 2}
!24 = !{i32 1, !"Debug Info Version", i32 3}
!25 = !{!"clang version 3.5.0 "}
!26 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !17, type: !27)
!27 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS1A")
!28 = !MDLocation(line: 0, scope: !17)
!29 = !MDLocation(line: 8, scope: !17)
!30 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !18, type: !27)
!31 = !MDLocation(line: 0, scope: !18)
!32 = !MDLocation(line: 6, scope: !18)
!33 = !MDLocation(line: 8, scope: !18)
!34 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "a", line: 11, scope: !19, file: !20, type: !"_ZTS1A")
!35 = !MDLocation(line: 11, scope: !19)
!36 = !MDLocation(line: 12, scope: !19)
