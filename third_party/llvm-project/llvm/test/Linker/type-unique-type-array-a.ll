; REQUIRES: default_triple, object-emission
;
; RUN: llvm-link %s %p/type-unique-type-array-b.ll -S -o - | %llc_dwarf -filetype=obj -O0 | llvm-dwarfdump -v -debug-info - | FileCheck %s
;
; rdar://problem/17628609
;
; cat -n a.cpp
;     1	struct SA {
;     2	  int a;
;     3	};
;     4	
;     5	class A {
;     6	public:
;     7	  void testA(SA a) {
;     8	  }
;     9	};
;    10	
;    11	void topA(A *a, SA sa) {
;    12	  a->testA(sa);
;    13	}
;
; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_class_type
; CHECK-NEXT:   DW_AT_name {{.*}} "A"
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}} "testA"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + 0x{{.*}} => {0x[[STRUCT:.*]]}
; CHECK: 0x[[STRUCT]]: DW_TAG_structure_type
; CHECK-NEXT:   DW_AT_name {{.*}} "SA"

; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_class_type
; CHECK-NEXT:   DW_AT_name {{.*}} "B"
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}} "testB"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[STRUCT]]

%class.A = type { i8 }
%struct.SA = type { i32 }

; Function Attrs: ssp uwtable
define void @_Z4topAP1A2SA(%class.A* %a, i32 %sa.coerce) #0 !dbg !15 {
entry:
  %sa = alloca %struct.SA, align 4
  %a.addr = alloca %class.A*, align 8
  %agg.tmp = alloca %struct.SA, align 4
  %coerce.dive = getelementptr %struct.SA, %struct.SA* %sa, i32 0, i32 0
  store i32 %sa.coerce, i32* %coerce.dive
  store %class.A* %a, %class.A** %a.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %a.addr, metadata !24, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata %struct.SA* %sa, metadata !26, metadata !DIExpression()), !dbg !27
  %0 = load %class.A*, %class.A** %a.addr, align 8, !dbg !28
  %1 = bitcast %struct.SA* %agg.tmp to i8*, !dbg !28
  %2 = bitcast %struct.SA* %sa to i8*, !dbg !28
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %1, i8* align 4 %2, i64 4, i1 false), !dbg !28
  %coerce.dive1 = getelementptr %struct.SA, %struct.SA* %agg.tmp, i32 0, i32 0, !dbg !28
  %3 = load i32, i32* %coerce.dive1, !dbg !28
  call void @_ZN1A5testAE2SA(%class.A* %0, i32 %3), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN1A5testAE2SA(%class.A* %this, i32 %a.coerce) #2 align 2 !dbg !20 {
entry:
  %a = alloca %struct.SA, align 4
  %this.addr = alloca %class.A*, align 8
  %coerce.dive = getelementptr %struct.SA, %struct.SA* %a, i32 0, i32 0
  store i32 %a.coerce, i32* %coerce.dive
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !30, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata %struct.SA* %a, metadata !32, metadata !DIExpression()), !dbg !33
  %this1 = load %class.A*, %class.A** %this.addr
  ret void, !dbg !34
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #3

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (trunk 214102:214113M) (llvm/trunk 214102:214115M)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/Users/manmanren/test-Nov/type_unique/rdar_di_array")
!2 = !{}
!3 = !{!4, !10}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 5, file: !1, elements: !5, identifier: "_ZTS1A")
!5 = !{!6}
!6 = !DISubprogram(name: "testA", linkageName: "_ZN1A5testAE2SA", line: 7, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 7, file: !1, scope: !4, type: !7)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !10}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "SA", line: 1, size: 32, align: 32, file: !1, elements: !11, identifier: "_ZTS2SA")
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !1, scope: !10, baseType: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!15 = distinct !DISubprogram(name: "topA", linkageName: "_Z4topAP1A2SA", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 11, file: !1, scope: !16, type: !17, retainedNodes: !2)
!16 = !DIFile(filename: "a.cpp", directory: "/Users/manmanren/test-Nov/type_unique/rdar_di_array")
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !10}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4)
!20 = distinct !DISubprogram(name: "testA", linkageName: "_ZN1A5testAE2SA", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !4, type: !7, declaration: !6, retainedNodes: !2)
!21 = !{i32 2, !"Dwarf Version", i32 2}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.5.0 (trunk 214102:214113M) (llvm/trunk 214102:214115M)"}
!24 = !DILocalVariable(name: "a", line: 11, arg: 1, scope: !15, file: !16, type: !19)
!25 = !DILocation(line: 11, column: 14, scope: !15)
!26 = !DILocalVariable(name: "sa", line: 11, arg: 2, scope: !15, file: !16, type: !10)
!27 = !DILocation(line: 11, column: 20, scope: !15)
!28 = !DILocation(line: 12, column: 3, scope: !15)
!29 = !DILocation(line: 13, column: 1, scope: !15)
!30 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !20, type: !19)
!31 = !DILocation(line: 0, scope: !20)
!32 = !DILocalVariable(name: "a", line: 7, arg: 2, scope: !20, file: !16, type: !10)
!33 = !DILocation(line: 7, column: 17, scope: !20)
!34 = !DILocation(line: 8, column: 3, scope: !20)
