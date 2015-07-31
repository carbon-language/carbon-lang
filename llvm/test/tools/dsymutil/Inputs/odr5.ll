; Generated from odr5.cpp and odr-types.h by running:
; clang -emit-llvm -g -S -std=c++11 odr5.cpp
; ModuleID = 'odr5.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%"struct.S::Nested" = type { double }

; Function Attrs: ssp uwtable
define double @_Z3bazv() #0 {
entry:
  %d = alloca %"struct.S::Nested", align 8
  call void @llvm.dbg.declare(metadata %"struct.S::Nested"* %d, metadata !39, metadata !40), !dbg !41
  call void @_ZN1S6Nested4initIiEEvT_(%"struct.S::Nested"* %d, i32 0), !dbg !42
  %D = getelementptr inbounds %"struct.S::Nested", %"struct.S::Nested"* %d, i32 0, i32 0, !dbg !43
  %0 = load double, double* %D, align 8, !dbg !43
  ret double %0, !dbg !44
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN1S6Nested4initIiEEvT_(%"struct.S::Nested"* %this, i32 %Val) #2 align 2 {
entry:
  %this.addr = alloca %"struct.S::Nested"*, align 8
  %Val.addr = alloca i32, align 4
  store %"struct.S::Nested"* %this, %"struct.S::Nested"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %"struct.S::Nested"** %this.addr, metadata !45, metadata !40), !dbg !47
  store i32 %Val, i32* %Val.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %Val.addr, metadata !48, metadata !40), !dbg !49
  %this1 = load %"struct.S::Nested"*, %"struct.S::Nested"** %this.addr
  %0 = load i32, i32* %Val.addr, align 4, !dbg !50
  %conv = sitofp i32 %0 to double, !dbg !50
  %D = getelementptr inbounds %"struct.S::Nested", %"struct.S::Nested"* %this1, i32 0, i32 0, !dbg !51
  store double %conv, double* %D, align 8, !dbg !52
  ret void, !dbg !53
}

attributes #0 = { ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36, !37}
!llvm.ident = !{!38}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 242534)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !3, subprograms: !24)
!1 = !DIFile(filename: "odr5.cpp", directory: "/Inputs")
!2 = !{}
!3 = !{!4, !20, !23}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !5, line: 1, size: 128, align: 64, elements: !6, identifier: "_ZTS1S")
!5 = !DIFile(filename: "./odr-types.h", directory: "/Inputs")
!6 = !{!7, !9, !10, !14, !17}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "I", scope: !"_ZTS1S", file: !5, line: 2, baseType: !8, size: 32, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "D", scope: !"_ZTS1S", file: !5, line: 15, baseType: !"_ZTSN1S6NestedE", size: 64, align: 64, offset: 64)
!10 = !DISubprogram(name: "incr", linkageName: "_ZN1S4incrEv", scope: !"_ZTS1S", file: !5, line: 4, type: !11, isLocal: false, isDefinition: false, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTS1S", size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!14 = !DISubprogram(name: "incr", linkageName: "_ZN1S4incrEi", scope: !"_ZTS1S", file: !5, line: 5, type: !15, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !13, !8}
!17 = !DISubprogram(name: "foo", linkageName: "_ZN1S3fooEv", scope: !"_ZTS1S", file: !5, line: 18, type: !18, isLocal: false, isDefinition: false, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: false)
!18 = !DISubroutineType(types: !19)
!19 = !{!8, !13}
!20 = !DICompositeType(tag: DW_TAG_structure_type, name: "Nested", scope: !"_ZTS1S", file: !5, line: 9, size: 64, align: 64, elements: !21, identifier: "_ZTSN1S6NestedE")
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "D", scope: !"_ZTSN1S6NestedE", file: !5, line: 10, baseType: !23, size: 64, align: 64)
!23 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!24 = !{!25, !28}
!25 = !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 3, type: !26, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, function: double ()* @_Z3bazv, variables: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{!23}
!28 = !DISubprogram(name: "init<int>", linkageName: "_ZN1S6Nested4initIiEEvT_", scope: !"_ZTSN1S6NestedE", file: !5, line: 12, type: !29, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, function: void (%"struct.S::Nested"*, i32)* @_ZN1S6Nested4initIiEEvT_, templateParams: !32, declaration: !34, variables: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31, !8}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTSN1S6NestedE", size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = !{!33}
!33 = !DITemplateTypeParameter(name: "T", type: !8)
!34 = !DISubprogram(name: "init<int>", linkageName: "_ZN1S6Nested4initIiEEvT_", scope: !"_ZTSN1S6NestedE", file: !5, line: 12, type: !29, isLocal: false, isDefinition: false, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, templateParams: !32)
!35 = !{i32 2, !"Dwarf Version", i32 2}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !{i32 1, !"PIC Level", i32 2}
!38 = !{!"clang version 3.8.0 (trunk 242534)"}
!39 = !DILocalVariable(name: "d", scope: !25, file: !1, line: 4, type: !"_ZTSN1S6NestedE")
!40 = !DIExpression()
!41 = !DILocation(line: 4, column: 12, scope: !25)
!42 = !DILocation(line: 5, column: 2, scope: !25)
!43 = !DILocation(line: 6, column: 11, scope: !25)
!44 = !DILocation(line: 6, column: 2, scope: !25)
!45 = !DILocalVariable(name: "this", arg: 1, scope: !28, type: !46, flags: DIFlagArtificial | DIFlagObjectPointer)
!46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTSN1S6NestedE", size: 64, align: 64)
!47 = !DILocation(line: 0, scope: !28)
!48 = !DILocalVariable(name: "Val", arg: 2, scope: !28, file: !5, line: 12, type: !8)
!49 = !DILocation(line: 12, column: 36, scope: !28)
!50 = !DILocation(line: 12, column: 54, scope: !28)
!51 = !DILocation(line: 12, column: 43, scope: !28)
!52 = !DILocation(line: 12, column: 45, scope: !28)
!53 = !DILocation(line: 12, column: 60, scope: !28)
