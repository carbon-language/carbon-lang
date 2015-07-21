; Generated from odr7.cpp and odr-types.h by running:
; clang -emit-llvm -g -S -std=c++11 odr7.cpp
; ModuleID = 'odr7.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%"struct.S::Nested" = type { double }

; Function Attrs: nounwind ssp uwtable
define void @_Z3foov() #0 {
entry:
  %N = alloca %"struct.S::Nested", align 8
  call void @llvm.dbg.declare(metadata %"struct.S::Nested"* %N, metadata !32, metadata !33), !dbg !34
  ret void, !dbg !35
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29, !30}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 242534)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !3, subprograms: !24)
!1 = !DIFile(filename: "odr7.cpp", directory: "/Inputs")
!2 = !{}
!3 = !{!4, !20}
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
!24 = !{!25}
!25 = !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !26, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, function: void ()* @_Z3foov, variables: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{null}
!28 = !{i32 2, !"Dwarf Version", i32 2}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 1, !"PIC Level", i32 2}
!31 = !{!"clang version 3.8.0 (trunk 242534)"}
!32 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "N", scope: !25, file: !1, line: 4, type: !"_ZTSN1S6NestedE")
!33 = !DIExpression()
!34 = !DILocation(line: 4, column: 12, scope: !25)
!35 = !DILocation(line: 5, column: 1, scope: !25)
