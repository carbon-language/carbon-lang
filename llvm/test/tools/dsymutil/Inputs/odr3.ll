; Generated from odr3.cpp and odr-types.h by running:
; clang -emit-llvm -g -S -std=c++11 odr3.cpp
; ModuleID = 'odr3.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%struct.S = type { i32, %"struct.S::Nested" }
%"struct.S::Nested" = type { double }

; Function Attrs: ssp uwtable
define i32 @_Z3barv() #0 {
entry:
  %this.addr.i = alloca %struct.S*, align 8
  %Add.addr.i = alloca i32, align 4
  %s = alloca %struct.S, align 8
  call void @llvm.dbg.declare(metadata %struct.S* %s, metadata !34, metadata !35), !dbg !36
  store %struct.S* %s, %struct.S** %this.addr.i, align 8, !dbg !37
  store i32 42, i32* %Add.addr.i, align 4, !dbg !37
  %this1.i = load %struct.S*, %struct.S** %this.addr.i, !dbg !37
  %0 = load i32, i32* %Add.addr.i, align 4, !dbg !38
  %I.i = getelementptr inbounds %struct.S, %struct.S* %this1.i, i32 0, i32 0, !dbg !40
  %1 = load i32, i32* %I.i, align 4, !dbg !41
  %add.i = add nsw i32 %1, %0, !dbg !41
  store i32 %add.i, i32* %I.i, align 4, !dbg !41
  %call = call i32 @_ZN1S3fooEv(%struct.S* %s), !dbg !42
  call void @llvm.dbg.declare(metadata %struct.S** %this.addr.i, metadata !43, metadata !35), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %Add.addr.i, metadata !46, metadata !35), !dbg !47
  ret i32 %call, !dbg !48
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define linkonce_odr i32 @_ZN1S3fooEv(%struct.S* %this) #2 align 2 {
entry:
  %this.addr = alloca %struct.S*, align 8
  store %struct.S* %this, %struct.S** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.S** %this.addr, metadata !49, metadata !35), !dbg !50
  %this1 = load %struct.S*, %struct.S** %this.addr
  %I = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0, !dbg !51
  %0 = load i32, i32* %I, align 4, !dbg !51
  ret i32 %0, !dbg !52
}

attributes #0 = { ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!30, !31, !32}
!llvm.ident = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 242534)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !3, subprograms: !24)
!1 = !DIFile(filename: "odr3.cpp", directory: "/Inputs")
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
!24 = !{!25, !28, !29}
!25 = !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 3, type: !26, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, function: i32 ()* @_Z3barv, variables: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{!8}
!28 = !DISubprogram(name: "incr", linkageName: "_ZN1S4incrEi", scope: !"_ZTS1S", file: !5, line: 5, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, declaration: !14, variables: !2)
!29 = !DISubprogram(name: "foo", linkageName: "_ZN1S3fooEv", scope: !"_ZTS1S", file: !5, line: 18, type: !18, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: false, function: i32 (%struct.S*)* @_ZN1S3fooEv, declaration: !17, variables: !2)
!30 = !{i32 2, !"Dwarf Version", i32 2}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 1, !"PIC Level", i32 2}
!33 = !{!"clang version 3.8.0 (trunk 242534)"}
!34 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "s", scope: !25, file: !1, line: 4, type: !"_ZTS1S")
!35 = !DIExpression()
!36 = !DILocation(line: 4, column: 4, scope: !25)
!37 = !DILocation(line: 5, column: 2, scope: !25)
!38 = !DILocation(line: 5, column: 59, scope: !28, inlinedAt: !39)
!39 = distinct !DILocation(line: 5, column: 2, scope: !25)
!40 = !DILocation(line: 5, column: 54, scope: !28, inlinedAt: !39)
!41 = !DILocation(line: 5, column: 56, scope: !28, inlinedAt: !39)
!42 = !DILocation(line: 6, column: 9, scope: !25)
!43 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, scope: !28, type: !44, flags: DIFlagArtificial | DIFlagObjectPointer)
!44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTS1S", size: 64, align: 64)
!45 = !DILocation(line: 0, scope: !28, inlinedAt: !39)
!46 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "Add", arg: 2, scope: !28, file: !5, line: 5, type: !8)
!47 = !DILocation(line: 5, column: 16, scope: !28, inlinedAt: !39)
!48 = !DILocation(line: 6, column: 2, scope: !25)
!49 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, scope: !29, type: !44, flags: DIFlagArtificial | DIFlagObjectPointer)
!50 = !DILocation(line: 0, scope: !29)
!51 = !DILocation(line: 18, column: 21, scope: !29)
!52 = !DILocation(line: 18, column: 14, scope: !29)
