; Generated from odr1.cpp and odr-types.h by running:
; clang -emit-llvm -g -S -std=c++11 odr1.cpp
; ModuleID = 'odr1.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%struct.S = type { i32, %"struct.S::Nested" }
%"struct.S::Nested" = type { double }
%"class.N::C" = type { %struct.S }
%"class.N::N::C" = type { i32 }
%"class.(anonymous namespace)::AnonC" = type { i8 }
%union.U = type { %"class.U::C" }
%"class.U::C" = type { i8 }
%class.anon = type { i8 }
%struct.CInsideFunc = type { i32 }

; Function Attrs: ssp uwtable
define i32 @_Z3foov() #0 {
entry:
  %s = alloca %struct.S, align 8
  %nc = alloca %"class.N::C", align 8
  %nnc = alloca %"class.N::N::C", align 4
  %ac = alloca %"class.(anonymous namespace)::AnonC", align 1
  %u = alloca %union.U, align 1
  call void @llvm.dbg.declare(metadata %struct.S* %s, metadata !59, metadata !60), !dbg !61
  call void @llvm.dbg.declare(metadata %"class.N::C"* %nc, metadata !62, metadata !60), !dbg !63
  call void @llvm.dbg.declare(metadata %"class.N::N::C"* %nnc, metadata !64, metadata !60), !dbg !65
  call void @llvm.dbg.declare(metadata %"class.(anonymous namespace)::AnonC"* %ac, metadata !66, metadata !60), !dbg !69
  call void @llvm.dbg.declare(metadata %union.U* %u, metadata !70, metadata !60), !dbg !71
  %call = call i32 @_Z4funcv(), !dbg !72
  ret i32 %call, !dbg !73
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: inlinehint ssp uwtable
define linkonce_odr i32 @_Z4funcv() #2 {
entry:
  %functor = alloca %class.anon, align 1
  call void @llvm.dbg.declare(metadata %class.anon* %functor, metadata !74, metadata !60), !dbg !75
  %call = call i32 @_ZZ4funcvENKUlvE_clEv(%class.anon* %functor), !dbg !76
  ret i32 %call, !dbg !77
}

; Function Attrs: inlinehint nounwind ssp uwtable
define linkonce_odr i32 @_ZZ4funcvENKUlvE_clEv(%class.anon* %this) #3 align 2 {
entry:
  %this.addr = alloca %class.anon*, align 8
  %dummy = alloca %struct.CInsideFunc, align 4
  store %class.anon* %this, %class.anon** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.anon** %this.addr, metadata !78, metadata !60), !dbg !80
  %this1 = load %class.anon*, %class.anon** %this.addr
  call void @llvm.dbg.declare(metadata %struct.CInsideFunc* %dummy, metadata !81, metadata !60), !dbg !82
  %i = getelementptr inbounds %struct.CInsideFunc, %struct.CInsideFunc* %dummy, i32 0, i32 0, !dbg !83
  %0 = load i32, i32* %i, align 4, !dbg !83
  ret i32 %0, !dbg !84
}

attributes #0 = { ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { inlinehint ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!55, !56, !57}
!llvm.ident = !{!58}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 242534)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !3, subprograms: !52)
!1 = !DIFile(filename: "odr1.cpp", directory: "/Inputs")
!2 = !{}
!3 = !{!4, !20, !24, !29, !33, !37, !38, !39, !49}
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
!24 = !DICompositeType(tag: DW_TAG_class_type, name: "C", scope: !25, file: !5, line: 24, size: 128, align: 64, elements: !26, identifier: "_ZTSN1N1CE")
!25 = !DINamespace(name: "N", scope: null, file: !5, line: 23)
!26 = !{!27}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "S", scope: !"_ZTSN1N1CE", file: !5, line: 25, baseType: !28, size: 128, align: 64)
!28 = !DIDerivedType(tag: DW_TAG_typedef, name: "AliasForS", file: !5, line: 21, baseType: !"_ZTS1S")
!29 = !DICompositeType(tag: DW_TAG_class_type, name: "C", scope: !30, file: !5, line: 31, size: 32, align: 32, elements: !31, identifier: "_ZTSN1N1N1CE")
!30 = !DINamespace(name: "N", scope: !25, file: !5, line: 30)
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "S", scope: !"_ZTSN1N1N1CE", file: !5, line: 32, baseType: !8, size: 32, align: 32)
!33 = !DICompositeType(tag: DW_TAG_union_type, name: "U", file: !5, line: 42, size: 8, align: 8, elements: !34, identifier: "_ZTS1U")
!34 = !{!35, !36}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "C", scope: !"_ZTS1U", file: !5, line: 43, baseType: !"_ZTSN1U1CE", size: 8, align: 8)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "S", scope: !"_ZTS1U", file: !5, line: 44, baseType: !"_ZTSN1U1SE", size: 8, align: 8)
!37 = !DICompositeType(tag: DW_TAG_class_type, name: "C", scope: !"_ZTS1U", file: !5, line: 43, size: 8, align: 8, elements: !2, identifier: "_ZTSN1U1CE")
!38 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !"_ZTS1U", file: !5, line: 44, size: 8, align: 8, elements: !2, identifier: "_ZTSN1U1SE")
!39 = !DICompositeType(tag: DW_TAG_class_type, scope: !40, file: !5, line: 49, size: 8, align: 8, elements: !43, identifier: "_ZTSZ4funcvEUlvE_")
!40 = !DISubprogram(name: "func", linkageName: "_Z4funcv", scope: !5, file: !5, line: 47, type: !41, isLocal: false, isDefinition: true, scopeLine: 47, flags: DIFlagPrototyped, isOptimized: false, function: i32 ()* @_Z4funcv, variables: !2)
!41 = !DISubroutineType(types: !42)
!42 = !{!8}
!43 = !{!44}
!44 = !DISubprogram(name: "operator()", scope: !"_ZTSZ4funcvEUlvE_", file: !5, line: 49, type: !45, isLocal: false, isDefinition: false, scopeLine: 49, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!45 = !DISubroutineType(types: !46)
!46 = !{!8, !47}
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !48, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!48 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !"_ZTSZ4funcvEUlvE_")
!49 = !DICompositeType(tag: DW_TAG_structure_type, name: "CInsideFunc", scope: !40, file: !5, line: 48, size: 32, align: 32, elements: !50, identifier: "_ZTSZ4funcvE11CInsideFunc")
!50 = !{!51}
!51 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !"_ZTSZ4funcvE11CInsideFunc", file: !5, line: 48, baseType: !8, size: 32, align: 32)
!52 = !{!53, !40, !54}
!53 = !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !41, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, function: i32 ()* @_Z3foov, variables: !2)
!54 = !DISubprogram(name: "operator()", linkageName: "_ZZ4funcvENKUlvE_clEv", scope: !"_ZTSZ4funcvEUlvE_", file: !5, line: 49, type: !45, isLocal: false, isDefinition: true, scopeLine: 49, flags: DIFlagPrototyped, isOptimized: false, function: i32 (%class.anon*)* @_ZZ4funcvENKUlvE_clEv, declaration: !44, variables: !2)
!55 = !{i32 2, !"Dwarf Version", i32 2}
!56 = !{i32 2, !"Debug Info Version", i32 3}
!57 = !{i32 1, !"PIC Level", i32 2}
!58 = !{!"clang version 3.8.0 (trunk 242534)"}
!59 = !DILocalVariable(name: "s", scope: !53, file: !1, line: 4, type: !28)
!60 = !DIExpression()
!61 = !DILocation(line: 4, column: 12, scope: !53)
!62 = !DILocalVariable(name: "nc", scope: !53, file: !1, line: 5, type: !"_ZTSN1N1CE")
!63 = !DILocation(line: 5, column: 7, scope: !53)
!64 = !DILocalVariable(name: "nnc", scope: !53, file: !1, line: 6, type: !"_ZTSN1N1N1CE")
!65 = !DILocation(line: 6, column: 10, scope: !53)
!66 = !DILocalVariable(name: "ac", scope: !53, file: !1, line: 7, type: !67)
!67 = !DICompositeType(tag: DW_TAG_class_type, name: "AnonC", scope: !68, file: !5, line: 38, size: 8, align: 8, elements: !2)
!68 = !DINamespace(scope: null, file: !5, line: 37)
!69 = !DILocation(line: 7, column: 8, scope: !53)
!70 = !DILocalVariable(name: "u", scope: !53, file: !1, line: 8, type: !"_ZTS1U")
!71 = !DILocation(line: 8, column: 4, scope: !53)
!72 = !DILocation(line: 10, column: 9, scope: !53)
!73 = !DILocation(line: 10, column: 2, scope: !53)
!74 = !DILocalVariable(name: "functor", scope: !40, file: !5, line: 49, type: !"_ZTSZ4funcvEUlvE_")
!75 = !DILocation(line: 49, column: 7, scope: !40)
!76 = !DILocation(line: 50, column: 9, scope: !40)
!77 = !DILocation(line: 50, column: 2, scope: !40)
!78 = !DILocalVariable(name: "this", arg: 1, scope: !54, type: !79, flags: DIFlagArtificial | DIFlagObjectPointer)
!79 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !48, size: 64, align: 64)
!80 = !DILocation(line: 0, scope: !54)
!81 = !DILocalVariable(name: "dummy", scope: !54, file: !5, line: 49, type: !"_ZTSZ4funcvE11CInsideFunc")
!82 = !DILocation(line: 49, column: 36, scope: !54)
!83 = !DILocation(line: 49, column: 56, scope: !54)
!84 = !DILocation(line: 49, column: 43, scope: !54)
