; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; Based on the debuginfo-tests/sret.cpp code.

; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x51ac5644b1937aa1)
; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x51ac5644b1937aa1)

%class.A = type { i32 (...)**, i32 }
%class.B = type { i8 }

@_ZTV1A = linkonce_odr unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%class.A*)* @_ZN1AD2Ev to i8*), i8* bitcast (void (%class.A*)* @_ZN1AD0Ev to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"
@_ZTI1A = linkonce_odr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1A, i32 0, i32 0) }

@_ZN1AC1Ei = alias void (%class.A*, i32)* @_ZN1AC2Ei
@_ZN1AC1ERKS_ = alias void (%class.A*, %class.A*)* @_ZN1AC2ERKS_

; Function Attrs: nounwind uwtable
define void @_ZN1AC2Ei(%class.A* %this, i32 %i) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %i.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !67, metadata !MDExpression()), !dbg !69
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !70, metadata !MDExpression()), !dbg !71
  %this1 = load %class.A*, %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !72
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !72
  %m_int = getelementptr inbounds %class.A, %class.A* %this1, i32 0, i32 1, !dbg !72
  %1 = load i32, i32* %i.addr, align 4, !dbg !72
  store i32 %1, i32* %m_int, align 4, !dbg !72
  ret void, !dbg !73
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define void @_ZN1AC2ERKS_(%class.A* %this, %class.A* %rhs) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %rhs.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !74, metadata !MDExpression()), !dbg !75
  store %class.A* %rhs, %class.A** %rhs.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %rhs.addr, metadata !76, metadata !MDExpression()), !dbg !77
  %this1 = load %class.A*, %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !78
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !78
  %m_int = getelementptr inbounds %class.A, %class.A* %this1, i32 0, i32 1, !dbg !78
  %1 = load %class.A*, %class.A** %rhs.addr, align 8, !dbg !78
  %m_int2 = getelementptr inbounds %class.A, %class.A* %1, i32 0, i32 1, !dbg !78
  %2 = load i32, i32* %m_int2, align 4, !dbg !78
  store i32 %2, i32* %m_int, align 4, !dbg !78
  ret void, !dbg !79
}

; Function Attrs: nounwind uwtable
define %class.A* @_ZN1AaSERKS_(%class.A* %this, %class.A* %rhs) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %rhs.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !80, metadata !MDExpression()), !dbg !81
  store %class.A* %rhs, %class.A** %rhs.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %rhs.addr, metadata !82, metadata !MDExpression()), !dbg !83
  %this1 = load %class.A*, %class.A** %this.addr
  %0 = load %class.A*, %class.A** %rhs.addr, align 8, !dbg !84
  %m_int = getelementptr inbounds %class.A, %class.A* %0, i32 0, i32 1, !dbg !84
  %1 = load i32, i32* %m_int, align 4, !dbg !84
  %m_int2 = getelementptr inbounds %class.A, %class.A* %this1, i32 0, i32 1, !dbg !84
  store i32 %1, i32* %m_int2, align 4, !dbg !84
  ret %class.A* %this1, !dbg !85
}

; Function Attrs: nounwind uwtable
define i32 @_ZN1A7get_intEv(%class.A* %this) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !86, metadata !MDExpression()), !dbg !87
  %this1 = load %class.A*, %class.A** %this.addr
  %m_int = getelementptr inbounds %class.A, %class.A* %this1, i32 0, i32 1, !dbg !88
  %0 = load i32, i32* %m_int, align 4, !dbg !88
  ret i32 %0, !dbg !88
}

; Function Attrs: uwtable
define void @_ZN1B9AInstanceEv(%class.A* noalias sret %agg.result, %class.B* %this) #2 align 2 {
entry:
  %this.addr = alloca %class.B*, align 8
  %nrvo = alloca i1
  %cleanup.dest.slot = alloca i32
  store %class.B* %this, %class.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.B** %this.addr, metadata !89, metadata !MDExpression()), !dbg !91
  %this1 = load %class.B*, %class.B** %this.addr
  store i1 false, i1* %nrvo, !dbg !92
  call void @llvm.dbg.declare(metadata %class.A* %agg.result, metadata !93, metadata !MDExpression(DW_OP_deref)), !dbg !92
  call void @_ZN1AC1Ei(%class.A* %agg.result, i32 12), !dbg !92
  store i1 true, i1* %nrvo, !dbg !94
  store i32 1, i32* %cleanup.dest.slot
  %nrvo.val = load i1, i1* %nrvo, !dbg !95
  br i1 %nrvo.val, label %nrvo.skipdtor, label %nrvo.unused, !dbg !95

nrvo.unused:                                      ; preds = %entry
  call void @_ZN1AD2Ev(%class.A* %agg.result), !dbg !96
  br label %nrvo.skipdtor, !dbg !96

nrvo.skipdtor:                                    ; preds = %nrvo.unused, %entry
  ret void, !dbg !98
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN1AD2Ev(%class.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !101, metadata !MDExpression()), !dbg !102
  %this1 = load %class.A*, %class.A** %this.addr
  ret void, !dbg !103
}

; Function Attrs: uwtable
define i32 @main(i32 %argc, i8** %argv) #2 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %b = alloca %class.B, align 1
  %return_val = alloca i32, align 4
  %temp.lvalue = alloca %class.A, align 8
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %a = alloca %class.A, align 8
  %cleanup.dest.slot = alloca i32
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !104, metadata !MDExpression()), !dbg !105
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !106, metadata !MDExpression()), !dbg !105
  call void @llvm.dbg.declare(metadata %class.B* %b, metadata !107, metadata !MDExpression()), !dbg !108
  call void @_ZN1BC2Ev(%class.B* %b), !dbg !108
  call void @llvm.dbg.declare(metadata i32* %return_val, metadata !109, metadata !MDExpression()), !dbg !110
  call void @_ZN1B9AInstanceEv(%class.A* sret %temp.lvalue, %class.B* %b), !dbg !110
  %call = invoke i32 @_ZN1A7get_intEv(%class.A* %temp.lvalue)
          to label %invoke.cont unwind label %lpad, !dbg !110

invoke.cont:                                      ; preds = %entry
  call void @_ZN1AD2Ev(%class.A* %temp.lvalue), !dbg !111
  store i32 %call, i32* %return_val, align 4, !dbg !111
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !113, metadata !MDExpression()), !dbg !114
  call void @_ZN1B9AInstanceEv(%class.A* sret %a, %class.B* %b), !dbg !114
  %0 = load i32, i32* %return_val, align 4, !dbg !115
  store i32 %0, i32* %retval, !dbg !115
  store i32 1, i32* %cleanup.dest.slot
  call void @_ZN1AD2Ev(%class.A* %a), !dbg !116
  %1 = load i32, i32* %retval, !dbg !116
  ret i32 %1, !dbg !116

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup, !dbg !116
  %3 = extractvalue { i8*, i32 } %2, 0, !dbg !116
  store i8* %3, i8** %exn.slot, !dbg !116
  %4 = extractvalue { i8*, i32 } %2, 1, !dbg !116
  store i32 %4, i32* %ehselector.slot, !dbg !116
  invoke void @_ZN1AD2Ev(%class.A* %temp.lvalue)
          to label %invoke.cont1 unwind label %terminate.lpad, !dbg !116

invoke.cont1:                                     ; preds = %lpad
  br label %eh.resume, !dbg !117

eh.resume:                                        ; preds = %invoke.cont1
  %exn = load i8*, i8** %exn.slot, !dbg !119
  %sel = load i32, i32* %ehselector.slot, !dbg !119
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0, !dbg !119
  %lpad.val2 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1, !dbg !119
  resume { i8*, i32 } %lpad.val2, !dbg !119

terminate.lpad:                                   ; preds = %lpad
  %5 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null, !dbg !121
  %6 = extractvalue { i8*, i32 } %5, 0, !dbg !121
  call void @__clang_call_terminate(i8* %6) #5, !dbg !121
  unreachable, !dbg !121
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN1BC2Ev(%class.B* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.B*, align 8
  store %class.B* %this, %class.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.B** %this.addr, metadata !123, metadata !MDExpression()), !dbg !124
  %this1 = load %class.B*, %class.B** %this.addr
  ret void, !dbg !125
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) #3 {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #6
  call void @_ZSt9terminatev() #5
  unreachable
}

declare i8* @__cxa_begin_catch(i8*)

declare void @_ZSt9terminatev()

; Function Attrs: uwtable
define linkonce_odr void @_ZN1AD0Ev(%class.A* %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !126, metadata !MDExpression()), !dbg !127
  %this1 = load %class.A*, %class.A** %this.addr
  invoke void @_ZN1AD2Ev(%class.A* %this1)
          to label %invoke.cont unwind label %lpad, !dbg !128

invoke.cont:                                      ; preds = %entry
  %0 = bitcast %class.A* %this1 to i8*, !dbg !129
  call void @_ZdlPv(i8* %0) #7, !dbg !129
  ret void, !dbg !129

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup, !dbg !131
  %2 = extractvalue { i8*, i32 } %1, 0, !dbg !131
  store i8* %2, i8** %exn.slot, !dbg !131
  %3 = extractvalue { i8*, i32 } %1, 1, !dbg !131
  store i32 %3, i32* %ehselector.slot, !dbg !131
  %4 = bitcast %class.A* %this1 to i8*, !dbg !131
  call void @_ZdlPv(i8* %4) #7, !dbg !131
  br label %eh.resume, !dbg !131

eh.resume:                                        ; preds = %lpad
  %exn = load i8*, i8** %exn.slot, !dbg !133
  %sel = load i32, i32* %ehselector.slot, !dbg !133
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0, !dbg !133
  %lpad.val2 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1, !dbg !133
  resume { i8*, i32 } %lpad.val2, !dbg !133
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) #4

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline noreturn nounwind }
attributes #4 = { nobuiltin nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }
attributes #7 = { builtin nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!64, !65}
!llvm.ident = !{!66}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (trunk 203283) (llvm/trunk 203307)", isOptimized: false, splitDebugFilename: "sret.dwo", emissionKind: 1, file: !1, enums: !2, retainedTypes: !3, subprograms: !48, globals: !2, imports: !2)
!1 = !MDFile(filename: "sret.cpp", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!3 = !{!4, !37}
!4 = !MDCompositeType(tag: DW_TAG_class_type, name: "A", line: 1, size: 128, align: 64, file: !1, elements: !5, vtableHolder: !"_ZTS1A", identifier: "_ZTS1A")
!5 = !{!6, !13, !14, !19, !25, !29, !33}
!6 = !MDDerivedType(tag: DW_TAG_member, name: "_vptr$A", size: 64, flags: DIFlagArtificial, file: !1, scope: !7, baseType: !8)
!7 = !MDFile(filename: "sret.cpp", directory: "/usr/local/google/home/echristo/tmp")
!8 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !9)
!9 = !MDDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", size: 64, baseType: !10)
!10 = !MDSubroutineType(types: !11)
!11 = !{!12}
!12 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !MDDerivedType(tag: DW_TAG_member, name: "m_int", line: 13, size: 32, align: 32, offset: 64, flags: DIFlagProtected, file: !1, scope: !"_ZTS1A", baseType: !12)
!14 = !MDSubprogram(name: "A", line: 4, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !1, scope: !"_ZTS1A", type: !15)
!15 = !MDSubroutineType(types: !16)
!16 = !{null, !17, !12}
!17 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1A")
!19 = !MDSubprogram(name: "A", line: 5, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !1, scope: !"_ZTS1A", type: !20)
!20 = !MDSubroutineType(types: !21)
!21 = !{null, !17, !22}
!22 = !MDDerivedType(tag: DW_TAG_reference_type, baseType: !23)
!23 = !MDDerivedType(tag: DW_TAG_const_type, baseType: !"_ZTS1A")
!25 = !MDSubprogram(name: "operator=", linkageName: "_ZN1AaSERKS_", line: 7, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 7, file: !1, scope: !"_ZTS1A", type: !26)
!26 = !MDSubroutineType(types: !27)
!27 = !{!22, !17, !22}
!29 = !MDSubprogram(name: "~A", line: 8, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 8, file: !1, scope: !"_ZTS1A", type: !30, containingType: !"_ZTS1A")
!30 = !MDSubroutineType(types: !31)
!31 = !{null, !17}
!33 = !MDSubprogram(name: "get_int", linkageName: "_ZN1A7get_intEv", line: 10, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 10, file: !1, scope: !"_ZTS1A", type: !34)
!34 = !MDSubroutineType(types: !35)
!35 = !{!12, !17}
!37 = !MDCompositeType(tag: DW_TAG_class_type, name: "B", line: 38, size: 8, align: 8, file: !1, elements: !38, identifier: "_ZTS1B")
!38 = !{!39, !44}
!39 = !MDSubprogram(name: "B", line: 41, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 41, file: !1, scope: !"_ZTS1B", type: !40)
!40 = !MDSubroutineType(types: !41)
!41 = !{null, !42}
!42 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1B")
!44 = !MDSubprogram(name: "AInstance", linkageName: "_ZN1B9AInstanceEv", line: 43, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 43, file: !1, scope: !"_ZTS1B", type: !45)
!45 = !MDSubroutineType(types: !46)
!46 = !{!4, !42}
!48 = !{!49, !50, !51, !52, !53, !54, !61, !62, !63}
!49 = !MDSubprogram(name: "A", linkageName: "_ZN1AC2Ei", line: 16, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 18, file: !1, scope: !"_ZTS1A", type: !15, function: void (%class.A*, i32)* @_ZN1AC2Ei, declaration: !14, variables: !2)
!50 = !MDSubprogram(name: "A", linkageName: "_ZN1AC2ERKS_", line: 21, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 23, file: !1, scope: !"_ZTS1A", type: !20, function: void (%class.A*, %class.A*)* @_ZN1AC2ERKS_, declaration: !19, variables: !2)
!51 = !MDSubprogram(name: "operator=", linkageName: "_ZN1AaSERKS_", line: 27, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 28, file: !1, scope: !"_ZTS1A", type: !26, function: %class.A* (%class.A*, %class.A*)* @_ZN1AaSERKS_, declaration: !25, variables: !2)
!52 = !MDSubprogram(name: "get_int", linkageName: "_ZN1A7get_intEv", line: 33, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 34, file: !1, scope: !"_ZTS1A", type: !34, function: i32 (%class.A*)* @_ZN1A7get_intEv, declaration: !33, variables: !2)
!53 = !MDSubprogram(name: "AInstance", linkageName: "_ZN1B9AInstanceEv", line: 47, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 48, file: !1, scope: !"_ZTS1B", type: !45, function: void (%class.A*, %class.B*)* @_ZN1B9AInstanceEv, declaration: !44, variables: !2)
!54 = !MDSubprogram(name: "main", line: 53, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 54, file: !1, scope: !7, type: !55, function: i32 (i32, i8**)* @main, variables: !2)
!55 = !MDSubroutineType(types: !56)
!56 = !{!12, !12, !57}
!57 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !58)
!58 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !59)
!59 = !MDDerivedType(tag: DW_TAG_const_type, baseType: !60)
!60 = !MDBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!61 = !MDSubprogram(name: "~A", linkageName: "_ZN1AD0Ev", line: 8, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 8, file: !1, scope: !"_ZTS1A", type: !30, function: void (%class.A*)* @_ZN1AD0Ev, declaration: !29, variables: !2)
!62 = !MDSubprogram(name: "B", linkageName: "_ZN1BC2Ev", line: 41, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 41, file: !1, scope: !"_ZTS1B", type: !40, function: void (%class.B*)* @_ZN1BC2Ev, declaration: !39, variables: !2)
!63 = !MDSubprogram(name: "~A", linkageName: "_ZN1AD2Ev", line: 8, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 8, file: !1, scope: !"_ZTS1A", type: !30, function: void (%class.A*)* @_ZN1AD2Ev, declaration: !29, variables: !2)
!64 = !{i32 2, !"Dwarf Version", i32 4}
!65 = !{i32 1, !"Debug Info Version", i32 3}
!66 = !{!"clang version 3.5.0 (trunk 203283) (llvm/trunk 203307)"}
!67 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !49, type: !68)
!68 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS1A")
!69 = !MDLocation(line: 0, scope: !49)
!70 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "i", line: 16, arg: 2, scope: !49, file: !7, type: !12)
!71 = !MDLocation(line: 16, scope: !49)
!72 = !MDLocation(line: 18, scope: !49)
!73 = !MDLocation(line: 19, scope: !49)
!74 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !50, type: !68)
!75 = !MDLocation(line: 0, scope: !50)
!76 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "rhs", line: 21, arg: 2, scope: !50, file: !7, type: !22)
!77 = !MDLocation(line: 21, scope: !50)
!78 = !MDLocation(line: 23, scope: !50)
!79 = !MDLocation(line: 24, scope: !50)
!80 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !51, type: !68)
!81 = !MDLocation(line: 0, scope: !51)
!82 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "rhs", line: 27, arg: 2, scope: !51, file: !7, type: !22)
!83 = !MDLocation(line: 27, scope: !51)
!84 = !MDLocation(line: 29, scope: !51)
!85 = !MDLocation(line: 30, scope: !51)
!86 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !52, type: !68)
!87 = !MDLocation(line: 0, scope: !52)
!88 = !MDLocation(line: 35, scope: !52)
!89 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !53, type: !90)
!90 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS1B")
!91 = !MDLocation(line: 0, scope: !53)
!92 = !MDLocation(line: 49, scope: !53)
!93 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "a", line: 49, scope: !53, file: !7, type: !4)
!94 = !MDLocation(line: 50, scope: !53)
!95 = !MDLocation(line: 51, scope: !53)
!96 = !MDLocation(line: 51, scope: !97)
!97 = distinct !MDLexicalBlock(line: 51, column: 0, file: !1, scope: !53)
!98 = !MDLocation(line: 51, scope: !99)
!99 = distinct !MDLexicalBlock(line: 51, column: 0, file: !1, scope: !100)
!100 = distinct !MDLexicalBlock(line: 51, column: 0, file: !1, scope: !53)
!101 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !63, type: !68)
!102 = !MDLocation(line: 0, scope: !63)
!103 = !MDLocation(line: 8, scope: !63)
!104 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "argc", line: 53, arg: 1, scope: !54, file: !7, type: !12)
!105 = !MDLocation(line: 53, scope: !54)
!106 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "argv", line: 53, arg: 2, scope: !54, file: !7, type: !57)
!107 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "b", line: 55, scope: !54, file: !7, type: !37)
!108 = !MDLocation(line: 55, scope: !54)
!109 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "return_val", line: 56, scope: !54, file: !7, type: !12)
!110 = !MDLocation(line: 56, scope: !54)
!111 = !MDLocation(line: 56, scope: !112)
!112 = distinct !MDLexicalBlock(line: 56, column: 0, file: !1, scope: !54)
!113 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "a", line: 58, scope: !54, file: !7, type: !4)
!114 = !MDLocation(line: 58, scope: !54)
!115 = !MDLocation(line: 59, scope: !54)
!116 = !MDLocation(line: 60, scope: !54)
!117 = !MDLocation(line: 60, scope: !118)
!118 = distinct !MDLexicalBlock(line: 60, column: 0, file: !1, scope: !54)
!119 = !MDLocation(line: 60, scope: !120)
!120 = distinct !MDLexicalBlock(line: 60, column: 0, file: !1, scope: !54)
!121 = !MDLocation(line: 60, scope: !122)
!122 = distinct !MDLexicalBlock(line: 60, column: 0, file: !1, scope: !54)
!123 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !62, type: !90)
!124 = !MDLocation(line: 0, scope: !62)
!125 = !MDLocation(line: 41, scope: !62)
!126 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !61, type: !68)
!127 = !MDLocation(line: 0, scope: !61)
!128 = !MDLocation(line: 8, scope: !61)
!129 = !MDLocation(line: 8, scope: !130)
!130 = distinct !MDLexicalBlock(line: 8, column: 0, file: !1, scope: !61)
!131 = !MDLocation(line: 8, scope: !132)
!132 = distinct !MDLexicalBlock(line: 8, column: 0, file: !1, scope: !61)
!133 = !MDLocation(line: 8, scope: !134)
!134 = distinct !MDLexicalBlock(line: 8, column: 0, file: !1, scope: !61)
