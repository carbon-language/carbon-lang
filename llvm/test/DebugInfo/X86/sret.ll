; RUN: llc -split-dwarf=Enable -generate-cu-hash -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; Based on the debuginfo-tests/sret.cpp code.

; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0xc68148e4333befda)
; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0xc68148e4333befda)

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
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !67), !dbg !69
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !70), !dbg !71
  %this1 = load %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !72
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !72
  %m_int = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !72
  %1 = load i32* %i.addr, align 4, !dbg !72
  store i32 %1, i32* %m_int, align 4, !dbg !72
  ret void, !dbg !73
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind uwtable
define void @_ZN1AC2ERKS_(%class.A* %this, %class.A* %rhs) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %rhs.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !74), !dbg !75
  store %class.A* %rhs, %class.A** %rhs.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %rhs.addr}, metadata !76), !dbg !77
  %this1 = load %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !78
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !78
  %m_int = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !78
  %1 = load %class.A** %rhs.addr, align 8, !dbg !78
  %m_int2 = getelementptr inbounds %class.A* %1, i32 0, i32 1, !dbg !78
  %2 = load i32* %m_int2, align 4, !dbg !78
  store i32 %2, i32* %m_int, align 4, !dbg !78
  ret void, !dbg !79
}

; Function Attrs: nounwind uwtable
define %class.A* @_ZN1AaSERKS_(%class.A* %this, %class.A* %rhs) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %rhs.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !80), !dbg !81
  store %class.A* %rhs, %class.A** %rhs.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %rhs.addr}, metadata !82), !dbg !83
  %this1 = load %class.A** %this.addr
  %0 = load %class.A** %rhs.addr, align 8, !dbg !84
  %m_int = getelementptr inbounds %class.A* %0, i32 0, i32 1, !dbg !84
  %1 = load i32* %m_int, align 4, !dbg !84
  %m_int2 = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !84
  store i32 %1, i32* %m_int2, align 4, !dbg !84
  ret %class.A* %this1, !dbg !85
}

; Function Attrs: nounwind uwtable
define i32 @_ZN1A7get_intEv(%class.A* %this) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !86), !dbg !87
  %this1 = load %class.A** %this.addr
  %m_int = getelementptr inbounds %class.A* %this1, i32 0, i32 1, !dbg !88
  %0 = load i32* %m_int, align 4, !dbg !88
  ret i32 %0, !dbg !88
}

; Function Attrs: uwtable
define void @_ZN1B9AInstanceEv(%class.A* noalias sret %agg.result, %class.B* %this) #2 align 2 {
entry:
  %this.addr = alloca %class.B*, align 8
  %nrvo = alloca i1
  %cleanup.dest.slot = alloca i32
  store %class.B* %this, %class.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.B** %this.addr}, metadata !89), !dbg !91
  %this1 = load %class.B** %this.addr
  store i1 false, i1* %nrvo, !dbg !92
  call void @llvm.dbg.declare(metadata !{%class.A* %agg.result}, metadata !93), !dbg !92
  call void @_ZN1AC1Ei(%class.A* %agg.result, i32 12), !dbg !92
  store i1 true, i1* %nrvo, !dbg !94
  store i32 1, i32* %cleanup.dest.slot
  %nrvo.val = load i1* %nrvo, !dbg !95
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
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !101), !dbg !102
  %this1 = load %class.A** %this.addr
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
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !104), !dbg !105
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !106), !dbg !105
  call void @llvm.dbg.declare(metadata !{%class.B* %b}, metadata !107), !dbg !108
  call void @_ZN1BC2Ev(%class.B* %b), !dbg !108
  call void @llvm.dbg.declare(metadata !{i32* %return_val}, metadata !109), !dbg !110
  call void @_ZN1B9AInstanceEv(%class.A* sret %temp.lvalue, %class.B* %b), !dbg !110
  %call = invoke i32 @_ZN1A7get_intEv(%class.A* %temp.lvalue)
          to label %invoke.cont unwind label %lpad, !dbg !110

invoke.cont:                                      ; preds = %entry
  call void @_ZN1AD2Ev(%class.A* %temp.lvalue), !dbg !111
  store i32 %call, i32* %return_val, align 4, !dbg !111
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !113), !dbg !114
  call void @_ZN1B9AInstanceEv(%class.A* sret %a, %class.B* %b), !dbg !114
  %0 = load i32* %return_val, align 4, !dbg !115
  store i32 %0, i32* %retval, !dbg !115
  store i32 1, i32* %cleanup.dest.slot
  call void @_ZN1AD2Ev(%class.A* %a), !dbg !116
  %1 = load i32* %retval, !dbg !116
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
  %exn = load i8** %exn.slot, !dbg !119
  %sel = load i32* %ehselector.slot, !dbg !119
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
  call void @llvm.dbg.declare(metadata !{%class.B** %this.addr}, metadata !123), !dbg !124
  %this1 = load %class.B** %this.addr
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
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !126), !dbg !127
  %this1 = load %class.A** %this.addr
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
  %exn = load i8** %exn.slot, !dbg !133
  %sel = load i32* %ehselector.slot, !dbg !133
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

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (trunk 203283) (llvm/trunk 203307)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !48, metadata !2, metadata !2, metadata !"sret.dwo", i32 1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/sret.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"sret.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !37}
!4 = metadata !{i32 786434, metadata !1, null, metadata !"A", i32 1, i64 128, i64 64, i32 0, i32 0, null, metadata !5, i32 0, metadata !"_ZTS1A", null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 1, size 128, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !13, metadata !14, metadata !19, metadata !25, metadata !29, metadata !33}
!6 = metadata !{i32 786445, metadata !1, metadata !7, metadata !"_vptr$A", i32 0, i64 64, i64 0, i64 0, i32 64, metadata !8} ; [ DW_TAG_member ] [_vptr$A] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!7 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!8 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!9 = metadata !{i32 786447, null, null, metadata !"__vtbl_ptr_type", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!10 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1A", metadata !"m_int", i32 13, i64 32, i64 32, i64 64, i32 2, metadata !12} ; [ DW_TAG_member ] [m_int] [line 13, size 32, align 32, offset 64] [protected] [from int]
!14 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"", i32 4, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !18, i32 4} ; [ DW_TAG_subprogram ] [line 4] [A]
!15 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17, metadata !12}
!17 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!18 = metadata !{i32 786468}
!19 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"", i32 5, metadata !20, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !24, i32 5} ; [ DW_TAG_subprogram ] [line 5] [A]
!20 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !21, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{null, metadata !17, metadata !22}
!22 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !23} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _ZTS1A]
!24 = metadata !{i32 786468}
!25 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"operator=", metadata !"operator=", metadata !"_ZN1AaSERKS_", i32 7, metadata !26, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !28, i32 7} ; [ DW_TAG_subprogram ] [line 7] [operator=]
!26 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !27, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{metadata !22, metadata !17, metadata !22}
!28 = metadata !{i32 786468}
!29 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"~A", metadata !"~A", metadata !"", i32 8, metadata !30, i1 false, i1 false, i32 1, i32 0, metadata !"_ZTS1A", i32 256, i1 false, null, null, i32 0, metadata !32, i32 8} ; [ DW_TAG_subprogram ] [line 8] [~A]
!30 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !31, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = metadata !{null, metadata !17}
!32 = metadata !{i32 786468}
!33 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"get_int", metadata !"get_int", metadata !"_ZN1A7get_intEv", i32 10, metadata !34, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !36, i32 10} ; [ DW_TAG_subprogram ] [line 10] [get_int]
!34 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !35, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!35 = metadata !{metadata !12, metadata !17}
!36 = metadata !{i32 786468}
!37 = metadata !{i32 786434, metadata !1, null, metadata !"B", i32 38, i64 8, i64 8, i32 0, i32 0, null, metadata !38, i32 0, null, null, metadata !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 38, size 8, align 8, offset 0] [def] [from ]
!38 = metadata !{metadata !39, metadata !44}
!39 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"B", metadata !"B", metadata !"", i32 41, metadata !40, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !43, i32 41} ; [ DW_TAG_subprogram ] [line 41] [B]
!40 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !41, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!41 = metadata !{null, metadata !42}
!42 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1B]
!43 = metadata !{i32 786468}
!44 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"AInstance", metadata !"AInstance", metadata !"_ZN1B9AInstanceEv", i32 43, metadata !45, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !47, i32 43} ; [ DW_TAG_subprogram ] [line 43] [AInstance]
!45 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !46, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!46 = metadata !{metadata !4, metadata !42}
!47 = metadata !{i32 786468}
!48 = metadata !{metadata !49, metadata !50, metadata !51, metadata !52, metadata !53, metadata !54, metadata !61, metadata !62, metadata !63}
!49 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"_ZN1AC2Ei", i32 16, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*, i32)* @_ZN1AC2Ei, null, metadata !14, metadata !2, i32 18} ; [ DW_TAG_subprogram ] [line 16] [def] [scope 18] [A]
!50 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"_ZN1AC2ERKS_", i32 21, metadata !20, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*, %class.A*)* @_ZN1AC2ERKS_, null, metadata !19, metadata !2, i32 23} ; [ DW_TAG_subprogram ] [line 21] [def] [scope 23] [A]
!51 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"operator=", metadata !"operator=", metadata !"_ZN1AaSERKS_", i32 27, metadata !26, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, %class.A* (%class.A*, %class.A*)* @_ZN1AaSERKS_, null, metadata !25, metadata !2, i32 28} ; [ DW_TAG_subprogram ] [line 27] [def] [scope 28] [operator=]
!52 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"get_int", metadata !"get_int", metadata !"_ZN1A7get_intEv", i32 33, metadata !34, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%class.A*)* @_ZN1A7get_intEv, null, metadata !33, metadata !2, i32 34} ; [ DW_TAG_subprogram ] [line 33] [def] [scope 34] [get_int]
!53 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"AInstance", metadata !"AInstance", metadata !"_ZN1B9AInstanceEv", i32 47, metadata !45, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*, %class.B*)* @_ZN1B9AInstanceEv, null, metadata !44, metadata !2, i32 48} ; [ DW_TAG_subprogram ] [line 47] [def] [scope 48] [AInstance]
!54 = metadata !{i32 786478, metadata !1, metadata !7, metadata !"main", metadata !"main", metadata !"", i32 53, metadata !55, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32, i8**)* @main, null, null, metadata !2, i32 54} ; [ DW_TAG_subprogram ] [line 53] [def] [scope 54] [main]
!55 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !56, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!56 = metadata !{metadata !12, metadata !12, metadata !57}
!57 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !58} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!58 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !59} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!59 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !60} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
!60 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!61 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"~A", metadata !"~A", metadata !"_ZN1AD0Ev", i32 8, metadata !30, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*)* @_ZN1AD0Ev, null, metadata !29, metadata !2, i32 8} ; [ DW_TAG_subprogram ] [line 8] [def] [~A]
!62 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"B", metadata !"B", metadata !"_ZN1BC2Ev", i32 41, metadata !40, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.B*)* @_ZN1BC2Ev, null, metadata !39, metadata !2, i32 41} ; [ DW_TAG_subprogram ] [line 41] [def] [B]
!63 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"~A", metadata !"~A", metadata !"_ZN1AD2Ev", i32 8, metadata !30, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*)* @_ZN1AD2Ev, null, metadata !29, metadata !2, i32 8} ; [ DW_TAG_subprogram ] [line 8] [def] [~A]
!64 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!65 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!66 = metadata !{metadata !"clang version 3.5.0 (trunk 203283) (llvm/trunk 203307)"}
!67 = metadata !{i32 786689, metadata !49, metadata !"this", null, i32 16777216, metadata !68, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!68 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!69 = metadata !{i32 0, i32 0, metadata !49, null}
!70 = metadata !{i32 786689, metadata !49, metadata !"i", metadata !7, i32 33554448, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [i] [line 16]
!71 = metadata !{i32 16, i32 0, metadata !49, null}
!72 = metadata !{i32 18, i32 0, metadata !49, null}
!73 = metadata !{i32 19, i32 0, metadata !49, null}
!74 = metadata !{i32 786689, metadata !50, metadata !"this", null, i32 16777216, metadata !68, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!75 = metadata !{i32 0, i32 0, metadata !50, null}
!76 = metadata !{i32 786689, metadata !50, metadata !"rhs", metadata !7, i32 33554453, metadata !22, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [rhs] [line 21]
!77 = metadata !{i32 21, i32 0, metadata !50, null}
!78 = metadata !{i32 23, i32 0, metadata !50, null}
!79 = metadata !{i32 24, i32 0, metadata !50, null}
!80 = metadata !{i32 786689, metadata !51, metadata !"this", null, i32 16777216, metadata !68, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!81 = metadata !{i32 0, i32 0, metadata !51, null}
!82 = metadata !{i32 786689, metadata !51, metadata !"rhs", metadata !7, i32 33554459, metadata !22, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [rhs] [line 27]
!83 = metadata !{i32 27, i32 0, metadata !51, null}
!84 = metadata !{i32 29, i32 0, metadata !51, null}
!85 = metadata !{i32 30, i32 0, metadata !51, null}
!86 = metadata !{i32 786689, metadata !52, metadata !"this", null, i32 16777216, metadata !68, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!87 = metadata !{i32 0, i32 0, metadata !52, null}
!88 = metadata !{i32 35, i32 0, metadata !52, null}
!89 = metadata !{i32 786689, metadata !53, metadata !"this", null, i32 16777216, metadata !90, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!90 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1B]
!91 = metadata !{i32 0, i32 0, metadata !53, null}
!92 = metadata !{i32 49, i32 0, metadata !53, null}
!93 = metadata !{i32 786688, metadata !53, metadata !"a", metadata !7, i32 49, metadata !4, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 49]
!94 = metadata !{i32 50, i32 0, metadata !53, null}
!95 = metadata !{i32 51, i32 0, metadata !53, null}
!96 = metadata !{i32 51, i32 0, metadata !97, null}
!97 = metadata !{i32 786443, metadata !1, metadata !53, i32 51, i32 0, i32 2, i32 5} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!98 = metadata !{i32 51, i32 0, metadata !99, null}
!99 = metadata !{i32 786443, metadata !1, metadata !100, i32 51, i32 0, i32 3, i32 6} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!100 = metadata !{i32 786443, metadata !1, metadata !53, i32 51, i32 0, i32 1, i32 4} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!101 = metadata !{i32 786689, metadata !63, metadata !"this", null, i32 16777216, metadata !68, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!102 = metadata !{i32 0, i32 0, metadata !63, null}
!103 = metadata !{i32 8, i32 0, metadata !63, null} ; [ DW_TAG_imported_declaration ]
!104 = metadata !{i32 786689, metadata !54, metadata !"argc", metadata !7, i32 16777269, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [argc] [line 53]
!105 = metadata !{i32 53, i32 0, metadata !54, null}
!106 = metadata !{i32 786689, metadata !54, metadata !"argv", metadata !7, i32 33554485, metadata !57, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [argv] [line 53]
!107 = metadata !{i32 786688, metadata !54, metadata !"b", metadata !7, i32 55, metadata !37, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 55]
!108 = metadata !{i32 55, i32 0, metadata !54, null}
!109 = metadata !{i32 786688, metadata !54, metadata !"return_val", metadata !7, i32 56, metadata !12, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [return_val] [line 56]
!110 = metadata !{i32 56, i32 0, metadata !54, null}
!111 = metadata !{i32 56, i32 0, metadata !112, null}
!112 = metadata !{i32 786443, metadata !1, metadata !54, i32 56, i32 0, i32 1, i32 7} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!113 = metadata !{i32 786688, metadata !54, metadata !"a", metadata !7, i32 58, metadata !4, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 58]
!114 = metadata !{i32 58, i32 0, metadata !54, null} ; [ DW_TAG_imported_module ]
!115 = metadata !{i32 59, i32 0, metadata !54, null}
!116 = metadata !{i32 60, i32 0, metadata !54, null}
!117 = metadata !{i32 60, i32 0, metadata !118, null}
!118 = metadata !{i32 786443, metadata !1, metadata !54, i32 60, i32 0, i32 1, i32 8} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!119 = metadata !{i32 60, i32 0, metadata !120, null}
!120 = metadata !{i32 786443, metadata !1, metadata !54, i32 60, i32 0, i32 3, i32 10} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!121 = metadata !{i32 60, i32 0, metadata !122, null}
!122 = metadata !{i32 786443, metadata !1, metadata !54, i32 60, i32 0, i32 2, i32 9} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!123 = metadata !{i32 786689, metadata !62, metadata !"this", null, i32 16777216, metadata !90, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!124 = metadata !{i32 0, i32 0, metadata !62, null}
!125 = metadata !{i32 41, i32 0, metadata !62, null}
!126 = metadata !{i32 786689, metadata !61, metadata !"this", null, i32 16777216, metadata !68, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!127 = metadata !{i32 0, i32 0, metadata !61, null}
!128 = metadata !{i32 8, i32 0, metadata !61, null} ; [ DW_TAG_imported_declaration ]
!129 = metadata !{i32 8, i32 0, metadata !130, null} ; [ DW_TAG_imported_declaration ]
!130 = metadata !{i32 786443, metadata !1, metadata !61, i32 8, i32 0, i32 1, i32 11} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!131 = metadata !{i32 8, i32 0, metadata !132, null} ; [ DW_TAG_imported_declaration ]
!132 = metadata !{i32 786443, metadata !1, metadata !61, i32 8, i32 0, i32 2, i32 12} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
!133 = metadata !{i32 8, i32 0, metadata !134, null} ; [ DW_TAG_imported_declaration ]
!134 = metadata !{i32 786443, metadata !1, metadata !61, i32 8, i32 0, i32 3, i32 13} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/sret.cpp]
