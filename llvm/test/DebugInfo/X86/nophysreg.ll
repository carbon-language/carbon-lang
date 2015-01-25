; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
;
; PR22296: In this testcase the DBG_VALUE describing "p5" becomes unavailable
; because the register its address is in is clobbered and we (currently) aren't
; smart enough to realize that the value is rematerialized immediately after the
; DBG_VALUE and/or is actually a stack slot.
;
; Test that we handle this situation gracefully by omitting the DW_AT_location
; and not asserting.
;
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_name {{.*}}"p5"
; CHECK-NOT: DW_AT_location
; CHECK: DW_TAG
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%class.D.1.85 = type { %class.C.0.84 }
%class.C.0.84 = type { i8 }
%class.J.2.86 = type { i8 }
%class.B.7.91 = type { i8 }
%class.D.3.9.93 = type { %class.C.4.8.92 }
%class.C.4.8.92 = type { i8 }
%class.G.3.87 = type { i8 }
%class.F.4.88 = type { i8 }
%class.D.0.6.90 = type { %class.C.1.5.89 }
%class.C.1.5.89 = type { i8 }

; Function Attrs: nounwind readnone ssp uwtable
declare %class.D.1.85* @_Z9addressofR1DIPiE(%class.D.1.85* readnone dereferenceable(1)) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_ZN1J20buildRegisterClassesERi(%class.J.2.86* %this, i32* nocapture readnone dereferenceable(4)) #2 align 2 {
entry:
  %ref.tmp.i.i34 = alloca i32, align 4
  %d = alloca %class.D.1.85, align 1
  %__begin7 = alloca %class.B.7.91, align 1
  %__end9 = alloca %class.B.7.91, align 1
  %e = alloca %class.D.1.85, align 1
  %f = alloca %class.D.3.9.93, align 1
  br i1 undef, label %for.body.lr.ph, label %for.end28, !dbg !236

for.body.lr.ph:                                   ; preds = %entry
  %__tree_.i.i33 = getelementptr inbounds %class.D.1.85* %d, i64 0, i32 0, !dbg !237
  %__tree_.i.i35 = getelementptr inbounds %class.D.3.9.93* %f, i64 0, i32 0, !dbg !240
  br label %for.body, !dbg !236

for.body:                                         ; preds = %for.inc27, %for.body.lr.ph
  call void @_ZN1CIPiEC1ERKi(%class.C.0.84* %__tree_.i.i33, i32* dereferenceable(4) undef) #4, !dbg !237
  br i1 undef, label %for.body14, label %for.inc27, !dbg !243

for.body14:                                       ; preds = %for.body14, %for.body
  call void @_ZN1CIiEC1ERKi(%class.C.4.8.92* %__tree_.i.i35, i32* dereferenceable(4) %ref.tmp.i.i34) #4, !dbg !240
  call void @_ZN1DIPiE3endEv(%class.D.1.85* %e) #4, !dbg !244
  call void @llvm.dbg.value(metadata %class.D.1.85* %d, i64 0, metadata !245, metadata !247) #4, !dbg !248
  call void @_Z16set_intersectionI1BIiES0_IPiE1AI1DIS2_EEiEvT_T0_T1_T2_(%class.D.1.85* %d, i32 0) #4, !dbg !249
  call void @_ZN1BI1DIPiEEppEv(%class.B.7.91* %__begin7) #4, !dbg !243
  %call13 = call zeroext i1 @_Zne1BI1DIPiEERS3_(%class.B.7.91* dereferenceable(1) %__end9) #4, !dbg !243
  br i1 %call13, label %for.body14, label %for.inc27, !dbg !243

for.inc27:                                        ; preds = %for.body14, %for.body
  %call = call zeroext i1 @_Zne1FS_() #4, !dbg !236
  br i1 %call, label %for.body, label %for.end28, !dbg !236

for.end28:                                        ; preds = %for.inc27, %entry
  ret void, !dbg !250
}

declare void @_ZN1J12getRegistersEv(%class.J.2.86*) #3

declare void @_ZN1G5beginEv(%class.G.3.87*) #3

declare void @_ZN1G3endEv(%class.G.3.87*) #3

declare zeroext i1 @_Zne1FS_() #3

declare i32 @_ZN1FdeEv(%class.F.4.88*) #3

declare void @_ZN1DIS_IPiEE3endEv(%class.D.0.6.90*) #3

declare zeroext i1 @_Zne1BI1DIPiEERS3_(%class.B.7.91* dereferenceable(1)) #3

declare void @_ZN1BI1DIPiEEdeEv(%class.B.7.91*) #3

declare void @_ZN1DIiE3endEv(%class.D.3.9.93*) #3

declare void @_ZN1DIPiE3endEv(%class.D.1.85*) #3

declare void @_ZN1BI1DIPiEEppEv(%class.B.7.91*) #3

declare void @_ZN1FppEv(%class.F.4.88*) #3

declare void @_ZN1CI1DIPiEEC1ERKi(%class.C.1.5.89*, i32* dereferenceable(4)) #3

declare void @_ZN1CIPiEC1ERKi(%class.C.0.84*, i32* dereferenceable(4)) #3

declare void @_ZN1CIiEC1ERKi(%class.C.4.8.92*, i32* dereferenceable(4)) #3

declare void @_Z16set_intersectionI1BIiES0_IPiE1AI1DIS2_EEiEvT_T0_T1_T2_(%class.D.1.85*, i32) #3

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #4

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #4

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind ssp uwtable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!232, !233, !234}
!llvm.ident = !{!235}

!0 = !{!"0x11\004\00clang version 3.7.0 (trunk 226915) (llvm/trunk 226905)\001\00\000\00\001", !1, !2, !3, !146, !2, !2} ; [ DW_TAG_compile_unit ] [/AsmMatcherEmitter.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"AsmMatcherEmitter.cpp", !""}
!2 = !{}
!3 = !{!4, !23, !33, !42, !52, !59, !68, !84, !92, !101, !117, !125, !134}
!4 = !{!"0x2\00D<int *>\0022\008\008\000\000\000", !5, null, null, !6, null, !19, !"_ZTS1DIPiE"} ; [ DW_TAG_class_type ] [D<int *>] [line 22, size 8, align 8, offset 0] [def] [from ]
!5 = !{!"test1.cpp", !""}
!6 = !{!7, !9, !13, !18}
!7 = !{!"0xd\00__tree_\0024\008\008\000\000", !5, !"_ZTS1DIPiE", !8} ; [ DW_TAG_member ] [__tree_] [line 24, size 8, align 8, offset 0] [from __base]
!8 = !{!"0x16\00__base\0023\000\000\000\000", !5, !"_ZTS1DIPiE", !"_ZTS1CIPiE"} ; [ DW_TAG_typedef ] [__base] [line 23, size 0, align 0, offset 0] [from _ZTS1CIPiE]
!9 = !{!"0x2e\00D\00D\00\0027\000\000\000\000\00259\001\0027", !5, !"_ZTS1DIPiE", !10, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 27] [public] [D]
!10 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{null, !12}
!12 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1DIPiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1DIPiE]
!13 = !{!"0x2e\00begin\00begin\00_ZN1DIPiE5beginEv\0028\000\000\000\000\00259\001\0028", !5, !"_ZTS1DIPiE", !14, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 28] [public] [begin]
!14 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{!16, !12}
!16 = !{!"0x16\00iterator\0026\000\000\000\000", !5, !"_ZTS1DIPiE", !17} ; [ DW_TAG_typedef ] [iterator] [line 26, size 0, align 0, offset 0] [from const_iterator]
!17 = !{!"0x16\00const_iterator\0019\000\000\000\000", !5, !"_ZTS1CIPiE", !"_ZTS1BIPiE"} ; [ DW_TAG_typedef ] [const_iterator] [line 19, size 0, align 0, offset 0] [from _ZTS1BIPiE]
!18 = !{!"0x2e\00end\00end\00_ZN1DIPiE3endEv\0029\000\000\000\000\00259\001\0029", !5, !"_ZTS1DIPiE", !14, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 29] [public] [end]
!19 = !{!20}
!20 = !{!"0x2f\00_Key\000\000", null, !21, null}  ; [ DW_TAG_template_type_parameter ]
!21 = !{!"0xf\00\000\0064\0064\000\000", null, null, !22} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!22 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!23 = !{!"0x2\00C<int *>\0017\008\008\000\000\000", !5, null, null, !24, null, !31, !"_ZTS1CIPiE"} ; [ DW_TAG_class_type ] [C<int *>] [line 17, size 8, align 8, offset 0] [def] [from ]
!24 = !{!25}
!25 = !{!"0x2e\00C\00C\00\0020\000\000\000\000\00259\001\0020", !5, !"_ZTS1CIPiE", !26, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 20] [public] [C]
!26 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !27, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = !{null, !28, !29}
!28 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1CIPiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1CIPiE]
!29 = !{!"0x10\00\000\000\000\000\000", null, null, !30} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = !{!"0x26\00\000\000\000\000\000", null, null, !22} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!31 = !{!32}
!32 = !{!"0x2f\00_Tp\000\000", null, !21, null}   ; [ DW_TAG_template_type_parameter ]
!33 = !{!"0x2\00B<int *>\0011\008\008\000\000\000", !5, null, null, !34, null, !31, !"_ZTS1BIPiE"} ; [ DW_TAG_class_type ] [B<int *>] [line 11, size 8, align 8, offset 0] [def] [from ]
!34 = !{!35, !39}
!35 = !{!"0x2e\00operator*\00operator*\00_ZN1BIPiEdeEv\0013\000\000\000\000\00259\001\0013", !5, !"_ZTS1BIPiE", !36, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 13] [public] [operator*]
!36 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !37, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!37 = !{!21, !38}
!38 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1BIPiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1BIPiE]
!39 = !{!"0x2e\00operator++\00operator++\00_ZN1BIPiEppEv\0014\000\000\000\000\00259\001\0014", !5, !"_ZTS1BIPiE", !40, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 14] [public] [operator++]
!40 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !41, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!41 = !{null, !38}
!42 = !{!"0x2\00J\0042\008\008\000\000\000", !5, null, null, !43, null, null, !"_ZTS1J"} ; [ DW_TAG_class_type ] [J] [line 42, size 8, align 8, offset 0] [def] [from ]
!43 = !{!44, !48}
!44 = !{!"0x2e\00getRegisters\00getRegisters\00_ZN1J12getRegistersEv\0043\000\000\000\000\00256\001\0043", !5, !"_ZTS1J", !45, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 43] [getRegisters]
!45 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !46, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!46 = !{!"_ZTS1G", !47}
!47 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1J"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1J]
!48 = !{!"0x2e\00buildRegisterClasses\00buildRegisterClasses\00_ZN1J20buildRegisterClassesERi\0044\000\000\000\000\00256\001\0044", !5, !"_ZTS1J", !49, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 44] [buildRegisterClasses]
!49 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !50, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!50 = !{null, !47, !51}
!51 = !{!"0x10\00\000\000\000\000\000", null, null, !22} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from int]
!52 = !{!"0x2\00G\0037\008\008\000\000\000", !5, null, null, !53, null, null, !"_ZTS1G"} ; [ DW_TAG_class_type ] [G] [line 37, size 8, align 8, offset 0] [def] [from ]
!53 = !{!54, !58}
!54 = !{!"0x2e\00begin\00begin\00_ZN1G5beginEv\0039\000\000\000\000\00259\001\0039", !5, !"_ZTS1G", !55, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 39] [public] [begin]
!55 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !56, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!56 = !{!"_ZTS1F", !57}
!57 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1G"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1G]
!58 = !{!"0x2e\00end\00end\00_ZN1G3endEv\0040\000\000\000\000\00259\001\0040", !5, !"_ZTS1G", !55, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 40] [public] [end]
!59 = !{!"0x2\00F\0031\008\008\000\000\000", !5, null, null, !60, null, null, !"_ZTS1F"} ; [ DW_TAG_class_type ] [F] [line 31, size 8, align 8, offset 0] [def] [from ]
!60 = !{!61, !65}
!61 = !{!"0x2e\00operator*\00operator*\00_ZN1FdeEv\0033\000\000\000\000\00259\001\0033", !5, !"_ZTS1F", !62, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 33] [public] [operator*]
!62 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !63, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!63 = !{!22, !64}
!64 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1F"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1F]
!65 = !{!"0x2e\00operator++\00operator++\00_ZN1FppEv\0034\000\000\000\000\00259\001\0034", !5, !"_ZTS1F", !66, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 34] [public] [operator++]
!66 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !67, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!67 = !{null, !64}
!68 = !{!"0x2\00D<D<int *> >\0022\008\008\000\000\000", !5, null, null, !69, null, !82, !"_ZTS1DIS_IPiEE"} ; [ DW_TAG_class_type ] [D<D<int *> >] [line 22, size 8, align 8, offset 0] [def] [from ]
!69 = !{!70, !72, !76, !81}
!70 = !{!"0xd\00__tree_\0024\008\008\000\000", !5, !"_ZTS1DIS_IPiEE", !71} ; [ DW_TAG_member ] [__tree_] [line 24, size 8, align 8, offset 0] [from __base]
!71 = !{!"0x16\00__base\0023\000\000\000\000", !5, !"_ZTS1DIS_IPiEE", !"_ZTS1CI1DIPiEE"} ; [ DW_TAG_typedef ] [__base] [line 23, size 0, align 0, offset 0] [from _ZTS1CI1DIPiEE]
!72 = !{!"0x2e\00D\00D\00\0027\000\000\000\000\00259\001\0027", !5, !"_ZTS1DIS_IPiEE", !73, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 27] [public] [D]
!73 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !74, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!74 = !{null, !75}
!75 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1DIS_IPiEE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1DIS_IPiEE]
!76 = !{!"0x2e\00begin\00begin\00_ZN1DIS_IPiEE5beginEv\0028\000\000\000\000\00259\001\0028", !5, !"_ZTS1DIS_IPiEE", !77, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 28] [public] [begin]
!77 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !78, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!78 = !{!79, !75}
!79 = !{!"0x16\00iterator\0026\000\000\000\000", !5, !"_ZTS1DIS_IPiEE", !80} ; [ DW_TAG_typedef ] [iterator] [line 26, size 0, align 0, offset 0] [from const_iterator]
!80 = !{!"0x16\00const_iterator\0019\000\000\000\000", !5, !"_ZTS1CI1DIPiEE", !"_ZTS1BI1DIPiEE"} ; [ DW_TAG_typedef ] [const_iterator] [line 19, size 0, align 0, offset 0] [from _ZTS1BI1DIPiEE]
!81 = !{!"0x2e\00end\00end\00_ZN1DIS_IPiEE3endEv\0029\000\000\000\000\00259\001\0029", !5, !"_ZTS1DIS_IPiEE", !77, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 29] [public] [end]
!82 = !{!83}
!83 = !{!"0x2f\00_Key\000\000", null, !"_ZTS1DIPiE", null} ; [ DW_TAG_template_type_parameter ]
!84 = !{!"0x2\00C<D<int *> >\0017\008\008\000\000\000", !5, null, null, !85, null, !90, !"_ZTS1CI1DIPiEE"} ; [ DW_TAG_class_type ] [C<D<int *> >] [line 17, size 8, align 8, offset 0] [def] [from ]
!85 = !{!86}
!86 = !{!"0x2e\00C\00C\00\0020\000\000\000\000\00259\001\0020", !5, !"_ZTS1CI1DIPiEE", !87, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 20] [public] [C]
!87 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !88, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!88 = !{null, !89, !29}
!89 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1CI1DIPiEE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1CI1DIPiEE]
!90 = !{!91}
!91 = !{!"0x2f\00_Tp\000\000", null, !"_ZTS1DIPiE", null} ; [ DW_TAG_template_type_parameter ]
!92 = !{!"0x2\00B<D<int *> >\0011\008\008\000\000\000", !5, null, null, !93, null, !90, !"_ZTS1BI1DIPiEE"} ; [ DW_TAG_class_type ] [B<D<int *> >] [line 11, size 8, align 8, offset 0] [def] [from ]
!93 = !{!94, !98}
!94 = !{!"0x2e\00operator*\00operator*\00_ZN1BI1DIPiEEdeEv\0013\000\000\000\000\00259\001\0013", !5, !"_ZTS1BI1DIPiEE", !95, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 13] [public] [operator*]
!95 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !96, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!96 = !{!"_ZTS1DIPiE", !97}
!97 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1BI1DIPiEE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1BI1DIPiEE]
!98 = !{!"0x2e\00operator++\00operator++\00_ZN1BI1DIPiEEppEv\0014\000\000\000\000\00259\001\0014", !5, !"_ZTS1BI1DIPiEE", !99, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 14] [public] [operator++]
!99 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !100, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!100 = !{null, !97}
!101 = !{!"0x2\00D<int>\0022\008\008\000\000\000", !5, null, null, !102, null, !115, !"_ZTS1DIiE"} ; [ DW_TAG_class_type ] [D<int>] [line 22, size 8, align 8, offset 0] [def] [from ]
!102 = !{!103, !105, !109, !114}
!103 = !{!"0xd\00__tree_\0024\008\008\000\000", !5, !"_ZTS1DIiE", !104} ; [ DW_TAG_member ] [__tree_] [line 24, size 8, align 8, offset 0] [from __base]
!104 = !{!"0x16\00__base\0023\000\000\000\000", !5, !"_ZTS1DIiE", !"_ZTS1CIiE"} ; [ DW_TAG_typedef ] [__base] [line 23, size 0, align 0, offset 0] [from _ZTS1CIiE]
!105 = !{!"0x2e\00D\00D\00\0027\000\000\000\000\00259\001\0027", !5, !"_ZTS1DIiE", !106, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 27] [public] [D]
!106 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !107, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!107 = !{null, !108}
!108 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1DIiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1DIiE]
!109 = !{!"0x2e\00begin\00begin\00_ZN1DIiE5beginEv\0028\000\000\000\000\00259\001\0028", !5, !"_ZTS1DIiE", !110, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 28] [public] [begin]
!110 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !111, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!111 = !{!112, !108}
!112 = !{!"0x16\00iterator\0026\000\000\000\000", !5, !"_ZTS1DIiE", !113} ; [ DW_TAG_typedef ] [iterator] [line 26, size 0, align 0, offset 0] [from const_iterator]
!113 = !{!"0x16\00const_iterator\0019\000\000\000\000", !5, !"_ZTS1CIiE", !"_ZTS1BIiE"} ; [ DW_TAG_typedef ] [const_iterator] [line 19, size 0, align 0, offset 0] [from _ZTS1BIiE]
!114 = !{!"0x2e\00end\00end\00_ZN1DIiE3endEv\0029\000\000\000\000\00259\001\0029", !5, !"_ZTS1DIiE", !110, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 29] [public] [end]
!115 = !{!116}
!116 = !{!"0x2f\00_Key\000\000", null, !22, null} ; [ DW_TAG_template_type_parameter ]
!117 = !{!"0x2\00C<int>\0017\008\008\000\000\000", !5, null, null, !118, null, !123, !"_ZTS1CIiE"} ; [ DW_TAG_class_type ] [C<int>] [line 17, size 8, align 8, offset 0] [def] [from ]
!118 = !{!119}
!119 = !{!"0x2e\00C\00C\00\0020\000\000\000\000\00259\001\0020", !5, !"_ZTS1CIiE", !120, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 20] [public] [C]
!120 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !121, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!121 = !{null, !122, !29}
!122 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1CIiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1CIiE]
!123 = !{!124}
!124 = !{!"0x2f\00_Tp\000\000", null, !22, null}  ; [ DW_TAG_template_type_parameter ]
!125 = !{!"0x2\00B<int>\0011\008\008\000\000\000", !5, null, null, !126, null, !123, !"_ZTS1BIiE"} ; [ DW_TAG_class_type ] [B<int>] [line 11, size 8, align 8, offset 0] [def] [from ]
!126 = !{!127, !131}
!127 = !{!"0x2e\00operator*\00operator*\00_ZN1BIiEdeEv\0013\000\000\000\000\00259\001\0013", !5, !"_ZTS1BIiE", !128, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 13] [public] [operator*]
!128 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !129, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!129 = !{!22, !130}
!130 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1BIiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1BIiE]
!131 = !{!"0x2e\00operator++\00operator++\00_ZN1BIiEppEv\0014\000\000\000\000\00259\001\0014", !5, !"_ZTS1BIiE", !132, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 14] [public] [operator++]
!132 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !133, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!133 = !{null, !130}
!134 = !{!"0x2\00A<D<int *> >\001\00128\0064\000\000\000", !5, null, null, !135, null, !144, !"_ZTS1AI1DIPiEE"} ; [ DW_TAG_class_type ] [A<D<int *> >] [line 1, size 128, align 64, offset 0] [def] [from ]
!135 = !{!136, !138, !139}
!136 = !{!"0xd\00container\002\0064\0064\000\000", !5, !"_ZTS1AI1DIPiEE", !137} ; [ DW_TAG_member ] [container] [line 2, size 64, align 64, offset 0] [from ]
!137 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1DIPiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1DIPiE]
!138 = !{!"0xd\00iter\003\008\008\0064\000", !5, !"_ZTS1AI1DIPiEE", !16} ; [ DW_TAG_member ] [iter] [line 3, size 8, align 8, offset 64] [from iterator]
!139 = !{!"0x2e\00A\00A\00\005\000\000\000\000\00259\001\005", !5, !"_ZTS1AI1DIPiEE", !140, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 5] [public] [A]
!140 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !141, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!141 = !{null, !142, !143, !16}
!142 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1AI1DIPiEE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1AI1DIPiEE]
!143 = !{!"0x10\00\000\000\000\000\000", null, null, !"_ZTS1DIPiE"} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from _ZTS1DIPiE]
!144 = !{!145}
!145 = !{!"0x2f\00_Container\000\000", null, !"_ZTS1DIPiE", null} ; [ DW_TAG_template_type_parameter ]
!146 = !{!147, !153, !179, !183, !186, !189, !192, !195, !199, !202, !205, !211, !216, !219}
!147 = !{!"0x2e\00addressof\00addressof\00_Z9addressofR1DIPiE\0046\000\001\000\000\00256\001\0046", !5, !148, !149, null, %class.D.1.85* (%class.D.1.85*)* @_Z9addressofR1DIPiE, null, null, !151} ; [ DW_TAG_subprogram ] [line 46] [def] [addressof]
!148 = !{!"0x29", !5}                             ; [ DW_TAG_file_type ] [/test1.cpp]
!149 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !150, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!150 = !{!137, !143}
!151 = !{!152}
!152 = !{!"0x101\00p1\0016777262\000", !147, !148, !143} ; [ DW_TAG_arg_variable ] [p1] [line 46]
!153 = !{!"0x2e\00buildRegisterClasses\00buildRegisterClasses\00_ZN1J20buildRegisterClassesERi\0052\000\001\000\000\00256\001\0052", !5, !"_ZTS1J", !49, null, void (%class.J.2.86*, i32*)* @_ZN1J20buildRegisterClassesERi, null, !48, !154} ; [ DW_TAG_subprogram ] [line 52] [def] [buildRegisterClasses]
!154 = !{!155, !157, !158, !159, !160, !163, !164, !165, !167, !169, !172, !173, !174, !176, !178}
!155 = !{!"0x101\00this\0016777216\001088", !153, null, !156} ; [ DW_TAG_arg_variable ] [this] [line 0]
!156 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1J"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1J]
!157 = !{!"0x101\00\0033554484\000", !153, !148, !51} ; [ DW_TAG_arg_variable ] [line 52]
!158 = !{!"0x100\00a\0053\000", !153, !148, !"_ZTS1G"} ; [ DW_TAG_auto_variable ] [a] [line 53]
!159 = !{!"0x100\00b\0054\000", !153, !148, !"_ZTS1DIS_IPiEE"} ; [ DW_TAG_auto_variable ] [b] [line 54]
!160 = !{!"0x100\00__range\000\0064", !161, null, !162} ; [ DW_TAG_auto_variable ] [__range] [line 0]
!161 = !{!"0xb\0055\003\000", !5, !153}           ; [ DW_TAG_lexical_block ] [/test1.cpp]
!162 = !{!"0x10\00\000\000\000\000\000", null, null, !"_ZTS1G"} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from _ZTS1G]
!163 = !{!"0x100\00__begin\000\0064", !161, null, !"_ZTS1F"} ; [ DW_TAG_auto_variable ] [__begin] [line 0]
!164 = !{!"0x100\00__end\000\0064", !161, null, !"_ZTS1F"} ; [ DW_TAG_auto_variable ] [__end] [line 0]
!165 = !{!"0x100\00c\0055\000", !166, !148, !22}  ; [ DW_TAG_auto_variable ] [c] [line 55]
!166 = !{!"0xb\0055\003\001", !5, !161}           ; [ DW_TAG_lexical_block ] [/test1.cpp]
!167 = !{!"0x100\00d\0056\000", !168, !148, !"_ZTS1DIPiE"} ; [ DW_TAG_auto_variable ] [d] [line 56]
!168 = !{!"0xb\0055\0019\002", !5, !166}          ; [ DW_TAG_lexical_block ] [/test1.cpp]
!169 = !{!"0x100\00__range\000\0064", !170, null, !171} ; [ DW_TAG_auto_variable ] [__range] [line 0]
!170 = !{!"0xb\0057\005\003", !5, !168}           ; [ DW_TAG_lexical_block ] [/test1.cpp]
!171 = !{!"0x10\00\000\000\000\000\000", null, null, !"_ZTS1DIS_IPiEE"} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from _ZTS1DIS_IPiEE]
!172 = !{!"0x100\00__begin\000\0064", !170, null, !"_ZTS1BI1DIPiEE"} ; [ DW_TAG_auto_variable ] [__begin] [line 0]
!173 = !{!"0x100\00__end\000\0064", !170, null, !"_ZTS1BI1DIPiEE"} ; [ DW_TAG_auto_variable ] [__end] [line 0]
!174 = !{!"0x100\00e\0057\000", !175, !148, !"_ZTS1DIPiE"} ; [ DW_TAG_auto_variable ] [e] [line 57]
!175 = !{!"0xb\0057\005\004", !5, !170}           ; [ DW_TAG_lexical_block ] [/test1.cpp]
!176 = !{!"0x100\00f\0058\000", !177, !148, !"_ZTS1DIiE"} ; [ DW_TAG_auto_variable ] [f] [line 58]
!177 = !{!"0xb\0057\0026\005", !5, !175}          ; [ DW_TAG_lexical_block ] [/test1.cpp]
!178 = !{!"0x100\00g\0059\000", !177, !148, !"_ZTS1AI1DIPiEE"} ; [ DW_TAG_auto_variable ] [g] [line 59]
!179 = !{!"0x2e\00D\00D\00_ZN1DIS_IPiEEC1Ev\0027\000\001\000\000\00256\001\0027", !5, !"_ZTS1DIS_IPiEE", !73, null, null, null, !72, !180} ; [ DW_TAG_subprogram ] [line 27] [def] [D]
!180 = !{!181}
!181 = !{!"0x101\00this\0016777216\001088", !179, null, !182} ; [ DW_TAG_arg_variable ] [this] [line 0]
!182 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1DIS_IPiEE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1DIS_IPiEE]
!183 = !{!"0x2e\00D\00D\00_ZN1DIS_IPiEEC2Ev\0027\000\001\000\000\00256\001\0027", !5, !"_ZTS1DIS_IPiEE", !73, null, null, null, !72, !184} ; [ DW_TAG_subprogram ] [line 27] [def] [D]
!184 = !{!185}
!185 = !{!"0x101\00this\0016777216\001088", !183, null, !182} ; [ DW_TAG_arg_variable ] [this] [line 0]
!186 = !{!"0x2e\00D\00D\00_ZN1DIPiEC1Ev\0027\000\001\000\000\00256\001\0027", !5, !"_ZTS1DIPiE", !10, null, null, null, !9, !187} ; [ DW_TAG_subprogram ] [line 27] [def] [D]
!187 = !{!188}
!188 = !{!"0x101\00this\0016777216\001088", !186, null, !137} ; [ DW_TAG_arg_variable ] [this] [line 0]
!189 = !{!"0x2e\00D\00D\00_ZN1DIPiEC2Ev\0027\000\001\000\000\00256\001\0027", !5, !"_ZTS1DIPiE", !10, null, null, null, !9, !190} ; [ DW_TAG_subprogram ] [line 27] [def] [D]
!190 = !{!191}
!191 = !{!"0x101\00this\0016777216\001088", !189, null, !137} ; [ DW_TAG_arg_variable ] [this] [line 0]
!192 = !{!"0x2e\00begin\00begin\00_ZN1DIS_IPiEE5beginEv\0028\000\001\000\000\00256\001\0028", !5, !"_ZTS1DIS_IPiEE", !77, null, null, null, !76, !193} ; [ DW_TAG_subprogram ] [line 28] [def] [begin]
!193 = !{!194}
!194 = !{!"0x101\00this\0016777216\001088", !192, null, !182} ; [ DW_TAG_arg_variable ] [this] [line 0]
!195 = !{!"0x2e\00D\00D\00_ZN1DIiEC1Ev\0027\000\001\000\000\00256\001\0027", !5, !"_ZTS1DIiE", !106, null, null, null, !105, !196} ; [ DW_TAG_subprogram ] [line 27] [def] [D]
!196 = !{!197}
!197 = !{!"0x101\00this\0016777216\001088", !195, null, !198} ; [ DW_TAG_arg_variable ] [this] [line 0]
!198 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1DIiE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1DIiE]
!199 = !{!"0x2e\00D\00D\00_ZN1DIiEC2Ev\0027\000\001\000\000\00256\001\0027", !5, !"_ZTS1DIiE", !106, null, null, null, !105, !200} ; [ DW_TAG_subprogram ] [line 27] [def] [D]
!200 = !{!201}
!201 = !{!"0x101\00this\0016777216\001088", !199, null, !198} ; [ DW_TAG_arg_variable ] [this] [line 0]
!202 = !{!"0x2e\00begin\00begin\00_ZN1DIPiE5beginEv\0028\000\001\000\000\00256\001\0028", !5, !"_ZTS1DIPiE", !14, null, null, null, !13, !203} ; [ DW_TAG_subprogram ] [line 28] [def] [begin]
!203 = !{!204}
!204 = !{!"0x101\00this\0016777216\001088", !202, null, !137} ; [ DW_TAG_arg_variable ] [this] [line 0]
!205 = !{!"0x2e\00A\00A\00_ZN1AI1DIPiEEC1ERS2_1BIS1_E\005\000\001\000\000\00256\001\005", !5, !"_ZTS1AI1DIPiEE", !140, null, null, null, !139, !206} ; [ DW_TAG_subprogram ] [line 5] [def] [A]
!206 = !{!207, !209, !210}
!207 = !{!"0x101\00this\0016777216\001088", !205, null, !208} ; [ DW_TAG_arg_variable ] [this] [line 0]
!208 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1AI1DIPiEE"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1AI1DIPiEE]
!209 = !{!"0x101\00p1\0033554437\000", !205, !148, !143} ; [ DW_TAG_arg_variable ] [p1] [line 5]
!210 = !{!"0x101\00\0050331653\000", !205, !148, !16} ; [ DW_TAG_arg_variable ] [line 5]
!211 = !{!"0x2e\00A\00A\00_ZN1AI1DIPiEEC2ERS2_1BIS1_E\005\000\001\000\000\00256\001\005", !5, !"_ZTS1AI1DIPiEE", !140, null, null, null, !139, !212} ; [ DW_TAG_subprogram ] [line 5] [def] [A]
!212 = !{!213, !214, !215}
!213 = !{!"0x101\00this\0016777216\001088", !211, null, !208} ; [ DW_TAG_arg_variable ] [this] [line 0]
!214 = !{!"0x101\00p1\0033554437\000", !211, !148, !143} ; [ DW_TAG_arg_variable ] [p1] [line 5]
!215 = !{!"0x101\00\0050331653\000", !211, !148, !16} ; [ DW_TAG_arg_variable ] [line 5]
!216 = !{!"0x2e\00begin\00begin\00_ZN1DIiE5beginEv\0028\000\001\000\000\00256\001\0028", !5, !"_ZTS1DIiE", !110, null, null, null, !109, !217} ; [ DW_TAG_subprogram ] [line 28] [def] [begin]
!217 = !{!218}
!218 = !{!"0x101\00this\0016777216\001088", !216, null, !198} ; [ DW_TAG_arg_variable ] [this] [line 0]
!219 = !{!"0x2e\00set_intersection<B<int>, B<int *>, A<D<int *> > >\00set_intersection<B<int>, B<int *>, A<D<int *> > >\00_Z16set_intersectionI1BIiES0_IPiE1AI1DIS2_EEEvT_S8_T0_S9_T1_\0048\000\001\000\000\00256\001\0049", !5, !148, !220, null, null, !222, null, !226} ; [ DW_TAG_subprogram ] [line 48] [def] [scope 49] [set_intersection<B<int>, B<int *>, A<D<int *> > >]
!220 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !221, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!221 = !{null, !"_ZTS1BIiE", !"_ZTS1BIiE", !"_ZTS1BIPiE", !"_ZTS1BIPiE", !"_ZTS1AI1DIPiEE"}
!222 = !{!223, !224, !225}
!223 = !{!"0x2f\00_InputIterator1\000\000", null, !"_ZTS1BIiE", null} ; [ DW_TAG_template_type_parameter ]
!224 = !{!"0x2f\00_InputIterator2\000\000", null, !"_ZTS1BIPiE", null} ; [ DW_TAG_template_type_parameter ]
!225 = !{!"0x2f\00_OutputIterator\000\000", null, !"_ZTS1AI1DIPiEE", null} ; [ DW_TAG_template_type_parameter ]
!226 = !{!227, !228, !229, !230, !231}
!227 = !{!"0x101\00\0016777264\000", !219, !148, !"_ZTS1BIiE"} ; [ DW_TAG_arg_variable ] [line 48]
!228 = !{!"0x101\00p2\0033554480\000", !219, !148, !"_ZTS1BIiE"} ; [ DW_TAG_arg_variable ] [p2] [line 48]
!229 = !{!"0x101\00\0050331696\000", !219, !148, !"_ZTS1BIPiE"} ; [ DW_TAG_arg_variable ] [line 48]
!230 = !{!"0x101\00p4\0067108913\000", !219, !148, !"_ZTS1BIPiE"} ; [ DW_TAG_arg_variable ] [p4] [line 49]
!231 = !{!"0x101\00p5\0083886129\000", !219, !148, !"_ZTS1AI1DIPiEE"} ; [ DW_TAG_arg_variable ] [p5] [line 49]
!232 = !{i32 2, !"Dwarf Version", i32 2}
!233 = !{i32 2, !"Debug Info Version", i32 2}
!234 = !{i32 1, !"PIC Level", i32 2}
!235 = !{!"clang version 3.7.0 (trunk 226915) (llvm/trunk 226905)"}
!236 = !MDLocation(line: 55, column: 14, scope: !161)
!237 = !MDLocation(line: 27, column: 9, scope: !189, inlinedAt: !238)
!238 = distinct !MDLocation(line: 27, column: 24, scope: !186, inlinedAt: !239)
!239 = distinct !MDLocation(line: 56, column: 14, scope: !168)
!240 = !MDLocation(line: 27, column: 9, scope: !199, inlinedAt: !241)
!241 = distinct !MDLocation(line: 27, column: 24, scope: !195, inlinedAt: !242)
!242 = distinct !MDLocation(line: 58, column: 14, scope: !177)
!243 = !MDLocation(line: 57, column: 21, scope: !170)
!244 = !MDLocation(line: 60, column: 55, scope: !177)
!245 = !{!"0x101\00p5\0083886129\000", !219, !148, !"_ZTS1AI1DIPiEE", !246} ; [ DW_TAG_arg_variable ] [p5] [line 49]
!246 = distinct !MDLocation(line: 60, column: 7, scope: !177)
!247 = !{!"0x102\00147\000\008"}                  ; [ DW_TAG_expression ] [DW_OP_piece offset=0, size=8]
!248 = !MDLocation(line: 49, column: 59, scope: !219, inlinedAt: !246)
!249 = !MDLocation(line: 50, column: 3, scope: !219, inlinedAt: !246)
!250 = !MDLocation(line: 63, column: 1, scope: !153)
