; RUN: llc -mtriple=thumbv6-apple-darwin10 < %s | FileCheck %s
; RUN: opt -strip-debug < %s | llc -mtriple=thumbv6-apple-darwin10 | FileCheck %s
; Stripping out debug info formerly caused the last two multiplies to be emitted in
; the other order.  7797940 (part of it dated 6/29/2010..7/15/2010).

%0 = type { [3 x double] }

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%0*, i32, i32)* @_Z19getClosestDiagonal3ii to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define void @_Z19getClosestDiagonal3ii(%0* noalias sret, i32, i32) nounwind {
; CHECK: blx ___muldf3
; CHECK: blx ___muldf3
; CHECK: beq LBB0
; CHECK: blx ___muldf3
; <label>:3
  switch i32 %1, label %4 [
    i32 0, label %5
    i32 3, label %5
  ]

; <label>:4                                       ; preds = %3
  br label %5, !dbg !0

; <label>:5                                       ; preds = %4, %3, %3
  %storemerge = phi double [ -1.000000e+00, %4 ], [ 1.000000e+00, %3 ], [ 1.000000e+00, %3 ] ; <double> [#uses=1]
  %v_6 = icmp slt i32 %1, 2                         ; <i1> [#uses=1]
  %storemerge1 = select i1 %v_6, double 1.000000e+00, double -1.000000e+00 ; <double> [#uses=3]
  call void @llvm.dbg.value(metadata !{double %storemerge}, i64 0, metadata !91, metadata !{metadata !"0x102"}), !dbg !0
  %v_7 = icmp eq i32 %2, 1, !dbg !92                ; <i1> [#uses=1]
  %storemerge2 = select i1 %v_7, double 1.000000e+00, double -1.000000e+00 ; <double> [#uses=3]
  %v_8 = getelementptr inbounds %0* %0, i32 0, i32 0, i32 0 ; <double*> [#uses=1]
  %v_10 = getelementptr inbounds %0* %0, i32 0, i32 0, i32 2 ; <double*> [#uses=1]
  %v_11 = fmul double %storemerge1, %storemerge1, !dbg !93 ; <double> [#uses=1]
  %v_15 = tail call double @sqrt(double %v_11) nounwind readonly, !dbg !93 ; <double> [#uses=1]
  %v_16 = fdiv double 1.000000e+00, %v_15, !dbg !93   ; <double> [#uses=3]
  %v_17 = fmul double %storemerge, %v_16, !dbg !97    ; <double> [#uses=1]
  store double %v_17, double* %v_8, align 4, !dbg !97
  %v_19 = fmul double %storemerge2, %v_16, !dbg !97   ; <double> [#uses=1]
  store double %v_19, double* %v_10, align 4, !dbg !97
  ret void, !dbg !98
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare double @sqrt(double) nounwind readonly

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!104}
!0 = metadata !{i32 46, i32 0, metadata !1, null}
!1 = metadata !{metadata !"0xb\0044\000\000", metadata !101, metadata !2} ; [ DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0xb\0044\000\000", metadata !101, metadata !3} ; [ DW_TAG_lexical_block ]
!3 = metadata !{metadata !"0x2e\00getClosestDiagonal3\00getClosestDiagonal3\00_Z19getClosestDiagonal3ii\0044\000\001\000\006\000\000\000", metadata !101, null, metadata !6, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!4 = metadata !{metadata !"0x29", metadata !101} ; [ DW_TAG_file_type ]
!5 = metadata !{metadata !"0x11\004\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 00)\001\00\000\00\000", metadata !101, metadata !102, metadata !102, metadata !103, null, null} ; [ DW_TAG_compile_unit ]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !22, metadata !22}
!8 = metadata !{metadata !"0x13\00ggVector3\0066\00192\0032\000\000\000", metadata !99, null, null, metadata !10, null, null, null} ; [ DW_TAG_structure_type ] [ggVector3] [line 66, size 192, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !"0x29", metadata !"ggVector3.h", metadata !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src", metadata !5} ; [ DW_TAG_file_type ]
!99 = metadata !{metadata !"ggVector3.h", metadata !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src"}
!10 = metadata !{metadata !11, metadata !16, metadata !23, metadata !26, metadata !29, metadata !30, metadata !35, metadata !36, metadata !37, metadata !41, metadata !42, metadata !43, metadata !46, metadata !47, metadata !48, metadata !52, metadata !53, metadata !54, metadata !57, metadata !60, metadata !63, metadata !66, metadata !70, metadata !71, metadata !74, metadata !75, metadata !76, metadata !77, metadata !78, metadata !81, metadata !82, metadata !83, metadata !84, metadata !85, metadata !88, metadata !89, metadata !90}
!11 = metadata !{metadata !"0xd\00e\00160\00192\0032\000\000", metadata !99, metadata !8, metadata !12} ; [ DW_TAG_member ]
!12 = metadata !{metadata !"0x1\00\000\00192\0032\000\000", metadata !101, metadata !4, metadata !13, metadata !14, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 192, align 32, offset 0] [from double]
!13 = metadata !{metadata !"0x24\00double\000\0064\0032\000\000\004", metadata !101, metadata !4} ; [ DW_TAG_base_type ]
!14 = metadata !{metadata !15}
!15 = metadata !{metadata !"0x21\000\003"}        ; [ DW_TAG_subrange_type ]
!16 = metadata !{metadata !"0x2e\00ggVector3\00ggVector3\00\0072\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !17, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!17 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19, metadata !20}
!19 = metadata !{metadata !"0xf\00\000\0032\0032\000\0064", metadata !101, metadata !4, metadata !8} ; [ DW_TAG_pointer_type ]
!20 = metadata !{metadata !"0x16\00ggBoolean\00478\000\000\000\000", metadata !100, null, metadata !22} ; [ DW_TAG_typedef ]
!21 = metadata !{metadata !"0x29", metadata !"math.h", metadata !"/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS4.2.Internal.sdk/usr/include/architecture/arm", metadata !5} ; [ DW_TAG_file_type ]
!100 = metadata !{metadata !"math.h", metadata !"/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS4.2.Internal.sdk/usr/include/architecture/arm"}
!22 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !101, metadata !4} ; [ DW_TAG_base_type ]
!23 = metadata !{metadata !"0x2e\00ggVector3\00ggVector3\00\0073\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !24, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!24 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !25, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{null, metadata !19}
!26 = metadata !{metadata !"0x2e\00ggVector3\00ggVector3\00\0074\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !27, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!27 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !28, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{null, metadata !19, metadata !13, metadata !13, metadata !13}
!29 = metadata !{metadata !"0x2e\00Set\00Set\00_ZN9ggVector33SetEddd\0081\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !27, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!30 = metadata !{metadata !"0x2e\00x\00x\00_ZNK9ggVector31xEv\0082\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!31 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !32, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!32 = metadata !{metadata !13, metadata !33}
!33 = metadata !{metadata !"0xf\00\000\0032\0032\000\0064", metadata !101, metadata !4, metadata !34} ; [ DW_TAG_pointer_type ]
!34 = metadata !{metadata !"0x26\00\000\00192\0032\000\000", metadata !101, metadata !4, metadata !8} ; [ DW_TAG_const_type ]
!35 = metadata !{metadata !"0x2e\00y\00y\00_ZNK9ggVector31yEv\0083\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!36 = metadata !{metadata !"0x2e\00z\00z\00_ZNK9ggVector31zEv\0084\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!37 = metadata !{metadata !"0x2e\00x\00x\00_ZN9ggVector31xEv\0085\000\001\000\006\000\000\000", metadata !9, metadata !8, metadata !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!38 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !39, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!39 = metadata !{metadata !40, metadata !19}
!40 = metadata !{metadata !"0x10\00double\000\0032\0032\000\000", metadata !101, metadata !4, metadata !13} ; [ DW_TAG_reference_type ]
!41 = metadata !{metadata !"0x2e\00y\00y\00_ZN9ggVector31yEv\0086\000\001\000\006\000\000\000", metadata !9, metadata !8, metadata !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!42 = metadata !{metadata !"0x2e\00z\00z\00_ZN9ggVector31zEv\0087\000\001\000\006\000\000\000", metadata !9, metadata !8, metadata !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!43 = metadata !{metadata !"0x2e\00SetX\00SetX\00_ZN9ggVector34SetXEd\0088\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !44, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!44 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !45, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = metadata !{null, metadata !19, metadata !13}
!46 = metadata !{metadata !"0x2e\00SetY\00SetY\00_ZN9ggVector34SetYEd\0089\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !44, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!47 = metadata !{metadata !"0x2e\00SetZ\00SetZ\00_ZN9ggVector34SetZEd\0090\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !44, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!48 = metadata !{metadata !"0x2e\00ggVector3\00ggVector3\00\0092\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !49, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!49 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !50, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!50 = metadata !{null, metadata !19, metadata !51}
!51 = metadata !{metadata !"0x10\00\000\0032\0032\000\000", metadata !101, metadata !4, metadata !34} ; [ DW_TAG_reference_type ]
!52 = metadata !{metadata !"0x2e\00tolerance\00tolerance\00_ZNK9ggVector39toleranceEv\00100\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!53 = metadata !{metadata !"0x2e\00tolerance\00tolerance\00_ZN9ggVector39toleranceEv\00101\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !38, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!54 = metadata !{metadata !"0x2e\00operator+\00operator+\00_ZNK9ggVector3psEv\00107\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !55, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!55 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !56, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!56 = metadata !{metadata !51, metadata !33}
!57 = metadata !{metadata !"0x2e\00operator-\00operator-\00_ZNK9ggVector3ngEv\00108\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !58, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!58 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !59, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!59 = metadata !{metadata !8, metadata !33}
!60 = metadata !{metadata !"0x2e\00operator[]\00operator[]\00_ZNK9ggVector3ixEi\00290\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !61, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!61 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !62, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!62 = metadata !{metadata !13, metadata !33, metadata !22}
!63 = metadata !{metadata !"0x2e\00operator[]\00operator[]\00_ZN9ggVector3ixEi\00278\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !64, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!64 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !65, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!65 = metadata !{metadata !40, metadata !19, metadata !22}
!66 = metadata !{metadata !"0x2e\00operator+=\00operator+=\00_ZN9ggVector3pLERKS_\00303\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !67, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!67 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !68, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!68 = metadata !{metadata !69, metadata !19, metadata !51}
!69 = metadata !{metadata !"0x10\00ggVector3\000\0032\0032\000\000", metadata !101, metadata !4, metadata !8} ; [ DW_TAG_reference_type ]
!70 = metadata !{metadata !"0x2e\00operator-=\00operator-=\00_ZN9ggVector3mIERKS_\00310\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !67, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!71 = metadata !{metadata !"0x2e\00operator*=\00operator*=\00_ZN9ggVector3mLEd\00317\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !72, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!72 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !73, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!73 = metadata !{metadata !69, metadata !19, metadata !13}
!74 = metadata !{metadata !"0x2e\00operator/=\00operator/=\00_ZN9ggVector3dVEd\00324\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !72, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!75 = metadata !{metadata !"0x2e\00length\00length\00_ZNK9ggVector36lengthEv\00121\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!76 = metadata !{metadata !"0x2e\00squaredLength\00squaredLength\00_ZNK9ggVector313squaredLengthEv\00122\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!77 = metadata !{metadata !"0x2e\00MakeUnitVector\00MakeUnitVector\00_ZN9ggVector314MakeUnitVectorEv\00217\000\001\000\006\000\000\000", metadata !9, metadata !8, metadata !24, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!78 = metadata !{metadata !"0x2e\00Perturb\00Perturb\00_ZNK9ggVector37PerturbEdd\00126\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !79, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!79 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !80, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!80 = metadata !{metadata !8, metadata !33, metadata !13, metadata !13}
!81 = metadata !{metadata !"0x2e\00maxComponent\00maxComponent\00_ZNK9ggVector312maxComponentEv\00128\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!82 = metadata !{metadata !"0x2e\00minComponent\00minComponent\00_ZNK9ggVector312minComponentEv\00129\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!83 = metadata !{metadata !"0x2e\00maxAbsComponent\00maxAbsComponent\00_ZNK9ggVector315maxAbsComponentEv\00131\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!84 = metadata !{metadata !"0x2e\00minAbsComponent\00minAbsComponent\00_ZNK9ggVector315minAbsComponentEv\00132\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !31, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!85 = metadata !{metadata !"0x2e\00indexOfMinComponent\00indexOfMinComponent\00_ZNK9ggVector319indexOfMinComponentEv\00133\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!86 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !101, metadata !4, null, metadata !87, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!87 = metadata !{metadata !22, metadata !33}
!88 = metadata !{metadata !"0x2e\00indexOfMinAbsComponent\00indexOfMinAbsComponent\00_ZNK9ggVector322indexOfMinAbsComponentEv\00137\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!89 = metadata !{metadata !"0x2e\00indexOfMaxComponent\00indexOfMaxComponent\00_ZNK9ggVector319indexOfMaxComponentEv\00146\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!90 = metadata !{metadata !"0x2e\00indexOfMaxAbsComponent\00indexOfMaxAbsComponent\00_ZNK9ggVector322indexOfMaxAbsComponentEv\00150\000\000\000\006\000\000\000", metadata !9, metadata !8, metadata !86, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!91 = metadata !{metadata !"0x100\00vx\0046\000", metadata !1, metadata !4, metadata !13} ; [ DW_TAG_auto_variable ]
!92 = metadata !{i32 48, i32 0, metadata !1, null}
!93 = metadata !{i32 218, i32 0, metadata !94, metadata !96}
!94 = metadata !{metadata !"0xb\00217\000\000", metadata !101, metadata !95} ; [ DW_TAG_lexical_block ]
!95 = metadata !{metadata !"0xb\00217\000\000", metadata !101, metadata !77} ; [ DW_TAG_lexical_block ]
!96 = metadata !{i32 51, i32 0, metadata !1, null}
!97 = metadata !{i32 227, i32 0, metadata !94, metadata !96}
!98 = metadata !{i32 52, i32 0, metadata !1, null}
!101 = metadata !{metadata !"ggEdgeDiscrepancy.cc", metadata !"/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src"}
!102 = metadata !{i32 0}
!103 = metadata !{metadata !3, metadata !77}
!104 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
