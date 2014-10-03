; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu -o /dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.function = type { i8 }
%class.BPLFunctionWriter = type { %struct.BPLModuleWriter* }
%struct.BPLModuleWriter = type { i8 }
%class.anon = type { i8 }
%class.anon.0 = type { i8 }

@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_" = internal alias void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"
@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_0EET_" = internal alias void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"

define void @_ZN17BPLFunctionWriter9writeExprEv(%class.BPLFunctionWriter* %this) nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.BPLFunctionWriter*, align 8
  %agg.tmp = alloca %class.function, align 1
  %agg.tmp2 = alloca %class.anon, align 1
  %agg.tmp4 = alloca %class.function, align 1
  %agg.tmp5 = alloca %class.anon.0, align 1
  store %class.BPLFunctionWriter* %this, %class.BPLFunctionWriter** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.BPLFunctionWriter** %this.addr}, metadata !133, metadata !{metadata !"0x102"}), !dbg !135
  %this1 = load %class.BPLFunctionWriter** %this.addr
  %MW = getelementptr inbounds %class.BPLFunctionWriter* %this1, i32 0, i32 0, !dbg !136
  %0 = load %struct.BPLModuleWriter** %MW, align 8, !dbg !136
  call void @"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"(%class.function* %agg.tmp), !dbg !136
  call void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter* %0), !dbg !136
  %MW3 = getelementptr inbounds %class.BPLFunctionWriter* %this1, i32 0, i32 0, !dbg !138
  %1 = load %struct.BPLModuleWriter** %MW3, align 8, !dbg !138
  call void @"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"(%class.function* %agg.tmp4), !dbg !138
  call void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter* %1), !dbg !138
  ret void, !dbg !139
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter*)

define internal void @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"(%class.function* %this) unnamed_addr nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.function*, align 8
  %__f = alloca %class.anon.0, align 1
  store %class.function* %this, %class.function** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.function** %this.addr}, metadata !140, metadata !{metadata !"0x102"}), !dbg !142
  call void @llvm.dbg.declare(metadata !{%class.anon.0* %__f}, metadata !143, metadata !{metadata !"0x102"}), !dbg !144
  %this1 = load %class.function** %this.addr
  call void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_"(%class.anon.0* %__f), !dbg !145
  ret void, !dbg !147
}

define internal void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_"(%class.anon.0*) nounwind uwtable align 2 {
entry:
  %.addr = alloca %class.anon.0*, align 8
  store %class.anon.0* %0, %class.anon.0** %.addr, align 8
  ret void, !dbg !148
}

define internal void @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"(%class.function* %this) unnamed_addr nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.function*, align 8
  %__f = alloca %class.anon, align 1
  store %class.function* %this, %class.function** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.function** %this.addr}, metadata !150, metadata !{metadata !"0x102"}), !dbg !151
  call void @llvm.dbg.declare(metadata !{%class.anon* %__f}, metadata !152, metadata !{metadata !"0x102"}), !dbg !153
  %this1 = load %class.function** %this.addr
  call void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_"(%class.anon* %__f), !dbg !154
  ret void, !dbg !156
}

define internal void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_"(%class.anon*) nounwind uwtable align 2 {
entry:
  %.addr = alloca %class.anon*, align 8
  store %class.anon* %0, %class.anon** %.addr, align 8
  ret void, !dbg !157
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!162}

!0 = metadata !{metadata !"0x11\004\00clang version 3.2 \000\00\000\00\000", metadata !161, metadata !1, metadata !1, metadata !3, metadata !128, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5, metadata !106, metadata !107, metadata !126, metadata !127}
!5 = metadata !{metadata !"0x2e\00writeExpr\00writeExpr\00_ZN17BPLFunctionWriter9writeExprEv\0019\000\001\000\006\00256\000\0019", metadata !6, null, metadata !7, null, void (%class.BPLFunctionWriter*)* @_ZN17BPLFunctionWriter9writeExprEv, null, metadata !103, metadata !1} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !160} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{metadata !"0x2\00BPLFunctionWriter\0015\0064\0064\000\000\000", metadata !160, null, null, metadata !11, null, null, null} ; [ DW_TAG_class_type ] [BPLFunctionWriter] [line 15, size 64, align 64, offset 0] [def] [from ]
!11 = metadata !{metadata !12, metadata !103}
!12 = metadata !{metadata !"0xd\00MW\0016\0064\0064\000\001", metadata !160, metadata !10, metadata !13} ; [ DW_TAG_member ]
!13 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !14} ; [ DW_TAG_pointer_type ]
!14 = metadata !{metadata !"0x2\00BPLModuleWriter\0012\008\008\000\000\000", metadata !160, null, null, metadata !15, null, null, null} ; [ DW_TAG_class_type ] [BPLModuleWriter] [line 12, size 8, align 8, offset 0] [def] [from ]
!15 = metadata !{metadata !16}
!16 = metadata !{metadata !"0x2e\00writeIntrinsic\00writeIntrinsic\00_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE\0013\000\000\000\006\00256\000\0013", metadata !6, metadata !14, metadata !17, null, null, null, i32 0, metadata !101} ; [ DW_TAG_subprogram ]
!17 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19, metadata !20}
!19 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !14} ; [ DW_TAG_pointer_type ]
!20 = metadata !{metadata !"0x2\00function<void ()>\006\008\008\000\000\000", metadata !160, null, null, metadata !21, null, metadata !97, null} ; [ DW_TAG_class_type ] [function<void ()>] [line 6, size 8, align 8, offset 0] [def] [from ]
!21 = metadata !{metadata !22, metadata !51, metadata !58, metadata !86, metadata !92}
!22 = metadata !{metadata !"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00\008\000\000\000\006\00256\000\008", metadata !6, metadata !20, metadata !23, null, null, metadata !47, i32 0, metadata !49} ; [ DW_TAG_subprogram ]
!23 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !24, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = metadata !{null, metadata !25, metadata !26}
!25 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !20} ; [ DW_TAG_pointer_type ]
!26 = metadata !{metadata !"0x2\00\0020\008\008\000\000\000", metadata !160, metadata !5, null, metadata !27, null, null, null} ; [ DW_TAG_class_type ] [line 20, size 8, align 8, offset 0] [def] [from ]
!27 = metadata !{metadata !28, metadata !35, metadata !41}
!28 = metadata !{metadata !"0x2e\00operator()\00operator()\00\0020\000\000\000\006\00256\000\0020", metadata !6, metadata !26, metadata !29, null, null, null, i32 0, metadata !33} ; [ DW_TAG_subprogram ]
!29 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !30, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{null, metadata !31}
!31 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !32} ; [ DW_TAG_pointer_type ]
!32 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !26} ; [ DW_TAG_const_type ]
!33 = metadata !{metadata !34}
!34 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!35 = metadata !{metadata !"0x2e\00~\00~\00\0020\000\000\000\006\00320\000\0020", metadata !6, metadata !26, metadata !36, null, null, null, i32 0, metadata !39} ; [ DW_TAG_subprogram ]
!36 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !37, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!37 = metadata !{null, metadata !38}
!38 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !26} ; [ DW_TAG_pointer_type ]
!39 = metadata !{metadata !40}
!40 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!41 = metadata !{metadata !"0x2e\00\00\00\0020\000\000\000\006\00320\000\0020", metadata !6, metadata !26, metadata !42, null, null, null, i32 0, metadata !45} ; [ DW_TAG_subprogram ]
!42 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !43, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!43 = metadata !{null, metadata !38, metadata !44}
!44 = metadata !{metadata !"0x42\00\000\000\000\000\000", null, null, metadata !26} ; [ DW_TAG_rvalue_reference_type ]
!45 = metadata !{metadata !46}
!46 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!47 = metadata !{metadata !48}
!48 = metadata !{metadata !"0x2f\00_Functor\000\000", null, metadata !26, null} ; [ DW_TAG_template_type_parameter ]
!49 = metadata !{metadata !50}
!50 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!51 = metadata !{metadata !"0x2e\00function<function<void ()> >\00function<function<void ()> >\00\008\000\000\000\006\00256\000\008", metadata !6, metadata !20, metadata !52, null, null, metadata !54, i32 0, metadata !56} ; [ DW_TAG_subprogram ]
!52 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !53, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!53 = metadata !{null, metadata !25, metadata !20}
!54 = metadata !{metadata !55}
!55 = metadata !{metadata !"0x2f\00_Functor\000\000", null, metadata !20, null} ; [ DW_TAG_template_type_parameter ]
!56 = metadata !{metadata !57}
!57 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!58 = metadata !{metadata !"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00\008\000\000\000\006\00256\000\008", metadata !6, metadata !20, metadata !59, null, null, metadata !82, i32 0, metadata !84} ; [ DW_TAG_subprogram ]
!59 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !60, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!60 = metadata !{null, metadata !25, metadata !61}
!61 = metadata !{metadata !"0x2\00\0023\008\008\000\000\000", metadata !160, metadata !5, null, metadata !62, null, null, null} ; [ DW_TAG_class_type ] [line 23, size 8, align 8, offset 0] [def] [from ]
!62 = metadata !{metadata !63, metadata !70, metadata !76}
!63 = metadata !{metadata !"0x2e\00operator()\00operator()\00\0023\000\000\000\006\00256\000\0023", metadata !6, metadata !61, metadata !64, null, null, null, i32 0, metadata !68} ; [ DW_TAG_subprogram ]
!64 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !65, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!65 = metadata !{null, metadata !66}
!66 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !67} ; [ DW_TAG_pointer_type ]
!67 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !61} ; [ DW_TAG_const_type ]
!68 = metadata !{metadata !69}
!69 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!70 = metadata !{metadata !"0x2e\00~\00~\00\0023\000\000\000\006\00320\000\0023", metadata !6, metadata !61, metadata !71, null, null, null, i32 0, metadata !74} ; [ DW_TAG_subprogram ]
!71 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !72, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!72 = metadata !{null, metadata !73}
!73 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !61} ; [ DW_TAG_pointer_type ]
!74 = metadata !{metadata !75}
!75 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!76 = metadata !{metadata !"0x2e\00\00\00\0023\000\000\000\006\00320\000\0023", metadata !6, metadata !61, metadata !77, null, null, null, i32 0, metadata !80} ; [ DW_TAG_subprogram ]
!77 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !78, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!78 = metadata !{null, metadata !73, metadata !79}
!79 = metadata !{metadata !"0x42\00\000\000\000\000\000", null, null, metadata !61} ; [ DW_TAG_rvalue_reference_type ]
!80 = metadata !{metadata !81}
!81 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!82 = metadata !{metadata !83}
!83 = metadata !{metadata !"0x2f\00_Functor\000\000", null, metadata !61, null} ; [ DW_TAG_template_type_parameter ]
!84 = metadata !{metadata !85}
!85 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!86 = metadata !{metadata !"0x2e\00function\00function\00\006\000\000\000\006\00320\000\006", metadata !6, metadata !20, metadata !87, null, null, null, i32 0, metadata !90} ; [ DW_TAG_subprogram ]
!87 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !88, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!88 = metadata !{null, metadata !25, metadata !89}
!89 = metadata !{metadata !"0x42\00\000\000\000\000\000", null, null, metadata !20} ; [ DW_TAG_rvalue_reference_type ]
!90 = metadata !{metadata !91}
!91 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!92 = metadata !{metadata !"0x2e\00~function\00~function\00\006\000\000\000\006\00320\000\006", metadata !6, metadata !20, metadata !93, null, null, null, i32 0, metadata !95} ; [ DW_TAG_subprogram ]
!93 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !94, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!94 = metadata !{null, metadata !25}
!95 = metadata !{metadata !96}
!96 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!97 = metadata !{metadata !98}
!98 = metadata !{metadata !"0x2f\00T\000\000", null, metadata !99, null} ; [ DW_TAG_template_type_parameter ]
!99 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !100, i32 0} ; [ DW_TAG_subroutine_type ]
!100 = metadata !{null}
!101 = metadata !{metadata !102}
!102 = metadata !{metadata !"0x24"}                     ; [ DW_TAG_base_type ]
!103 = metadata !{metadata !"0x2e\00writeExpr\00writeExpr\00_ZN17BPLFunctionWriter9writeExprEv\0017\000\000\000\006\00257\000\0017", metadata !6, metadata !10, metadata !7, null, null, null, i32 0, metadata !104} ; [ DW_TAG_subprogram ]
!104 = metadata !{metadata !105}
!105 = metadata !{metadata !"0x24"}                     ; [ DW_TAG_base_type ]
!106 = metadata !{metadata !"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_\008\001\001\000\006\00256\000\008", metadata !6, null, metadata !59, null, void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_", metadata !82, metadata !58, metadata !1} ; [ DW_TAG_subprogram ]
!107 = metadata !{metadata !"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_\003\001\001\000\006\00256\000\003", metadata !6, null, metadata !108, null, void (%class.anon.0*)* @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_", metadata !111, metadata !113, metadata !1} ; [ DW_TAG_subprogram ]
!108 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !109, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!109 = metadata !{null, metadata !110}
!110 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !61} ; [ DW_TAG_reference_type ]
!111 = metadata !{metadata !112}
!112 = metadata !{metadata !"0x2f\00_Tp\000\000", null, metadata !61, null} ; [ DW_TAG_template_type_parameter ]
!113 = metadata !{metadata !"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_\003\000\000\000\006\00256\000\003", metadata !6, metadata !114, metadata !108, null, null, metadata !111, i32 0, metadata !124} ; [ DW_TAG_subprogram ]
!114 = metadata !{metadata !"0x2\00_Base_manager\001\008\008\000\000\000", metadata !160, null, null, metadata !115, null, null, null} ; [ DW_TAG_class_type ] [_Base_manager] [line 1, size 8, align 8, offset 0] [def] [from ]
!115 = metadata !{metadata !116, metadata !113}
!116 = metadata !{metadata !"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_\003\000\000\000\006\00256\000\003", metadata !6, metadata !114, metadata !117, null, null, metadata !120, i32 0, metadata !122} ; [ DW_TAG_subprogram ]
!117 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !118, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!118 = metadata !{null, metadata !119}
!119 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !26} ; [ DW_TAG_reference_type ]
!120 = metadata !{metadata !121}
!121 = metadata !{metadata !"0x2f\00_Tp\000\000", null, metadata !26, null} ; [ DW_TAG_template_type_parameter ]
!122 = metadata !{metadata !123}
!123 = metadata !{metadata !"0x24"}                     ; [ DW_TAG_base_type ]
!124 = metadata !{metadata !125}
!125 = metadata !{metadata !"0x24"}                     ; [ DW_TAG_base_type ]
!126 = metadata !{metadata !"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_\008\001\001\000\006\00256\000\008", metadata !6, null, metadata !23, null, void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_", metadata !47, metadata !22, metadata !1} ; [ DW_TAG_subprogram ]
!127 = metadata !{metadata !"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_\003\001\001\000\006\00256\000\003", metadata !6, null, metadata !117, null, void (%class.anon*)* @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_", metadata !120, metadata !116, metadata !1} ; [ DW_TAG_subprogram ]
!128 = metadata !{metadata !130}
!130 = metadata !{metadata !"0x34\00__stored_locally\00__stored_locally\00__stored_locally\002\001\001", metadata !114, metadata !6, metadata !131, i1 1, null} ; [ DW_TAG_variable ]
!131 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !132} ; [ DW_TAG_const_type ]
!132 = metadata !{metadata !"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ]
!133 = metadata !{metadata !"0x101\00this\0016777235\0064", metadata !5, metadata !6, metadata !134} ; [ DW_TAG_arg_variable ]
!134 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !10} ; [ DW_TAG_pointer_type ]
!135 = metadata !{i32 19, i32 39, metadata !5, null}
!136 = metadata !{i32 20, i32 17, metadata !137, null}
!137 = metadata !{metadata !"0xb\0019\0051\000", metadata !6, metadata !5} ; [ DW_TAG_lexical_block ]
!138 = metadata !{i32 23, i32 17, metadata !137, null}
!139 = metadata !{i32 26, i32 15, metadata !137, null}
!140 = metadata !{metadata !"0x101\00this\0016777224\0064", metadata !106, metadata !6, metadata !141} ; [ DW_TAG_arg_variable ]
!141 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !20} ; [ DW_TAG_pointer_type ]
!142 = metadata !{i32 8, i32 45, metadata !106, null}
!143 = metadata !{metadata !"0x101\00__f\0033554440\000", metadata !106, metadata !6, metadata !61} ; [ DW_TAG_arg_variable ]
!144 = metadata !{i32 8, i32 63, metadata !106, null}
!145 = metadata !{i32 9, i32 9, metadata !146, null}
!146 = metadata !{metadata !"0xb\008\0081\001", metadata !6, metadata !106} ; [ DW_TAG_lexical_block ]
!147 = metadata !{i32 10, i32 13, metadata !146, null}
!148 = metadata !{i32 4, i32 5, metadata !149, null}
!149 = metadata !{metadata !"0xb\003\00105\002", metadata !6, metadata !107} ; [ DW_TAG_lexical_block ]
!150 = metadata !{metadata !"0x101\00this\0016777224\0064", metadata !126, metadata !6, metadata !141} ; [ DW_TAG_arg_variable ]
!151 = metadata !{i32 8, i32 45, metadata !126, null}
!152 = metadata !{metadata !"0x101\00__f\0033554440\000", metadata !126, metadata !6, metadata !26} ; [ DW_TAG_arg_variable ]
!153 = metadata !{i32 8, i32 63, metadata !126, null}
!154 = metadata !{i32 9, i32 9, metadata !155, null}
!155 = metadata !{metadata !"0xb\008\0081\003", metadata !6, metadata !126} ; [ DW_TAG_lexical_block ]
!156 = metadata !{i32 10, i32 13, metadata !155, null}
!157 = metadata !{i32 4, i32 5, metadata !158, null}
!158 = metadata !{metadata !"0xb\003\00105\004", metadata !6, metadata !127} ; [ DW_TAG_lexical_block ]
!159 = metadata !{metadata !"0x29", metadata !161} ; [ DW_TAG_file_type ]
!160 = metadata !{metadata !"BPLFunctionWriter2.ii", metadata !"/home/peter/crashdelta"}
!161 = metadata !{metadata !"BPLFunctionWriter.cpp", metadata !"/home/peter/crashdelta"}
!162 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
