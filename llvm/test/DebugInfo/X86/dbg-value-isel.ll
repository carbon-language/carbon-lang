; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; PR 9879

; CHECK: ##DEBUG_VALUE: tid <-
%0 = type { i8*, i8*, i8*, i8*, i32 }

@sgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@fgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@lvgv = internal constant [0 x i8*] zeroinitializer
@llvm.global.annotations = appending global [1 x %0] [%0 { i8* bitcast (void (i32 addrspace(1)*)* @__OpenCL_nbt02_kernel to i8*), i8* addrspacecast ([1 x i8] addrspace(2)* @sgv to i8*), i8* addrspacecast ([1 x i8] addrspace(2)* @fgv to i8*), i8* bitcast ([0 x i8*]* @lvgv to i8*), i32 0 }], section "llvm.metadata"

define void @__OpenCL_nbt02_kernel(i32 addrspace(1)* %ip) nounwind {
entry:
  call void @llvm.dbg.value(metadata !{i32 addrspace(1)* %ip}, i64 0, metadata !8, metadata !{metadata !"0x102"}), !dbg !9
  %0 = call <4 x i32> @__amdil_get_local_id_int() nounwind
  %1 = extractelement <4 x i32> %0, i32 0
  br label %2

; <label>:2                                       ; preds = %entry
  %3 = phi i32 [ %1, %entry ]
  br label %4

; <label>:4                                       ; preds = %2
  %5 = phi i32 [ %3, %2 ]
  br label %get_local_id.exit

get_local_id.exit:                                ; preds = %4
  %6 = phi i32 [ %5, %4 ]
  call void @llvm.dbg.value(metadata !{i32 %6}, i64 0, metadata !10, metadata !{metadata !"0x102"}), !dbg !12
  %7 = call <4 x i32> @__amdil_get_global_id_int() nounwind, !dbg !12
  %8 = extractelement <4 x i32> %7, i32 0, !dbg !12
  br label %9

; <label>:9                                       ; preds = %get_local_id.exit
  %10 = phi i32 [ %8, %get_local_id.exit ]
  br label %11

; <label>:11                                      ; preds = %9
  %12 = phi i32 [ %10, %9 ]
  br label %get_global_id.exit

get_global_id.exit:                               ; preds = %11
  %13 = phi i32 [ %12, %11 ]
  call void @llvm.dbg.value(metadata !{i32 %13}, i64 0, metadata !13, metadata !{metadata !"0x102"}), !dbg !14
  %14 = call <4 x i32> @__amdil_get_local_size_int() nounwind
  %15 = extractelement <4 x i32> %14, i32 0
  br label %16

; <label>:16                                      ; preds = %get_global_id.exit
  %17 = phi i32 [ %15, %get_global_id.exit ]
  br label %18

; <label>:18                                      ; preds = %16
  %19 = phi i32 [ %17, %16 ]
  br label %get_local_size.exit

get_local_size.exit:                              ; preds = %18
  %20 = phi i32 [ %19, %18 ]
  call void @llvm.dbg.value(metadata !{i32 %20}, i64 0, metadata !15, metadata !{metadata !"0x102"}), !dbg !16
  %tmp5 = add i32 %6, %13, !dbg !17
  %tmp7 = add i32 %tmp5, %20, !dbg !17
  store i32 %tmp7, i32 addrspace(1)* %ip, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %get_local_size.exit
  ret void, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare <4 x i32> @__amdil_get_local_size_int() nounwind

declare <4 x i32> @__amdil_get_local_id_int() nounwind

declare <4 x i32> @__amdil_get_global_id_int() nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22}

!0 = metadata !{metadata !"0x2e\00__OpenCL_nbt02_kernel\00__OpenCL_nbt02_kernel\00__OpenCL_nbt02_kernel\002\000\001\000\006\000\000\000", metadata !20, metadata !1, metadata !3, null, void (i32 addrspace(1)*)* @__OpenCL_nbt02_kernel, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [__OpenCL_nbt02_kernel]
!1 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\001\00clc\000\00\000\00\001", metadata !20, metadata !21, metadata !21, metadata !19, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !20, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null, metadata !5}
!5 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !2, metadata !6} ; [ DW_TAG_pointer_type ]
!6 = metadata !{metadata !"0x16\00uint\000\000\000\000\000", metadata !20, metadata !2, metadata !7} ; [ DW_TAG_typedef ]
!7 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", null, metadata !2} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"0x101\00ip\001\000", metadata !0, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 1, i32 32, metadata !0, null}
!10 = metadata !{metadata !"0x100\00tid\003\000", metadata !11, metadata !1, metadata !6} ; [ DW_TAG_auto_variable ]
!11 = metadata !{metadata !"0xb\002\001\001", metadata !1, metadata !0} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 5, i32 24, metadata !11, null}
!13 = metadata !{metadata !"0x100\00gid\003\000", metadata !11, metadata !1, metadata !6} ; [ DW_TAG_auto_variable ]
!14 = metadata !{i32 6, i32 25, metadata !11, null}
!15 = metadata !{metadata !"0x100\00lsz\003\000", metadata !11, metadata !1, metadata !6} ; [ DW_TAG_auto_variable ]
!16 = metadata !{i32 7, i32 26, metadata !11, null}
!17 = metadata !{i32 9, i32 24, metadata !11, null}
!18 = metadata !{i32 10, i32 1, metadata !0, null}
!19 = metadata !{metadata !0}
!20 = metadata !{metadata !"OCLlLwTXZ.cl", metadata !"/tmp"}
!21 = metadata !{i32 0}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
