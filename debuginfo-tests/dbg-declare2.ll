; This test case checks handling of llvm.dbg.declare intrinsic during isel.
; RUN: %clang -arch x86_64 -mllvm -fast-isel=false -mllvm -regalloc=default -g %s -c -o %t.o
; RUN: %clang -arch x86_64 %t.o -o %t.out
; RUN: %test_debuginfo %s %t.out
; XFAIL: *
; XTARGET: darwin

target triple = "x86_64-apple-darwin"
%struct.XYZ = type { i32, i32, i32, i32, i32 }

; Check handling of llvm.dbg.declare for an argument referred through alloca, where
; alloca dominates llvm.dbg.declare
define i32 @f1(i32 %i) nounwind ssp {
; DEBUGGER: break f1
; DEBUGGER: r
; DEBUGGER: p i
; CHECK: $1 = 42
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !16), !dbg !17
  %tmp = load i32* %i.addr, align 4, !dbg !18
  ret i32 %tmp, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

; Check handling of llvm.dbg.declare for an argument referred through alloca, where
; llvm.dbg.declare dominates alloca.
define i32 @f2(i32 %i) nounwind ssp {
; DEBUGGER: break f2
; DEBUGGER: c
; DEBUGGER: p i
; CHECK: $2 = 43
entry:
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !20), !dbg !21
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %tmp = load i32* %i.addr, align 4, !dbg !22
  ret i32 %tmp, !dbg !22
}

; Check handling of an argument referred directly by llvm.dbg.declare where at least
; one argument use dominates llvm.dbg.declare.
; This is expected to not work because registor allocator has freedom to kill 'i'
; after its last use.
define i32 @f3(i32 %i) nounwind ssp {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32 %i}, metadata !24), !dbg !25
  %tmp = load i32* %i.addr, align 4, !dbg !26
  ret i32 %tmp, !dbg !26
}

; Check handling of an argument referred directly by llvm.dbg.declare where 
; llvm.dbg.declare dominates all uses of argument.
define i32 @f4(i32 %i) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !{i32 %i}, metadata !28), !dbg !29
  ret i32 %i, !dbg !30
}

; Check handling of an argument referred directly by llvm.dbg.declare where 
; llvm.dbg.declare dominates all uses of argument in separate basic block.
define i32 @f5(i32 %i) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !{i32 %i}, metadata !32), !dbg !33
  br label %bbr
bbr:
  ret i32 %i, !dbg !34
}

; Check handling of an argument referred directly by llvm.dbg.declare where 
; argument is not used.
define i32 @f6(i32 %i) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !{i32 %i}, metadata !36), !dbg !37
  ret i32 1, !dbg !38
}

; Check handling of an byval argument referred directly by llvm.dbg.declare where 
; argument is not used.
define i32 @f7(%struct.XYZ* byval %i) nounwind ssp {
; DEBUGGER: break f7
; DEBUGGER: c
; DEBUGGER: p i.x
; CHECK: $3 = 51
entry:
  call void @llvm.dbg.declare(metadata !{%struct.XYZ* %i}, metadata !40), !dbg !48
  ret i32 1, !dbg !49
}

; Check handling of an byval argument referred directly by llvm.dbg.declare where 
; argument use dominates llvm.dbg.declare.
define i32 @f8(%struct.XYZ* byval %i) nounwind ssp {
; DEBUGGER: break f8
; DEBUGGER: c
; DEBUGGER: p i.x
; CHECK: $4 = 51
entry:
  %tmp = getelementptr inbounds %struct.XYZ* %i, i32 0, i32 1, !dbg !53
  %tmp1 = load i32* %tmp, align 4, !dbg !53
  call void @llvm.dbg.declare(metadata !{%struct.XYZ* %i}, metadata !51), !dbg !52
  ret i32 %tmp1, !dbg !53
}

; Check handling of an byval argument referred directly by llvm.dbg.declare where 
; llvm.dbg.declare dominates all uses of argument.
define i32 @f9(%struct.XYZ* byval %i) nounwind ssp {
; DEBUGGER: break f9
; DEBUGGER: c
; DEBUGGER: p i.x
; CHECK: $5 = 51
entry:
  call void @llvm.dbg.declare(metadata !{%struct.XYZ* %i}, metadata !55), !dbg !56
  %tmp = getelementptr inbounds %struct.XYZ* %i, i32 0, i32 2, !dbg !57
  %tmp1 = load i32* %tmp, align 4, !dbg !57
  ret i32 %tmp1, !dbg !57
}

; Check handling of an byval argument referred directly by llvm.dbg.declare where 
; llvm.dbg.declare dominates all uses of argument in separate basic block.
define i32 @f10(%struct.XYZ* byval %i) nounwind ssp {
; DEBUGGER: break f10
; DEBUGGER: c
; DEBUGGER: p i.x
; CHECK: $6 = 51
entry:
  call void @llvm.dbg.declare(metadata !{%struct.XYZ* %i}, metadata !59), !dbg !60
  br label %bbr
bbr:
  %tmp = getelementptr inbounds %struct.XYZ* %i, i32 0, i32 3, !dbg !61
  %tmp1 = load i32* %tmp, align 4, !dbg !61
  ret i32 %tmp1, !dbg !61
}

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  %abc = alloca %struct.XYZ, align 4
  %agg.tmp = alloca %struct.XYZ, align 4
  %agg.tmp13 = alloca %struct.XYZ, align 4
  %agg.tmp17 = alloca %struct.XYZ, align 4
  %agg.tmp21 = alloca %struct.XYZ, align 4
  store i32 0, i32* %retval
  %call = call i32 @f1(i32 42), !dbg !63
  %call1 = call i32 @f2(i32 43), !dbg !65
  %call2 = call i32 @f3(i32 44), !dbg !66
  %call3 = call i32 @f4(i32 45), !dbg !67
  %call4 = call i32 @f5(i32 46), !dbg !68
  %call5 = call i32 @f6(i32 47), !dbg !69
  call void @llvm.dbg.declare(metadata !{%struct.XYZ* %abc}, metadata !70), !dbg !71
  %tmp = getelementptr inbounds %struct.XYZ* %abc, i32 0, i32 0, !dbg !72
  store i32 51, i32* %tmp, align 4, !dbg !72
  %tmp6 = getelementptr inbounds %struct.XYZ* %abc, i32 0, i32 1, !dbg !72
  store i32 52, i32* %tmp6, align 4, !dbg !72
  %tmp7 = getelementptr inbounds %struct.XYZ* %abc, i32 0, i32 2, !dbg !72
  store i32 53, i32* %tmp7, align 4, !dbg !72
  %tmp8 = getelementptr inbounds %struct.XYZ* %abc, i32 0, i32 3, !dbg !72
  store i32 54, i32* %tmp8, align 4, !dbg !72
  %tmp9 = getelementptr inbounds %struct.XYZ* %abc, i32 0, i32 4, !dbg !72
  store i32 55, i32* %tmp9, align 4, !dbg !72
  %tmp10 = bitcast %struct.XYZ* %agg.tmp to i8*, !dbg !73
  %tmp11 = bitcast %struct.XYZ* %abc to i8*, !dbg !73
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp10, i8* %tmp11, i64 20, i32 4, i1 false), !dbg !73
  %call12 = call i32 @f7(%struct.XYZ* byval %agg.tmp), !dbg !73
  %tmp14 = bitcast %struct.XYZ* %agg.tmp13 to i8*, !dbg !74
  %tmp15 = bitcast %struct.XYZ* %abc to i8*, !dbg !74
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp14, i8* %tmp15, i64 20, i32 4, i1 false), !dbg !74
  %call16 = call i32 @f8(%struct.XYZ* byval %agg.tmp13), !dbg !74
  %tmp18 = bitcast %struct.XYZ* %agg.tmp17 to i8*, !dbg !75
  %tmp19 = bitcast %struct.XYZ* %abc to i8*, !dbg !75
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp18, i8* %tmp19, i64 20, i32 4, i1 false), !dbg !75
  %call20 = call i32 @f9(%struct.XYZ* byval %agg.tmp17), !dbg !75
  %tmp22 = bitcast %struct.XYZ* %agg.tmp21 to i8*, !dbg !76
  %tmp23 = bitcast %struct.XYZ* %abc to i8*, !dbg !76
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp22, i8* %tmp23, i64 20, i32 4, i1 false), !dbg !76
  %call24 = call i32 @f10(%struct.XYZ* byval %agg.tmp21), !dbg !76
  ret i32 0, !dbg !77
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

!llvm.dbg.sp = !{!0, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f1", metadata !"f1", metadata !"f1", metadata !1, i32 11, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32)* @f1} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524329, metadata !"/Users/manav/dbg_info_bugs/fastisel_arg.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 524305, i32 0, i32 12, metadata !"/Users/manav/dbg_info_bugs/fastisel_arg.c", metadata !"/private/tmp", metadata !"clang version 2.8 (trunk 112967)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 524309, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 524324, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f2", metadata !"f2", metadata !"f2", metadata !1, i32 12, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32)* @f2} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f3", metadata !"f3", metadata !"f3", metadata !1, i32 13, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32)* @f3} ; [ DW_TAG_subprogram ]
!8 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f4", metadata !"f4", metadata !"f4", metadata !1, i32 14, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32)* @f4} ; [ DW_TAG_subprogram ]
!9 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f5", metadata !"f5", metadata !"f5", metadata !1, i32 15, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32)* @f5} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f6", metadata !"f6", metadata !"f6", metadata !1, i32 16, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32)* @f6} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f7", metadata !"f7", metadata !"f7", metadata !1, i32 17, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (%struct.XYZ*)* @f7} ; [ DW_TAG_subprogram ]
!12 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f8", metadata !"f8", metadata !"f8", metadata !1, i32 18, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (%struct.XYZ*)* @f8} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f9", metadata !"f9", metadata !"f9", metadata !1, i32 19, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (%struct.XYZ*)* @f9} ; [ DW_TAG_subprogram ]
!14 = metadata !{i32 524334, i32 0, metadata !1, metadata !"f10", metadata !"f10", metadata !"f10", metadata !1, i32 20, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (%struct.XYZ*)* @f10} ; [ DW_TAG_subprogram ]
!15 = metadata !{i32 524334, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 23, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @main} ; [ DW_TAG_subprogram ]
!16 = metadata !{i32 524545, metadata !0, metadata !"i", metadata !1, i32 11, metadata !5} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 11, i32 12, metadata !0, null}
!18 = metadata !{i32 11, i32 17, metadata !19, null}
!19 = metadata !{i32 524299, metadata !0, i32 11, i32 15, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!20 = metadata !{i32 524545, metadata !6, metadata !"i", metadata !1, i32 12, metadata !5} ; [ DW_TAG_arg_variable ]
!21 = metadata !{i32 12, i32 12, metadata !6, null}
!22 = metadata !{i32 12, i32 17, metadata !23, null}
!23 = metadata !{i32 524299, metadata !6, i32 12, i32 15, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!24 = metadata !{i32 524545, metadata !7, metadata !"i", metadata !1, i32 13, metadata !5} ; [ DW_TAG_arg_variable ]
!25 = metadata !{i32 13, i32 12, metadata !7, null}
!26 = metadata !{i32 13, i32 17, metadata !27, null}
!27 = metadata !{i32 524299, metadata !7, i32 13, i32 15, metadata !1, i32 2} ; [ DW_TAG_lexical_block ]
!28 = metadata !{i32 524545, metadata !8, metadata !"i", metadata !1, i32 14, metadata !5} ; [ DW_TAG_arg_variable ]
!29 = metadata !{i32 14, i32 12, metadata !8, null}
!30 = metadata !{i32 14, i32 17, metadata !31, null}
!31 = metadata !{i32 524299, metadata !8, i32 14, i32 15, metadata !1, i32 3} ; [ DW_TAG_lexical_block ]
!32 = metadata !{i32 524545, metadata !9, metadata !"i", metadata !1, i32 15, metadata !5} ; [ DW_TAG_arg_variable ]
!33 = metadata !{i32 15, i32 12, metadata !9, null}
!34 = metadata !{i32 15, i32 17, metadata !35, null}
!35 = metadata !{i32 524299, metadata !9, i32 15, i32 15, metadata !1, i32 4} ; [ DW_TAG_lexical_block ]
!36 = metadata !{i32 524545, metadata !10, metadata !"i", metadata !1, i32 16, metadata !5} ; [ DW_TAG_arg_variable ]
!37 = metadata !{i32 16, i32 12, metadata !10, null}
!38 = metadata !{i32 16, i32 17, metadata !39, null}
!39 = metadata !{i32 524299, metadata !10, i32 16, i32 15, metadata !1, i32 5} ; [ DW_TAG_lexical_block ]
!40 = metadata !{i32 524545, metadata !11, metadata !"i", metadata !1, i32 17, metadata !41} ; [ DW_TAG_arg_variable ]
!41 = metadata !{i32 524307, metadata !1, metadata !"XYZ", metadata !1, i32 2, i64 160, i64 32, i64 0, i32 0, null, metadata !42, i32 0, null} ; [ DW_TAG_structure_type ]
!42 = metadata !{metadata !43, metadata !44, metadata !45, metadata !46, metadata !47}
!43 = metadata !{i32 524301, metadata !1, metadata !"x", metadata !1, i32 3, i64 32, i64 32, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!44 = metadata !{i32 524301, metadata !1, metadata !"y", metadata !1, i32 4, i64 32, i64 32, i64 32, i32 0, metadata !5} ; [ DW_TAG_member ]
!45 = metadata !{i32 524301, metadata !1, metadata !"z", metadata !1, i32 5, i64 32, i64 32, i64 64, i32 0, metadata !5} ; [ DW_TAG_member ]
!46 = metadata !{i32 524301, metadata !1, metadata !"a", metadata !1, i32 6, i64 32, i64 32, i64 96, i32 0, metadata !5} ; [ DW_TAG_member ]
!47 = metadata !{i32 524301, metadata !1, metadata !"b", metadata !1, i32 7, i64 32, i64 32, i64 128, i32 0, metadata !5} ; [ DW_TAG_member ]
!48 = metadata !{i32 17, i32 19, metadata !11, null}
!49 = metadata !{i32 17, i32 24, metadata !50, null}
!50 = metadata !{i32 524299, metadata !11, i32 17, i32 22, metadata !1, i32 6} ; [ DW_TAG_lexical_block ]
!51 = metadata !{i32 524545, metadata !12, metadata !"i", metadata !1, i32 18, metadata !41} ; [ DW_TAG_arg_variable ]
!52 = metadata !{i32 18, i32 19, metadata !12, null}
!53 = metadata !{i32 18, i32 24, metadata !54, null}
!54 = metadata !{i32 524299, metadata !12, i32 18, i32 22, metadata !1, i32 7} ; [ DW_TAG_lexical_block ]
!55 = metadata !{i32 524545, metadata !13, metadata !"i", metadata !1, i32 19, metadata !41} ; [ DW_TAG_arg_variable ]
!56 = metadata !{i32 19, i32 19, metadata !13, null}
!57 = metadata !{i32 19, i32 24, metadata !58, null}
!58 = metadata !{i32 524299, metadata !13, i32 19, i32 22, metadata !1, i32 8} ; [ DW_TAG_lexical_block ]
!59 = metadata !{i32 524545, metadata !14, metadata !"i", metadata !1, i32 20, metadata !41} ; [ DW_TAG_arg_variable ]
!60 = metadata !{i32 20, i32 20, metadata !14, null}
!61 = metadata !{i32 20, i32 25, metadata !62, null}
!62 = metadata !{i32 524299, metadata !14, i32 20, i32 23, metadata !1, i32 9} ; [ DW_TAG_lexical_block ]
!63 = metadata !{i32 24, i32 3, metadata !64, null}
!64 = metadata !{i32 524299, metadata !15, i32 23, i32 12, metadata !1, i32 10} ; [ DW_TAG_lexical_block ]
!65 = metadata !{i32 25, i32 3, metadata !64, null}
!66 = metadata !{i32 26, i32 3, metadata !64, null}
!67 = metadata !{i32 27, i32 3, metadata !64, null}
!68 = metadata !{i32 28, i32 3, metadata !64, null}
!69 = metadata !{i32 29, i32 3, metadata !64, null}
!70 = metadata !{i32 524544, metadata !64, metadata !"abc", metadata !1, i32 30, metadata !41} ; [ DW_TAG_auto_variable ]
!71 = metadata !{i32 30, i32 14, metadata !64, null}
!72 = metadata !{i32 30, i32 17, metadata !64, null}
!73 = metadata !{i32 31, i32 3, metadata !64, null}
!74 = metadata !{i32 32, i32 3, metadata !64, null}
!75 = metadata !{i32 33, i32 3, metadata !64, null}
!76 = metadata !{i32 34, i32 3, metadata !64, null}
!77 = metadata !{i32 36, i32 3, metadata !64, null}
