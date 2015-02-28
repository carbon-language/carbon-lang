; RUN: opt -inline -S < %s | FileCheck %s
; struct A {
;   int arg0;
;   double arg1[2];
; } a, b;
;  
; void fn3(A p1) {
;   if (p1.arg0)
;     a = p1;
; }
;  
; void fn4() { fn3(b); }
;  
; void fn5() {
;   while (1)
;     fn4();
; }
; ModuleID = 'test.cpp'
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-darwin"

%struct.A = type { i32, [2 x double] }

@a = global %struct.A zeroinitializer, align 8
@b = global %struct.A zeroinitializer, align 8

; Function Attrs: nounwind
declare void @_Z3fn31A(%struct.A* nocapture readonly %p1) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind
define void @_Z3fn4v() #0 {
entry:
; Test that the dbg.declare is moved together with the alloca.
; CHECK: define void @_Z3fn5v()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg.tmp.sroa.3.i = alloca [20 x i8], align 4
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata [20 x i8]* %agg.tmp.sroa.3.i,
  %agg.tmp.sroa.3 = alloca [20 x i8], align 4
  tail call void @llvm.dbg.declare(metadata [20 x i8]* %agg.tmp.sroa.3, metadata !46, metadata !48), !dbg !49
  %agg.tmp.sroa.0.0.copyload = load i32, i32* getelementptr inbounds (%struct.A* @b, i64 0, i32 0), align 8, !dbg !50
  tail call void @llvm.dbg.value(metadata i32 %agg.tmp.sroa.0.0.copyload, i64 0, metadata !46, metadata !51), !dbg !49
  %agg.tmp.sroa.3.0..sroa_idx = getelementptr inbounds [20 x i8], [20 x i8]* %agg.tmp.sroa.3, i64 0, i64 0, !dbg !50
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.3.0..sroa_idx, i8* getelementptr (i8* bitcast (%struct.A* @b to i8*), i64 4), i64 20, i32 4, i1 false), !dbg !50
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !46, metadata !31) #2, !dbg !49
  %tobool.i = icmp eq i32 %agg.tmp.sroa.0.0.copyload, 0, !dbg !52
  br i1 %tobool.i, label %_Z3fn31A.exit, label %if.then.i, !dbg !53

if.then.i:                                        ; preds = %entry
  store i32 %agg.tmp.sroa.0.0.copyload, i32* getelementptr inbounds (%struct.A* @a, i64 0, i32 0), align 8, !dbg !54
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr (i8* bitcast (%struct.A* @a to i8*), i64 4), i8* %agg.tmp.sroa.3.0..sroa_idx, i64 20, i32 4, i1 false), !dbg !54
  br label %_Z3fn31A.exit, !dbg !54

_Z3fn31A.exit:                                    ; preds = %entry, %if.then.i
  ret void, !dbg !50
}

; Function Attrs: noreturn nounwind
define void @_Z3fn5v() #3 {
entry:
  br label %while.body, !dbg !55

while.body:                                       ; preds = %entry, %while.body
  call void @_Z3fn4v(), !dbg !56
  br label %while.body, !dbg !55
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29}
!llvm.ident = !{!30}

!0 = !{!"0x11\004\00clang version 3.7.0 (trunk 227480) (llvm/trunk 227517)\001\00\000\00\001", !1, !2, !3, !14, !25, !2} ; [ DW_TAG_compile_unit ] [/<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00A\001\00192\0064\000\000\000", !5, null, null, !6, null, null, !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 192, align 64, offset 0] [def] [from ]
!5 = !{!"test.cpp", !""}
!6 = !{!7, !9}
!7 = !{!"0xd\00arg0\002\0032\0032\000\000", !5, !"_ZTS1A", !8} ; [ DW_TAG_member ] [arg0] [line 2, size 32, align 32, offset 0] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0xd\00arg1\003\00128\0064\0064\000", !5, !"_ZTS1A", !10} ; [ DW_TAG_member ] [arg1] [line 3, size 128, align 64, offset 64] [from ]
!10 = !{!"0x1\00\000\00128\0064\000\000\000", null, null, !11, !12, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 64, offset 0] [from double]
!11 = !{!"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!12 = !{!13}
!13 = !{!"0x21\000\002"}                          ; [ DW_TAG_subrange_type ] [0, 1]
!14 = !{!15, !21, !24}
!15 = !{!"0x2e\00fn3\00fn3\00_Z3fn31A\006\000\001\000\000\00256\001\006", !5, !16, !17, null, void (%struct.A*)* @_Z3fn31A, null, null, !19} ; [ DW_TAG_subprogram ] [line 6] [def] [fn3]
!16 = !{!"0x29", !5}                              ; [ DW_TAG_file_type ] [/test.cpp]
!17 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !"_ZTS1A"}
!19 = !{!20}
!20 = !{!"0x101\00p1\0016777222\000", !15, !16, !"_ZTS1A"} ; [ DW_TAG_arg_variable ] [p1] [line 6]
!21 = !{!"0x2e\00fn4\00fn4\00_Z3fn4v\0011\000\001\000\000\00256\001\0011", !5, !16, !22, null, void ()* @_Z3fn4v, null, null, !2} ; [ DW_TAG_subprogram ] [line 11] [def] [fn4]
!22 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = !{null}
!24 = !{!"0x2e\00fn5\00fn5\00_Z3fn5v\0013\000\001\000\000\00256\001\0013", !5, !16, !22, null, void ()* @_Z3fn5v, null, null, !2} ; [ DW_TAG_subprogram ] [line 13] [def] [fn5]
!25 = !{!26, !27}
!26 = !{!"0x34\00a\00a\00\004\000\001", null, !16, !"_ZTS1A", %struct.A* @a, null} ; [ DW_TAG_variable ] [a] [line 4] [def]
!27 = !{!"0x34\00b\00b\00\004\000\001", null, !16, !"_ZTS1A", %struct.A* @b, null} ; [ DW_TAG_variable ] [b] [line 4] [def]
!28 = !{i32 2, !"Dwarf Version", i32 4}
!29 = !{i32 2, !"Debug Info Version", i32 2}
!30 = !{!"clang version 3.7.0 (trunk 227480) (llvm/trunk 227517)"}
!31 = !{!"0x102\006"}                             ; [ DW_TAG_expression ] [DW_OP_deref]
!32 = !MDLocation(line: 6, scope: !15)
!33 = !MDLocation(line: 7, scope: !34)
!34 = !{!"0xb\007\000\000", !5, !15}              ; [ DW_TAG_lexical_block ] [/test.cpp]
!35 = !{!36, !37, i64 0}
!36 = !{!"_ZTS1A", !37, i64 0, !38, i64 8}
!37 = !{!"int", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !MDLocation(line: 7, scope: !15)
!41 = !MDLocation(line: 8, scope: !34)
!42 = !{i64 0, i64 4, !43, i64 8, i64 16, !44}
!43 = !{!37, !37, i64 0}
!44 = !{!38, !38, i64 0}
!45 = !MDLocation(line: 9, scope: !15)
!46 = !{!"0x101\00p1\0016777222\000", !15, !16, !"_ZTS1A", !47} ; [ DW_TAG_arg_variable ] [p1] [line 6]
!47 = distinct !MDLocation(line: 11, scope: !21)
!48 = !{!"0x102\00157\0032\00160"}                  ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=32, size=160]
!49 = !MDLocation(line: 6, scope: !15, inlinedAt: !47)
!50 = !MDLocation(line: 11, scope: !21)
!51 = !{!"0x102\00157\000\0032"}                   ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=0, size=32]
!52 = !MDLocation(line: 7, scope: !34, inlinedAt: !47)
!53 = !MDLocation(line: 7, scope: !15, inlinedAt: !47)
!54 = !MDLocation(line: 8, scope: !34, inlinedAt: !47)
!55 = !MDLocation(line: 14, scope: !24)
!56 = !MDLocation(line: 15, scope: !24)
