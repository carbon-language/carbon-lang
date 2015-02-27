; RUN: llc -O0 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
; Test that a variable with multiple entries in the MMI table makes it into the
; debug info.
;
; CHECK: DW_TAG_inlined_subroutine
; CHECK:    "_Z3f111A"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location [DW_FORM_block1]    (<0x0b> 91 51 9d 78 08 91 4a 9d 38 88 01 )
;  -- fbreg -47, bit-piece 120 8 , fbreg -54, bit-piece 56 136 ------^
; CHECK: DW_AT_abstract_origin {{.*}} "p1"
;
; long a;
; struct A {
;   bool x4;
;   void *x5;
;   bool x6;
; };
; int *b;
; struct B {
;   B(long);
;   ~B();
; };
; void f9(A);
; void f13(A p1) {
;   b = (int *)__builtin_operator_new(a);
;   f9(p1);
; }
; void f11(A p1) { f13(p1); }
; void f16() {
;   A c;
;   B d(a);
;   c.x6 = c.x4 = true;
;   f11(c);
; }
; ModuleID = 'test.cpp'
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios"

%struct.A = type { i8, i8*, i8 }
%struct.B = type { i8 }

@a = global i64 0, align 8
@b = global i32* null, align 8

define void @_Z3f131A(%struct.A* nocapture readonly %p1) #0 {
entry:
  %agg.tmp = alloca %struct.A, align 8
  tail call void @llvm.dbg.declare(metadata %struct.A* %p1, metadata !30, metadata !46), !dbg !47
  %0 = load i64, i64* @a, align 8, !dbg !48, !tbaa !49
  %call = tail call noalias i8* @_Znwm(i64 %0) #5, !dbg !53
  store i8* %call, i8** bitcast (i32** @b to i8**), align 8, !dbg !54, !tbaa !55
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp, i64 0, i32 0, !dbg !57
  %2 = getelementptr inbounds %struct.A, %struct.A* %p1, i64 0, i32 0, !dbg !57
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 24, i32 8, i1 false), !dbg !57, !tbaa.struct !58
  call void @_Z2f91A(%struct.A* %agg.tmp), !dbg !61
  ret void, !dbg !62
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nobuiltin
declare noalias i8* @_Znwm(i64) #2

declare void @_Z2f91A(%struct.A*) #0

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #3

define void @_Z3f111A(%struct.A* nocapture readonly %p1) #0 {
entry:
  %agg.tmp.i = alloca %struct.A, align 8
  tail call void @llvm.dbg.declare(metadata %struct.A* %p1, metadata !33, metadata !46), !dbg !63
  %0 = getelementptr inbounds %struct.A, %struct.A* %p1, i64 0, i32 0, !dbg !64
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i, i64 0, i32 0, !dbg !65
  call void @llvm.lifetime.start(i64 24, i8* %1), !dbg !65
  %2 = load i64, i64* @a, align 8, !dbg !67, !tbaa !49
  %call.i = tail call noalias i8* @_Znwm(i64 %2) #5, !dbg !68
  store i8* %call.i, i8** bitcast (i32** @b to i8**), align 8, !dbg !69, !tbaa !55
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %0, i64 24, i32 8, i1 false), !dbg !70
  call void @_Z2f91A(%struct.A* %agg.tmp.i), !dbg !71
  call void @llvm.lifetime.end(i64 24, i8* %1), !dbg !72
  ret void, !dbg !73
}

define void @_Z3f16v() #0 {
entry:
  %agg.tmp.i.i = alloca %struct.A, align 8
  %d = alloca %struct.B, align 1
  %agg.tmp.sroa.2 = alloca [15 x i8], align 1
  %agg.tmp.sroa.4 = alloca [7 x i8], align 1
  tail call void @llvm.dbg.declare(metadata [15 x i8]* %agg.tmp.sroa.2, metadata !74, metadata !76), !dbg !77
  tail call void @llvm.dbg.declare(metadata [7 x i8]* %agg.tmp.sroa.4, metadata !74, metadata !78), !dbg !77
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !38, metadata !79), !dbg !80
  %0 = load i64, i64* @a, align 8, !dbg !81, !tbaa !49
  tail call void @llvm.dbg.value(metadata %struct.B* %d, i64 0, metadata !39, metadata !79), !dbg !82
  %call = call %struct.B* @_ZN1BC1El(%struct.B* %d, i64 %0), !dbg !82
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !38, metadata !83), !dbg !80
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !38, metadata !84), !dbg !80
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !74, metadata !83), !dbg !77
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !74, metadata !84), !dbg !77
  call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !74, metadata !46), !dbg !77
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i.i, i64 0, i32 0, !dbg !85
  call void @llvm.lifetime.start(i64 24, i8* %1), !dbg !85
  %2 = load i64, i64* @a, align 8, !dbg !87, !tbaa !49
  %call.i.i5 = invoke noalias i8* @_Znwm(i64 %2) #5
          to label %call.i.i.noexc unwind label %lpad, !dbg !88

call.i.i.noexc:                                   ; preds = %entry
  %agg.tmp.sroa.4.17..sroa_idx = getelementptr inbounds [7 x i8], [7 x i8]* %agg.tmp.sroa.4, i64 0, i64 0, !dbg !89
  %agg.tmp.sroa.2.1..sroa_idx = getelementptr inbounds [15 x i8], [15 x i8]* %agg.tmp.sroa.2, i64 0, i64 0, !dbg !89
  store i8* %call.i.i5, i8** bitcast (i32** @b to i8**), align 8, !dbg !90, !tbaa !55
  store i8 1, i8* %1, align 8, !dbg !91
  %agg.tmp.sroa.2.0..sroa_raw_idx = getelementptr inbounds i8, i8* %1, i64 1, !dbg !91
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.2.0..sroa_raw_idx, i8* %agg.tmp.sroa.2.1..sroa_idx, i64 15, i32 1, i1 false), !dbg !91
  %agg.tmp.sroa.3.0..sroa_idx = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i.i, i64 0, i32 2, !dbg !91
  store i8 1, i8* %agg.tmp.sroa.3.0..sroa_idx, align 8, !dbg !91
  %agg.tmp.sroa.4.0..sroa_raw_idx = getelementptr inbounds i8, i8* %1, i64 17, !dbg !91
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.4.0..sroa_raw_idx, i8* %agg.tmp.sroa.4.17..sroa_idx, i64 7, i32 1, i1 false), !dbg !91
  invoke void @_Z2f91A(%struct.A* %agg.tmp.i.i)
          to label %invoke.cont unwind label %lpad, !dbg !92

invoke.cont:                                      ; preds = %call.i.i.noexc
  call void @llvm.lifetime.end(i64 24, i8* %1), !dbg !93
  call void @llvm.dbg.value(metadata %struct.B* %d, i64 0, metadata !39, metadata !79), !dbg !82
  %call1 = call %struct.B* @_ZN1BD1Ev(%struct.B* %d) #3, !dbg !94
  ret void, !dbg !94

lpad:                                             ; preds = %call.i.i.noexc, %entry
  %3 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup, !dbg !94
  call void @llvm.dbg.value(metadata %struct.B* %d, i64 0, metadata !39, metadata !79), !dbg !82
  %call2 = call %struct.B* @_ZN1BD1Ev(%struct.B* %d) #3, !dbg !94
  resume { i8*, i32 } %3, !dbg !94
}

declare %struct.B* @_ZN1BC1El(%struct.B*, i64)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare %struct.B* @_ZN1BD1Ev(%struct.B*) #4

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #3

attributes #1 = { nounwind readnone }
attributes #2 = { nobuiltin }
attributes #3 = { nounwind }
attributes #4 = { nounwind  }
attributes #5 = { builtin }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!43, !44}
!llvm.ident = !{!45}

!0 = !{!"0x11\004\00clang version 3.7.0 \001\00\000\00\001", !1, !2, !3, !24, !40, !2} ; [ DW_TAG_compile_unit ] [/<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !""}
!2 = !{}
!3 = !{!4, !12, !14}
!4 = !{!"0x13\00A\002\00192\0064\000\000\000", !5, null, null, !6, null, null, !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 2, size 192, align 64, offset 0] [def] [from ]
!5 = !{!"test.cpp", !""}
!6 = !{!7, !9, !11}
!7 = !{!"0xd\00x4\003\008\008\000\000", !5, !"_ZTS1A", !8} ; [ DW_TAG_member ] [x4] [line 3, size 8, align 8, offset 0] [from bool]
!8 = !{!"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!9 = !{!"0xd\00x5\004\0064\0064\0064\000", !5, !"_ZTS1A", !10} ; [ DW_TAG_member ] [x5] [line 4, size 64, align 64, offset 64] [from ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!11 = !{!"0xd\00x6\005\008\008\00128\000", !5, !"_ZTS1A", !8} ; [ DW_TAG_member ] [x6] [line 5, size 8, align 8, offset 128] [from bool]
!12 = !{!"0xf\00\000\0064\0064\000\000", null, null, !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = !{!"0x13\00B\008\008\008\000\000\000", !5, null, null, !15, null, null, !"_ZTS1B"} ; [ DW_TAG_structure_type ] [B] [line 8, size 8, align 8, offset 0] [def] [from ]
!15 = !{!16, !21}
!16 = !{!"0x2e\00B\00B\00\009\000\000\000\000\00256\001\009", !5, !"_ZTS1B", !17, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 9] [B]
!17 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !19, !20}
!19 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1B]
!20 = !{!"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!21 = !{!"0x2e\00~B\00~B\00\0010\000\000\000\000\00256\001\0010", !5, !"_ZTS1B", !22, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 10] [~B]
!22 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = !{null, !19}
!24 = !{!25, !31, !34}
!25 = !{!"0x2e\00f13\00f13\00_Z3f131A\0013\000\001\000\000\00256\001\0013", !5, !26, !27, null, void (%struct.A*)* @_Z3f131A, null, null, !29} ; [ DW_TAG_subprogram ] [line 13] [def] [f13]
!26 = !{!"0x29", !5}                              ; [ DW_TAG_file_type ] [/test.cpp]
!27 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !28, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = !{null, !"_ZTS1A"}
!29 = !{!30}
!30 = !{!"0x101\00p1\0016777229\000", !25, !26, !"_ZTS1A"} ; [ DW_TAG_arg_variable ] [p1] [line 13]
!31 = !{!"0x2e\00f11\00f11\00_Z3f111A\0017\000\001\000\000\00256\001\0017", !5, !26, !27, null, void (%struct.A*)* @_Z3f111A, null, null, !32} ; [ DW_TAG_subprogram ] [line 17] [def] [f11]
!32 = !{!33}
!33 = !{!"0x101\00p1\0016777233\000", !31, !26, !"_ZTS1A"} ; [ DW_TAG_arg_variable ] [p1] [line 17]
!34 = !{!"0x2e\00f16\00f16\00_Z3f16v\0018\000\001\000\000\00256\001\0018", !5, !26, !35, null, void ()* @_Z3f16v, null, null, !37} ; [ DW_TAG_subprogram ] [line 18] [def] [f16]
!35 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !36, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!36 = !{null}
!37 = !{!38, !39}
!38 = !{!"0x100\00c\0019\000", !34, !26, !"_ZTS1A"} ; [ DW_TAG_auto_variable ] [c] [line 19]
!39 = !{!"0x100\00d\0020\000", !34, !26, !"_ZTS1B"} ; [ DW_TAG_auto_variable ] [d] [line 20]
!40 = !{!41, !42}
!41 = !{!"0x34\00a\00a\00\001\000\001", null, !26, !20, i64* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!42 = !{!"0x34\00b\00b\00\007\000\001", null, !26, !12, i32** @b, null} ; [ DW_TAG_variable ] [b] [line 7] [def]
!43 = !{i32 2, !"Dwarf Version", i32 2}
!44 = !{i32 2, !"Debug Info Version", i32 2}
!45 = !{!"clang version 3.7.0 "}
!46 = !{!"0x102\006"}                             ; [ DW_TAG_expression ] [DW_OP_deref]
!47 = !MDLocation(line: 13, column: 12, scope: !25)
!48 = !MDLocation(line: 14, column: 37, scope: !25)
!49 = !{!50, !50, i64 0}
!50 = !{!"long", !51, i64 0}
!51 = !{!"omnipotent char", !52, i64 0}
!52 = !{!"Simple C/C++ TBAA"}
!53 = !MDLocation(line: 14, column: 14, scope: !25)
!54 = !MDLocation(line: 14, column: 5, scope: !25)
!55 = !{!56, !56, i64 0}
!56 = !{!"any pointer", !51, i64 0}
!57 = !MDLocation(line: 15, column: 6, scope: !25)
!58 = !{i64 0, i64 1, !59, i64 8, i64 8, !55, i64 16, i64 1, !59}
!59 = !{!60, !60, i64 0}
!60 = !{!"bool", !51, i64 0}
!61 = !MDLocation(line: 15, column: 3, scope: !25)
!62 = !MDLocation(line: 16, column: 1, scope: !25)
!63 = !MDLocation(line: 17, column: 12, scope: !31)
!64 = !MDLocation(line: 17, column: 22, scope: !31)
!65 = !MDLocation(line: 13, column: 12, scope: !25, inlinedAt: !66)
!66 = distinct !MDLocation(line: 17, column: 18, scope: !31)
!67 = !MDLocation(line: 14, column: 37, scope: !25, inlinedAt: !66)
!68 = !MDLocation(line: 14, column: 14, scope: !25, inlinedAt: !66)
!69 = !MDLocation(line: 14, column: 5, scope: !25, inlinedAt: !66)
!70 = !MDLocation(line: 15, column: 6, scope: !25, inlinedAt: !66)
!71 = !MDLocation(line: 15, column: 3, scope: !25, inlinedAt: !66)
!72 = !MDLocation(line: 16, column: 1, scope: !25, inlinedAt: !66)
!73 = !MDLocation(line: 17, column: 27, scope: !31)
!74 = !{!"0x101\00p1\0016777233\000", !31, !26, !"_ZTS1A", !75} ; [ DW_TAG_arg_variable ] [p1] [line 17]
!75 = distinct !MDLocation(line: 22, column: 3, scope: !34)
!76 = !{!"0x102\00157\008\00120"}                 ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=8, size=120]
!77 = !MDLocation(line: 17, column: 12, scope: !31, inlinedAt: !75)
!78 = !{!"0x102\00157\00136\0056"}                ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=136, size=56]
!79 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!80 = !MDLocation(line: 19, column: 5, scope: !34)
!81 = !MDLocation(line: 20, column: 7, scope: !34)
!82 = !MDLocation(line: 20, column: 5, scope: !34)
!83 = !{!"0x102\00157\000\008"}                   ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=0, size=8]
!84 = !{!"0x102\00157\00128\008"}                 ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=128, size=8]
!85 = !MDLocation(line: 13, column: 12, scope: !25, inlinedAt: !86)
!86 = distinct !MDLocation(line: 17, column: 18, scope: !31, inlinedAt: !75)
!87 = !MDLocation(line: 14, column: 37, scope: !25, inlinedAt: !86)
!88 = !MDLocation(line: 14, column: 14, scope: !25, inlinedAt: !86)
!89 = !MDLocation(line: 22, column: 7, scope: !34)
!90 = !MDLocation(line: 14, column: 5, scope: !25, inlinedAt: !86)
!91 = !MDLocation(line: 15, column: 6, scope: !25, inlinedAt: !86)
!92 = !MDLocation(line: 15, column: 3, scope: !25, inlinedAt: !86)
!93 = !MDLocation(line: 16, column: 1, scope: !25, inlinedAt: !86)
!94 = !MDLocation(line: 23, column: 1, scope: !34)
