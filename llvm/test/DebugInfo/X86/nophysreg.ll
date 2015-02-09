; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
;
; PR22296: In this testcase the DBG_VALUE describing "p5" becomes unavailable
; because the register its address is in is clobbered and we (currently) aren't
; smart enough to realize that the value is rematerialized immediately after the
; DBG_VALUE and/or is actually a stack slot.
;
; Test that we handle this situation gracefully by omitting the DW_AT_location
; and not asserting.
; Note that this check may XPASS in the future if DbgValueHistoryCalculator
; becoms smarter. That would be fine, too.
;
; CHECK: DW_TAG_subprogram
; CHECK: linkage_name{{.*}}_Z2f21A
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_AT_location
; CHECK-NEXT: DW_AT_name {{.*}}"p5"
;
; // Compile at -O1
; struct A {
;   int *m1;
;   int m2;
; };
;
; void f1(int *p1, int p2);
; void __attribute__((always_inline)) f2(A p5) { f1(p5.m1, p5.m2); }
;
; void func(void*);
; void func(const int &, const int&);
; int cond();
; void f() {
;   while (cond()) {
;     int x;
;     func(x, 0);
;     while (cond()) {
;       char y;
;       func(&y);
;       char j;
;       func(&j);
;       char I;
;       func(&I);
;       func(0, 0);
;       A g;
;       g.m1 = &x;
;       f2(g);
;     }
;   }
; }
; ModuleID = 'test.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.A = type { i32*, i32 }

; Function Attrs: alwaysinline ssp uwtable
define void @_Z2f21A(i32* %p5.coerce0, i32 %p5.coerce1) #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %p5.coerce0, i64 0, metadata !16, metadata !33), !dbg !34
  tail call void @llvm.dbg.value(metadata i32 %p5.coerce1, i64 0, metadata !16, metadata !35), !dbg !34
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !16, metadata !36), !dbg !34
  tail call void @_Z2f1Pii(i32* %p5.coerce0, i32 %p5.coerce1), !dbg !37
  ret void, !dbg !38
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z2f1Pii(i32*, i32) #2

; Function Attrs: ssp uwtable
define void @_Z1fv() #3 {
entry:
  %x = alloca i32, align 4
  %ref.tmp = alloca i32, align 4
  %y = alloca i8, align 1
  %j = alloca i8, align 1
  %I = alloca i8, align 1
  %ref.tmp5 = alloca i32, align 4
  %ref.tmp6 = alloca i32, align 4
  %call11 = call i32 @_Z4condv(), !dbg !39
  %tobool12 = icmp eq i32 %call11, 0, !dbg !39
  br i1 %tobool12, label %while.end7, label %while.body, !dbg !40

while.cond.loopexit:                              ; preds = %while.body4, %while.body
  %call = call i32 @_Z4condv(), !dbg !39
  %tobool = icmp eq i32 %call, 0, !dbg !39
  br i1 %tobool, label %while.end7, label %while.body, !dbg !40

while.body:                                       ; preds = %entry, %while.cond.loopexit
  store i32 0, i32* %ref.tmp, align 4, !dbg !41, !tbaa !42
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !21, metadata !36), !dbg !46
  call void @_Z4funcRKiS0_(i32* dereferenceable(4) %x, i32* dereferenceable(4) %ref.tmp), !dbg !47
  %call29 = call i32 @_Z4condv(), !dbg !48
  %tobool310 = icmp eq i32 %call29, 0, !dbg !48
  br i1 %tobool310, label %while.cond.loopexit, label %while.body4, !dbg !49

while.body4:                                      ; preds = %while.body, %while.body4
  call void @llvm.dbg.value(metadata i8* %y, i64 0, metadata !23, metadata !36), !dbg !50
  call void @_Z4funcPv(i8* %y), !dbg !51
  call void @llvm.dbg.value(metadata i8* %j, i64 0, metadata !26, metadata !36), !dbg !52
  call void @_Z4funcPv(i8* %j), !dbg !53
  call void @llvm.dbg.value(metadata i8* %I, i64 0, metadata !27, metadata !36), !dbg !54
  call void @_Z4funcPv(i8* %I), !dbg !55
  store i32 0, i32* %ref.tmp5, align 4, !dbg !56, !tbaa !42
  store i32 0, i32* %ref.tmp6, align 4, !dbg !57, !tbaa !42
  call void @_Z4funcRKiS0_(i32* dereferenceable(4) %ref.tmp5, i32* dereferenceable(4) %ref.tmp6), !dbg !58
  call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !28, metadata !36), !dbg !59
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !28, metadata !33), !dbg !59
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !21, metadata !36), !dbg !46
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !60, metadata !33), !dbg !62
  call void @llvm.dbg.value(metadata i32 undef, i64 0, metadata !60, metadata !35), !dbg !62
  call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !60, metadata !36), !dbg !62
  call void @_Z2f1Pii(i32* %x, i32 undef), !dbg !63
  %call2 = call i32 @_Z4condv(), !dbg !48
  %tobool3 = icmp eq i32 %call2, 0, !dbg !48
  br i1 %tobool3, label %while.cond.loopexit, label %while.body4, !dbg !49

while.end7:                                       ; preds = %while.cond.loopexit, %entry
  ret void, !dbg !64
}

declare i32 @_Z4condv()

declare void @_Z4funcRKiS0_(i32* dereferenceable(4), i32* dereferenceable(4))

declare void @_Z4funcPv(i8*)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { alwaysinline ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #3 = { ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29, !30, !31}
!llvm.ident = !{!32}

!0 = !{!"0x11\004\00clang version 3.7.0 (trunk 227088) (llvm/trunk 227091)\001\00\000\00\001", !1, !2, !3, !10, !2, !2} ; [ DW_TAG_compile_unit ] [/test.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"test.cpp", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00A\001\00128\0064\000\000\000", !1, null, null, !5, null, null, !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 128, align 64, offset 0] [def] [from ]
!5 = !{!6, !9}
!6 = !{!"0xd\00m1\002\0064\0064\000\000", !1, !"_ZTS1A", !7} ; [ DW_TAG_member ] [m1] [line 2, size 64, align 64, offset 0] [from ]
!7 = !{!"0xf\00\000\0064\0064\000\000", null, null, !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0xd\00m2\003\0032\0032\0064\000", !1, !"_ZTS1A", !8} ; [ DW_TAG_member ] [m2] [line 3, size 32, align 32, offset 64] [from int]
!10 = !{!11, !17}
!11 = !{!"0x2e\00f2\00f2\00_Z2f21A\007\000\001\000\000\00256\001\007", !1, !12, !13, null, void (i32*, i32)* @_Z2f21A, null, null, !15} ; [ DW_TAG_subprogram ] [line 7] [def] [f2]
!12 = !{!"0x29", !1}                              ; [ DW_TAG_file_type ] [/test.cpp]
!13 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{null, !"_ZTS1A"}
!15 = !{!16}
!16 = !{!"0x101\00p5\0016777223\000", !11, !12, !"_ZTS1A"} ; [ DW_TAG_arg_variable ] [p5] [line 7]
!17 = !{!"0x2e\00f\00f\00_Z1fv\0012\000\001\000\000\00256\001\0012", !1, !12, !18, null, void ()* @_Z1fv, null, null, !20} ; [ DW_TAG_subprogram ] [line 12] [def] [f]
!18 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !19, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = !{null}
!20 = !{!21, !23, !26, !27, !28}
!21 = !{!"0x100\00x\0014\000", !22, !12, !8}      ; [ DW_TAG_auto_variable ] [x] [line 14]
!22 = !{!"0xb\0013\0018\000", !1, !17}            ; [ DW_TAG_lexical_block ] [/test.cpp]
!23 = !{!"0x100\00y\0017\000", !24, !12, !25}     ; [ DW_TAG_auto_variable ] [y] [line 17]
!24 = !{!"0xb\0016\0020\001", !1, !22}            ; [ DW_TAG_lexical_block ] [/test.cpp]
!25 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!26 = !{!"0x100\00j\0019\000", !24, !12, !25}     ; [ DW_TAG_auto_variable ] [j] [line 19]
!27 = !{!"0x100\00I\0021\000", !24, !12, !25}     ; [ DW_TAG_auto_variable ] [I] [line 21]
!28 = !{!"0x100\00g\0024\000", !24, !12, !"_ZTS1A"} ; [ DW_TAG_auto_variable ] [g] [line 24]
!29 = !{i32 2, !"Dwarf Version", i32 2}
!30 = !{i32 2, !"Debug Info Version", i32 2}
!31 = !{i32 1, !"PIC Level", i32 2}
!32 = !{!"clang version 3.7.0 (trunk 227088) (llvm/trunk 227091)"}
!33 = !{!"0x102\00157\000\008"}                   ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=0, size=8]
!34 = !MDLocation(line: 7, column: 42, scope: !11)
!35 = !{!"0x102\00157\008\004"}                   ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=8, size=4]
!36 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!37 = !MDLocation(line: 7, column: 48, scope: !11)
!38 = !MDLocation(line: 7, column: 66, scope: !11)
!39 = !MDLocation(line: 13, column: 10, scope: !17)
!40 = !MDLocation(line: 13, column: 3, scope: !17)
!41 = !MDLocation(line: 15, column: 13, scope: !22)
!42 = !{!43, !43, i64 0}
!43 = !{!"int", !44, i64 0}
!44 = !{!"omnipotent char", !45, i64 0}
!45 = !{!"Simple C/C++ TBAA"}
!46 = !MDLocation(line: 14, column: 9, scope: !22)
!47 = !MDLocation(line: 15, column: 5, scope: !22)
!48 = !MDLocation(line: 16, column: 12, scope: !22)
!49 = !MDLocation(line: 16, column: 5, scope: !22)
!50 = !MDLocation(line: 17, column: 12, scope: !24)
!51 = !MDLocation(line: 18, column: 7, scope: !24)
!52 = !MDLocation(line: 19, column: 12, scope: !24)
!53 = !MDLocation(line: 20, column: 7, scope: !24)
!54 = !MDLocation(line: 21, column: 12, scope: !24)
!55 = !MDLocation(line: 22, column: 7, scope: !24)
!56 = !MDLocation(line: 23, column: 12, scope: !24)
!57 = !MDLocation(line: 23, column: 15, scope: !24)
!58 = !MDLocation(line: 23, column: 7, scope: !24)
!59 = !MDLocation(line: 24, column: 9, scope: !24)
!60 = !{!"0x101\00p5\0016777223\000", !11, !12, !"_ZTS1A", !61} ; [ DW_TAG_arg_variable ] [p5] [line 7]
!61 = distinct !MDLocation(line: 26, column: 7, scope: !24)
!62 = !MDLocation(line: 7, column: 42, scope: !11, inlinedAt: !61)
!63 = !MDLocation(line: 7, column: 48, scope: !11, inlinedAt: !61)
!64 = !MDLocation(line: 29, column: 1, scope: !17)
