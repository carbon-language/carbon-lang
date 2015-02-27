; RUN: opt -sroa < %s -S -o - | FileCheck %s
;
; Test that recursively splitting an alloca updates the debug info correctly.
; CHECK: %[[T:.*]] = load i64* @t, align 8
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T]], i64 0, metadata ![[Y:.*]], metadata ![[P1:.*]])
; CHECK: %[[T1:.*]] = load i64* @t, align 8
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T1]], i64 0, metadata ![[Y]], metadata ![[P2:.*]])
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T]], i64 0, metadata ![[R:.*]], metadata ![[P3:.*]])
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T1]], i64 0, metadata ![[R]], metadata ![[P4:.*]])
; CHECK: ![[P1]] = {{.*}} [DW_OP_bit_piece offset=0, size=64]
; CHECK: ![[P2]] = {{.*}} [DW_OP_bit_piece offset=64, size=64]
; CHECK: ![[P3]] = {{.*}} [DW_OP_bit_piece offset=192, size=64]
; CHECK: ![[P4]] = {{.*}} [DW_OP_bit_piece offset=256, size=64]
; 
; struct p {
;   __SIZE_TYPE__ s;
;   __SIZE_TYPE__ t;
; };
;  
; struct r {
;   int i;
;   struct p x;
;   struct p y;
; };
;  
; extern int call_me(struct r);
; extern int maybe();
; extern __SIZE_TYPE__ t;
;  
; int test() {
;   if (maybe())
;     return 0;
;   struct p y = {t, t};
;   struct r r = {.y = y};
;   return call_me(r);
; }

; ModuleID = 'pr22393.cc'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%struct.p = type { i64, i64 }
%struct.r = type { i32, %struct.p, %struct.p }

@t = external global i64

; Function Attrs: nounwind
define i32 @_Z4testv() #0 {
entry:
  %retval = alloca i32, align 4
  %y = alloca %struct.p, align 8
  %r = alloca %struct.r, align 8
  %agg.tmp = alloca %struct.r, align 8
  %call = call i32 @_Z5maybev(), !dbg !24
  %tobool = icmp ne i32 %call, 0, !dbg !24
  br i1 %tobool, label %if.then, label %if.end, !dbg !26

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, !dbg !27
  br label %return, !dbg !27

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata %struct.p* %y, metadata !28, metadata !29), !dbg !30
  %s = getelementptr inbounds %struct.p, %struct.p* %y, i32 0, i32 0, !dbg !30
  %0 = load i64* @t, align 8, !dbg !30
  store i64 %0, i64* %s, align 8, !dbg !30
  %t = getelementptr inbounds %struct.p, %struct.p* %y, i32 0, i32 1, !dbg !30
  %1 = load i64* @t, align 8, !dbg !30
  store i64 %1, i64* %t, align 8, !dbg !30
  call void @llvm.dbg.declare(metadata %struct.r* %r, metadata !31, metadata !29), !dbg !32
  %i = getelementptr inbounds %struct.r, %struct.r* %r, i32 0, i32 0, !dbg !32
  store i32 0, i32* %i, align 4, !dbg !32
  %x = getelementptr inbounds %struct.r, %struct.r* %r, i32 0, i32 1, !dbg !32
  %s1 = getelementptr inbounds %struct.p, %struct.p* %x, i32 0, i32 0, !dbg !32
  store i64 0, i64* %s1, align 8, !dbg !32
  %t2 = getelementptr inbounds %struct.p, %struct.p* %x, i32 0, i32 1, !dbg !32
  store i64 0, i64* %t2, align 8, !dbg !32
  %y3 = getelementptr inbounds %struct.r, %struct.r* %r, i32 0, i32 2, !dbg !32
  %2 = bitcast %struct.p* %y3 to i8*, !dbg !32
  %3 = bitcast %struct.p* %y to i8*, !dbg !32
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %3, i64 16, i32 8, i1 false), !dbg !32
  %4 = bitcast %struct.r* %agg.tmp to i8*, !dbg !33
  %5 = bitcast %struct.r* %r to i8*, !dbg !33
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %4, i8* %5, i64 40, i32 8, i1 false), !dbg !33
  %call4 = call i32 @_Z7call_me1r(%struct.r* byval align 8 %agg.tmp), !dbg !33
  store i32 %call4, i32* %retval, !dbg !33
  br label %return, !dbg !33

return:                                           ; preds = %if.end, %if.then
  %6 = load i32* %retval, !dbg !34
  ret i32 %6, !dbg !34
}

declare i32 @_Z5maybev()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #3

declare i32 @_Z7call_me1r(%struct.r* byval align 8)

attributes #0 = { nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = !{!"0x11\004\00clang version 3.7.0 \000\00\000\00\001", !1, !2, !3, !16, !2, !2} ; [ DW_TAG_compile_unit ] [/<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !""}
!2 = !{}
!3 = !{!4, !10}
!4 = !{!"0x13\00p\003\00128\0064\000\000\000", !5, null, null, !6, null, null, !"_ZTS1p"} ; [ DW_TAG_structure_type ] [p] [line 3, size 128, align 64, offset 0] [def] [from ]
!5 = !{!"pr22393.cc", !""}
!6 = !{!7, !9}
!7 = !{!"0xd\00s\004\0064\0064\000\000", !5, !"_ZTS1p", !8} ; [ DW_TAG_member ] [s] [line 4, size 64, align 64, offset 0] [from long unsigned int]
!8 = !{!"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!9 = !{!"0xd\00t\005\0064\0064\0064\000", !5, !"_ZTS1p", !8} ; [ DW_TAG_member ] [t] [line 5, size 64, align 64, offset 64] [from long unsigned int]
!10 = !{!"0x13\00r\008\00320\0064\000\000\000", !5, null, null, !11, null, null, !"_ZTS1r"} ; [ DW_TAG_structure_type ] [r] [line 8, size 320, align 64, offset 0] [def] [from ]
!11 = !{!12, !14, !15}
!12 = !{!"0xd\00i\009\0032\0032\000\000", !5, !"_ZTS1r", !13} ; [ DW_TAG_member ] [i] [line 9, size 32, align 32, offset 0] [from int]
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = !{!"0xd\00x\0010\00128\0064\0064\000", !5, !"_ZTS1r", !"_ZTS1p"} ; [ DW_TAG_member ] [x] [line 10, size 128, align 64, offset 64] [from _ZTS1p]
!15 = !{!"0xd\00y\0011\00128\0064\00192\000", !5, !"_ZTS1r", !"_ZTS1p"} ; [ DW_TAG_member ] [y] [line 11, size 128, align 64, offset 192] [from _ZTS1p]
!16 = !{!17}
!17 = !{!"0x2e\00test\00test\00_Z4testv\0018\000\001\000\000\00256\000\0018", !5, !18, !19, null, i32 ()* @_Z4testv, null, null, !2} ; [ DW_TAG_subprogram ] [line 18] [def] [test]
!18 = !{!"0x29", !5}                              ; [ DW_TAG_file_type ] [/pr22393.cc]
!19 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !20, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = !{!13}
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 2}
!23 = !{!"clang version 3.7.0 "}
!24 = !MDLocation(line: 19, scope: !25)
!25 = !{!"0xb\0019\000\000", !5, !17}             ; [ DW_TAG_lexical_block ] [/pr22393.cc]
!26 = !MDLocation(line: 19, scope: !17)
!27 = !MDLocation(line: 20, scope: !25)
!28 = !{!"0x100\00y\0021\000", !17, !18, !"_ZTS1p"} ; [ DW_TAG_auto_variable ] [y] [line 21]
!29 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!30 = !MDLocation(line: 21, scope: !17)
!31 = !{!"0x100\00r\0022\000", !17, !18, !"_ZTS1r"} ; [ DW_TAG_auto_variable ] [r] [line 22]
!32 = !MDLocation(line: 22, scope: !17)
!33 = !MDLocation(line: 23, scope: !17)
!34 = !MDLocation(line: 24, scope: !17)
