; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; bool f();
; inline __attribute__((always_inline)) int f1() {
;   if (bool b = f())
;     return 1;
;   return 2;
; }
;
; inline __attribute__((always_inline)) int f2() {
; # 2 "y.cc"
;   if (bool b = f())
;     return 3;
;   return 4;
; }
;
; int main() {
;   f1();
;   f2();
; }

; Ensure that lexical_blocks within inlined_subroutines are preserved/emitted.
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK-NOT: NULL
; CHECK: DW_TAG_lexical_block
; CHECK-NOT: DW_TAG
; CHECK-NOT: NULL
; CHECK: DW_TAG_variable
; Ensure that file changes don't interfere with creating inlined subroutines.
; (see the line directive inside 'f2' in thesource)
; CHECK: DW_TAG_inlined_subroutine
; CHECK:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %retval.i2 = alloca i32, align 4
  %b.i3 = alloca i8, align 1
  %retval.i = alloca i32, align 4
  %b.i = alloca i8, align 1
  call void @llvm.dbg.declare(metadata !{i8* %b.i}, metadata !16, metadata !{metadata !"0x102"}), !dbg !19
  %call.i = call zeroext i1 @_Z1fv(), !dbg !19
  %frombool.i = zext i1 %call.i to i8, !dbg !19
  store i8 %frombool.i, i8* %b.i, align 1, !dbg !19
  %0 = load i8* %b.i, align 1, !dbg !19
  %tobool.i = trunc i8 %0 to i1, !dbg !19
  br i1 %tobool.i, label %if.then.i, label %if.end.i, !dbg !19

if.then.i:                                        ; preds = %entry
  store i32 1, i32* %retval.i, !dbg !21
  br label %_Z2f1v.exit, !dbg !21

if.end.i:                                         ; preds = %entry
  store i32 2, i32* %retval.i, !dbg !22
  br label %_Z2f1v.exit, !dbg !22

_Z2f1v.exit:                                      ; preds = %if.then.i, %if.end.i
  %1 = load i32* %retval.i, !dbg !23
  call void @llvm.dbg.declare(metadata !{i8* %b.i3}, metadata !24, metadata !{metadata !"0x102"}), !dbg !27
  %call.i4 = call zeroext i1 @_Z1fv(), !dbg !27
  %frombool.i5 = zext i1 %call.i4 to i8, !dbg !27
  store i8 %frombool.i5, i8* %b.i3, align 1, !dbg !27
  %2 = load i8* %b.i3, align 1, !dbg !27
  %tobool.i6 = trunc i8 %2 to i1, !dbg !27
  br i1 %tobool.i6, label %if.then.i7, label %if.end.i8, !dbg !27

if.then.i7:                                       ; preds = %_Z2f1v.exit
  store i32 3, i32* %retval.i2, !dbg !29
  br label %_Z2f2v.exit, !dbg !29

if.end.i8:                                        ; preds = %_Z2f1v.exit
  store i32 4, i32* %retval.i2, !dbg !30
  br label %_Z2f2v.exit, !dbg !30

_Z2f2v.exit:                                      ; preds = %if.then.i7, %if.end.i8
  %3 = load i32* %retval.i2, !dbg !31
  ret i32 0, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare zeroext i1 @_Z1fv() #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/inline-scopes.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"inline-scopes.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !10, metadata !12}
!4 = metadata !{metadata !"0x2e\00main\00main\00\007\000\001\000\006\00256\000\007", metadata !5, metadata !6, metadata !7, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!5 = metadata !{metadata !"y.cc", metadata !"/tmp/dbginfo"}
!6 = metadata !{metadata !"0x29", metadata !5}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/y.cc]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !"0x2e\00f2\00f2\00_Z2f2v\008\000\001\000\006\00256\000\008", metadata !1, metadata !11, metadata !7, null, null, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 8] [def] [f2]
!11 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/inline-scopes.cpp]
!12 = metadata !{metadata !"0x2e\00f1\00f1\00_Z2f1v\002\000\001\000\006\00256\000\002", metadata !1, metadata !11, metadata !7, null, null, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 2] [def] [f1]
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!15 = metadata !{metadata !"clang version 3.5.0 "}
!16 = metadata !{metadata !"0x100\00b\003\000", metadata !17, metadata !11, metadata !18} ; [ DW_TAG_auto_variable ] [b] [line 3]
!17 = metadata !{metadata !"0xb\003\000\001", metadata !1, metadata !12} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/inline-scopes.cpp]
!18 = metadata !{metadata !"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!19 = metadata !{i32 3, i32 0, metadata !17, metadata !20}
!20 = metadata !{i32 8, i32 0, metadata !4, null}
!21 = metadata !{i32 4, i32 0, metadata !17, metadata !20}
!22 = metadata !{i32 5, i32 0, metadata !12, metadata !20}
!23 = metadata !{i32 6, i32 0, metadata !12, metadata !20}
!24 = metadata !{metadata !"0x100\00b\002\000", metadata !25, metadata !6, metadata !18} ; [ DW_TAG_auto_variable ] [b] [line 2]
!25 = metadata !{metadata !"0xb\002\000\000", metadata !5, metadata !26} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/y.cc]
!26 = metadata !{metadata !"0xb\000", metadata !5, metadata !10} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/y.cc]
!27 = metadata !{i32 2, i32 0, metadata !25, metadata !28}
!28 = metadata !{i32 9, i32 0, metadata !4, null}
!29 = metadata !{i32 3, i32 0, metadata !25, metadata !28}
!30 = metadata !{i32 4, i32 0, metadata !26, metadata !28}
!31 = metadata !{i32 5, i32 0, metadata !26, metadata !28}
!32 = metadata !{i32 10, i32 0, metadata !4, null}
