; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; bool f1();
; inline __attribute__((always_inline))
; int f() {
;   if (bool b = f1())
;     return 1;
;   return 2;
; }
; 
; int main() {
;   f();
; }

; Ensure that lexical_blocks within inlined_subroutines are preserved/emitted.
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_lexical_block
; CHECK: DW_TAG_variable

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %retval.i = alloca i32, align 4
  %b.i = alloca i8, align 1
  call void @llvm.dbg.declare(metadata !{i8* %b.i}, metadata !13), !dbg !16
  %call.i = call zeroext i1 @_Z2f1v(), !dbg !16
  %frombool.i = zext i1 %call.i to i8, !dbg !16
  store i8 %frombool.i, i8* %b.i, align 1, !dbg !16
  %0 = load i8* %b.i, align 1, !dbg !16
  %tobool.i = trunc i8 %0 to i1, !dbg !16
  br i1 %tobool.i, label %if.then.i, label %if.end.i, !dbg !16

if.then.i:                                        ; preds = %entry
  store i32 1, i32* %retval.i, !dbg !18
  br label %_Z1fv.exit, !dbg !18

if.end.i:                                         ; preds = %entry
  store i32 2, i32* %retval.i, !dbg !19
  br label %_Z1fv.exit, !dbg !19

_Z1fv.exit:                                       ; preds = %if.then.i, %if.end.i
  %1 = load i32* %retval.i, !dbg !20
  ret i32 0, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare zeroext i1 @_Z2f1v() #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/inline-scopes.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"inline-scopes.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !9}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 9, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [main]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/inline-scopes.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f", metadata !"f", metadata !"_Z1fv", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [f]
!10 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!12 = metadata !{metadata !"clang version 3.5.0 "}
!13 = metadata !{i32 786688, metadata !14, metadata !"b", metadata !5, i32 4, metadata !15, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 4]
!14 = metadata !{i32 786443, metadata !1, metadata !9, i32 4, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/inline-scopes.cpp]
!15 = metadata !{i32 786468, null, null, metadata !"bool", i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!16 = metadata !{i32 4, i32 0, metadata !14, metadata !17}
!17 = metadata !{i32 10, i32 0, metadata !4, null}
!18 = metadata !{i32 5, i32 0, metadata !14, metadata !17}
!19 = metadata !{i32 6, i32 0, metadata !9, metadata !17}
!20 = metadata !{i32 7, i32 0, metadata !9, metadata !17}
!21 = metadata !{i32 11, i32 0, metadata !4, null}
