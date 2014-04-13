; RUN: llc -mtriple=x86_64-linux-gnu -fast-isel=false -filetype=obj < %s -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -fast-isel=false -filetype=asm < %s -o - | FileCheck --check-prefix=ASM %s

; Generated from:
; clang-tot -c -S -emit-llvm -g inline-seldag-test.c
; inline int __attribute__((always_inline)) f(int y) {
;   return y ? 4 : 7;
; }
; void func() {
;   volatile int x;
;   x = f(x);
; }

; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} {[[F:0x.*]]}
; CHECK: [[F]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_name {{.*}} "f"

; Make sure the condition test is attributed to the inline function
; ASM: # inline-seldag-test.c:2:0
; ASM-NOT: .loc
; ASM: testl

; Function Attrs: nounwind uwtable
define void @func() #0 {
entry:
  %y.addr.i = alloca i32, align 4
  %x = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x}, metadata !15), !dbg !17
  %0 = load volatile i32* %x, align 4, !dbg !18
  store i32 %0, i32* %y.addr.i, align 4
  call void @llvm.dbg.declare(metadata !{i32* %y.addr.i}, metadata !19), !dbg !20
  %1 = load i32* %y.addr.i, align 4, !dbg !21
  %tobool.i = icmp ne i32 %1, 0, !dbg !21
  %cond.i = select i1 %tobool.i, i32 4, i32 7, !dbg !21
  store volatile i32 %cond.i, i32* %x, align 4, !dbg !18
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/inline-seldag-test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"inline-seldag-test.c", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func", metadata !"func", metadata !"", i32 4, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @func, null, null, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [func]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/inline-seldag-test.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f", metadata !"f", metadata !"", i32 1, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{metadata !11, metadata !11}
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!14 = metadata !{metadata !"clang version 3.5.0 "}
!15 = metadata !{i32 786688, metadata !4, metadata !"x", metadata !5, i32 5, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [x] [line 5]
!16 = metadata !{i32 786485, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_volatile_type ] [line 0, size 0, align 0, offset 0] [from int]
!17 = metadata !{i32 5, i32 0, metadata !4, null}
!18 = metadata !{i32 6, i32 7, metadata !4, null}
!19 = metadata !{i32 786689, metadata !8, metadata !"y", metadata !5, i32 16777217, metadata !11, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [y] [line 1]
!20 = metadata !{i32 1, i32 0, metadata !8, metadata !18}
!21 = metadata !{i32 2, i32 0, metadata !8, metadata !18}
!22 = metadata !{i32 7, i32 0, metadata !4, null}
