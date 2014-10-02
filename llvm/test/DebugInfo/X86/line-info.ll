; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: [[FILEID:[0-9]+]]]{{.*}}list0.h
; CHECK: [[FILEID]]      0      1   0  0 is_stmt{{$}}

; IR generated from clang -g -emit-llvm with the following source:
; list0.h:
; int foo (int x) {
;     return ++x;
; }
; list0.c:
; #include "list0.h"
; int main() {
; }

define i32 @foo(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !14, metadata !{metadata !"0x102"}), !dbg !15
  %0 = load i32* %x.addr, align 4, !dbg !16
  %inc = add nsw i32 %0, 1, !dbg !16
  store i32 %inc, i32* %x.addr, align 4, !dbg !16
  ret i32 %inc, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define i32 @main() #0 {
entry:
  ret i32 0, !dbg !17
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.3 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2,  metadata !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/list0.c] [DW_LANG_C99]
!1 = metadata !{metadata !"list0.c", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !10}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !5, metadata !6, metadata !7, null, i32 (i32)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"./list0.h", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!6 = metadata !{metadata !"0x29", metadata !5}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/./list0.h]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !"0x2e\00main\00main\00\002\000\001\000\006\000\000\002", metadata !1, metadata !11, metadata !12, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 2] [def] [main]
!11 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/list0.c]
!12 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{metadata !9}
!14 = metadata !{metadata !"0x101\00x\0016777217\000", metadata !4, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ] [x] [line 1]
!15 = metadata !{i32 1, i32 0, metadata !4, null}
!16 = metadata !{i32 2, i32 0, metadata !4, null}
!17 = metadata !{i32 3, i32 0, metadata !18, null}
!18 = metadata !{metadata !"0xb\000", metadata !11, metadata !10} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/blaikie/dev/scratch/list0.c]
!19 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
