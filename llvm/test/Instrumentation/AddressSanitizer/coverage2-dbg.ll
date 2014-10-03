; Test that coverage instrumentation does not lose debug location.

; RUN: opt < %s -asan -asan-module -asan-coverage=2 -S | FileCheck %s

; C++ source:
; 1: void foo(int *a) {
; 2:     if (a)
; 3:         *a = 0;
; 4: }
; clang++ if.cc -O3 -g -S -emit-llvm
; and add sanitize_address to @_Z3fooPi


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check that __sanitizer_cov call has !dgb pointing to the beginning
; of appropriate basic blocks.
; CHECK-LABEL:_Z3fooPi
; CHECK: call void @__sanitizer_cov(), !dbg [[A:!.*]]
; CHECK: call void @__sanitizer_cov(), !dbg [[B:!.*]]
; CHECK: call void @__sanitizer_cov(), !dbg [[C:!.*]]
; CHECK: ret void
; CHECK: [[A]] = metadata !{i32 1, i32 0, metadata !{{.*}}, null}
; CHECK: [[B]] = metadata !{i32 3, i32 5, metadata !{{.*}}, null}
; CHECK: [[C]] = metadata !{i32 4, i32 1, metadata !{{.*}}, null}

define void @_Z3fooPi(i32* %a) #0 {
entry:
  tail call void @llvm.dbg.value(metadata !{i32* %a}, i64 0, metadata !11, metadata !{metadata !"0x102"}), !dbg !15
  %tobool = icmp eq i32* %a, null, !dbg !16
  br i1 %tobool, label %if.end, label %if.then, !dbg !16

if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4, !dbg !18, !tbaa !19
  br label %if.end, !dbg !18

if.end:                                           ; preds = %entry, %if.then
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" sanitize_address}
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 (217079)\001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [FOO/if.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"if.cc", metadata !"FOO"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3fooPi\001\000\001\000\006\00256\001\001", metadata !1, metadata !5, metadata !6, null, void (i32*)* @_Z3fooPi, null, null, metadata !10} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [FOO/if.cc]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"0x101\00a\0016777217\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [a] [line 1]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!13 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!14 = metadata !{metadata !"clang version 3.6.0 (217079)"}
!15 = metadata !{i32 1, i32 15, metadata !4, null}
!16 = metadata !{i32 2, i32 7, metadata !17, null}
!17 = metadata !{metadata !"0xb\002\007\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [FOO/if.cc]
!18 = metadata !{i32 3, i32 5, metadata !17, null}
!19 = metadata !{metadata !20, metadata !20, i64 0}
!20 = metadata !{metadata !"int", metadata !21, i64 0}
!21 = metadata !{metadata !"omnipotent char", metadata !22, i64 0}
!22 = metadata !{metadata !"Simple C/C++ TBAA"}
!23 = metadata !{i32 4, i32 1, metadata !4, null}
