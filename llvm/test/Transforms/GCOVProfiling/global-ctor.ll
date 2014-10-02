; RUN: echo '!16 = metadata !{metadata !"%T/global-ctor.ll", metadata !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: not grep '_GLOBAL__sub_I_global-ctor' %T/global-ctor.gcno
; RUN: rm %T/global-ctor.gcno

; REQUIRES: shell

@x = global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_global-ctor.ll, i8* null }]

; Function Attrs: nounwind
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  %call = call i32 @_Z1fv(), !dbg !13
  store i32 %call, i32* @x, align 4, !dbg !13
  ret void, !dbg !13
}

declare i32 @_Z1fv() #1

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_global-ctor.ll() #0 section ".text.startup" {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  call void @__cxx_global_var_init(), !dbg !14
  ret void, !dbg !14
}

attributes #0 = { nounwind }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.gcov = !{!16}
!llvm.ident = !{!12}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (trunk 210217)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2} ; [ DW_TAG_compile_unit ] [/home/nlewycky/<stdin>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<stdin>", metadata !"/home/nlewycky"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"__cxx_global_var_init", metadata !"__cxx_global_var_init", metadata !"", i32 2, metadata !7, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @__cxx_global_var_init, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [local] [def] [__cxx_global_var_init]
!5 = metadata !{metadata !"global-ctor.ll", metadata !"/home/nlewycky"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [/home/nlewycky/global-ctor.ll]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{i32 786478, metadata !1, metadata !9, metadata !"", metadata !"", metadata !"_GLOBAL__sub_I_global-ctor.ll", i32 0, metadata !7, i1 true, i1 true, i32 0, i32 0, null, i32 64, i1 false, void ()* @_GLOBAL__sub_I_global-ctor.ll, null, null, metadata !2, i32 0} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
!9 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/home/nlewycky/<stdin>]
!10 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!11 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!12 = metadata !{metadata !"clang version 3.5.0 (trunk 210217)"}
!13 = metadata !{i32 2, i32 0, metadata !4, null}
!14 = metadata !{i32 0, i32 0, metadata !15, null}
!15 = metadata !{i32 786443, metadata !5, metadata !8} ; [ DW_TAG_lexical_block ] [/home/nlewycky/global-ctor.ll]
