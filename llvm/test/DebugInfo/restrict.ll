; REQUIRES: object-emission

; RUN: %llc_dwarf -dwarf-version=2 -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=V2 %s
; RUN: %llc_dwarf -dwarf-version=3 -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=V3 %s

; CHECK: DW_AT_name {{.*}} "dst"
; V2: DW_AT_type {{.*}} {[[PTR:0x.*]]}
; V3: DW_AT_type {{.*}} {[[RESTRICT:0x.*]]}
; V3: [[RESTRICT]]: {{.*}}DW_TAG_restrict_type
; V3-NEXT: DW_AT_type {{.*}} {[[PTR:0x.*]]}
; CHECK: [[PTR]]: {{.*}}DW_TAG_pointer_type
; CHECK-NOT: DW_AT_type

; Generated with clang from:
; void foo(void* __restrict__ dst) {
; }


; Function Attrs: nounwind uwtable
define void @_Z3fooPv(i8* noalias %dst) #0 {
entry:
  %dst.addr = alloca i8*, align 8
  store i8* %dst, i8** %dst.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8** %dst.addr}, metadata !13), !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/restrict.c] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"restrict.c", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"_Z3fooPv", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8*)* @_Z3fooPv, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/restrict.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 786487, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_restrict_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!10 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!12 = metadata !{metadata !"clang version 3.5.0 "}
!13 = metadata !{i32 786689, metadata !4, metadata !"dst", metadata !5, i32 16777217, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [dst] [line 1]
!14 = metadata !{i32 1, i32 0, metadata !4, null}
!15 = metadata !{i32 2, i32 0, metadata !4, null}
