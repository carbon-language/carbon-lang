; RUN: true

; ModuleID = 'bar.cpp'

%struct.Base = type { i32 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1gi(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %struct.Base, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !18, metadata !{metadata !"0x102"}), !dbg !19
  call void @llvm.dbg.declare(metadata !{%struct.Base* %t}, metadata !20, metadata !{metadata !"0x102"}), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: ssp uwtable
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  call void @_Z1fi(i32 0), !dbg !23
  call void @_Z1gi(i32 1), !dbg !24
  ret i32 0, !dbg !25
}

declare void @_Z1fi(i32) #3

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !26}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 (http://llvm.org/git/clang.git c23b1db6268c8e7ce64026d57d1510c1aac200a0) (http://llvm.org/git/llvm.git 09b98fe3978eddefc2145adc1056cf21580ce945)\000\00\000\00\000", metadata !1, metadata !2, metadata !3, metadata !9, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/Users/mren/c_testing/type_unique_air/simple/bar.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"bar.cpp", metadata !"/Users/mren/c_testing/type_unique_air/simple"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00Base\001\0032\0032\000\000\000", metadata !5, null, null, metadata !6, null, null, metadata !"_ZTS4Base"} ; [ DW_TAG_structure_type ] [Base] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"./a.hpp", metadata !"/Users/mren/c_testing/type_unique_air/simple"}
!6 = metadata !{metadata !7}
!7 = metadata !{metadata !"0xd\00a\002\0032\0032\000\000", metadata !5, metadata !"_ZTS4Base", metadata !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10, metadata !14}
!10 = metadata !{metadata !"0x2e\00g\00g\00_Z1gi\004\000\001\000\006\00256\000\004", metadata !1, metadata !11, metadata !12, null, void (i32)* @_Z1gi, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 4] [def] [g]
!11 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/Users/mren/c_testing/type_unique_air/simple/bar.cpp]
!12 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{null, metadata !8}
!14 = metadata !{metadata !"0x2e\00main\00main\00\007\000\001\000\006\00256\000\007", metadata !1, metadata !11, metadata !15, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !8}
!17 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!18 = metadata !{metadata !"0x101\00a\0016777220\000", metadata !10, metadata !11, metadata !8} ; [ DW_TAG_arg_variable ] [a] [line 4]
!19 = metadata !{i32 4, i32 0, metadata !10, null}
!20 = metadata !{metadata !"0x100\00t\005\000", metadata !10, metadata !11, metadata !4} ; [ DW_TAG_auto_variable ] [t] [line 5]
!21 = metadata !{i32 5, i32 0, metadata !10, null}
!22 = metadata !{i32 6, i32 0, metadata !10, null}
!23 = metadata !{i32 8, i32 0, metadata !14, null}
!24 = metadata !{i32 9, i32 0, metadata !14, null}
!25 = metadata !{i32 10, i32 0, metadata !14, null}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
