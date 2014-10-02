; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; rdar://13071590
; Check we are not emitting mutliple AT_const_value for a single member.
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_member
; CHECK: badbit
; CHECK: DW_AT_const_value [DW_FORM_sdata]      (1)
; CHECK-NOT: DW_AT_const_value
; CHECK: NULL

%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_os" }
%"class.std::basic_os" = type { %"class.std::os_base", %"class.std::basic_ostream"*, i8, i8 }
%"class.std::os_base" = type { i32 (...)**, i64, i64, i32, i32, i32 }

@_ZSt4cout = external global %"class.std::basic_ostream"
@.str = private unnamed_addr constant [6 x i8] c"c is \00", align 1

define i32 @main() {
entry:
  %call1.i = tail call %"class.std::basic_ostream"* @test(%"class.std::basic_ostream"* @_ZSt4cout, i8* getelementptr inbounds ([6 x i8]* @.str, i64 0, i64 0), i64 5)
  ret i32 0
}

declare %"class.std::basic_ostream"* @test(%"class.std::basic_ostream"*, i8*, i64)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1803}

!0 = metadata !{metadata !"0x11\004\00clang version 3.3 (trunk 174207)\001\00\000\00\000", metadata !1802, metadata !1, metadata !955, metadata !956, metadata !1786,  metadata !955} ; [ DW_TAG_compile_unit ] [/privite/tmp/student2.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !26}
!4 = metadata !{metadata !"0x39\00std\0048", null, metadata !5} ; [ DW_TAG_namespace ]
!5 = metadata !{metadata !"0x29", metadata !1801} ; [ DW_TAG_file_type ]
!25 = metadata !{metadata !"0x28\00_S_os_fmtflags_end\0065536"} ; [ DW_TAG_enumerator ]
!26 = metadata !{metadata !"0x4\00_Ios_Iostate\00146\0032\0032\000\000\000", metadata !1801, metadata !4, null, metadata !27, null, null, null} ; [ DW_TAG_enumeration_type ] [_Ios_Iostate] [line 146, size 32, align 32, offset 0] [def] [from ]
!27 = metadata !{metadata !28, metadata !29, metadata !30, metadata !31, metadata !32}
!28 = metadata !{metadata !"0x28\00_S_goodbit\000"} ; [ DW_TAG_enumerator ] [_S_goodbit :: 0]
!29 = metadata !{metadata !"0x28\00_S_badbit\001"} ; [ DW_TAG_enumerator ] [_S_badbit :: 1]
!30 = metadata !{metadata !"0x28\00_S_eofbit\002"} ; [ DW_TAG_enumerator ] [_S_eofbit :: 2]
!31 = metadata !{metadata !"0x28\00_S_failbit\004"} ; [ DW_TAG_enumerator ] [_S_failbit :: 4]
!32 = metadata !{metadata !"0x28\00_S_os_ostate_end\0065536"} ; [ DW_TAG_enumerator ] [_S_os_ostate_end :: 65536]
!49 = metadata !{metadata !"0x2\00os_base\00200\001728\0064\000\000\000", metadata !1801, metadata !4, null, metadata !50, metadata !49, null, null} ; [ DW_TAG_class_type ] [os_base] [line 200, size 1728, align 64, offset 0] [def] [from ]
!50 = metadata !{metadata !77}
!54 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !55, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!55 = metadata !{metadata !56}
!56 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!77 = metadata !{metadata !"0xd\00badbit\00331\000\000\000\004096", metadata !1801, metadata !49, metadata !78, i32 1} ; [ DW_TAG_member ]
!78 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !79} ; [ DW_TAG_const_type ]
!79 = metadata !{metadata !"0x16\00ostate\00327\000\000\000\000", metadata !1801, metadata !49, metadata !26} ; [ DW_TAG_typedef ]
!955 = metadata !{}
!956 = metadata !{metadata !960}
!960 = metadata !{metadata !"0x2e\00main\00main\00\0073\000\001\000\006\00256\001\0073", metadata !1802, null, metadata !54, null, i32 ()* @main, null, null, metadata !955} ; [ DW_TAG_subprogram ]
!961 = metadata !{metadata !"0x29", metadata !1802} ; [ DW_TAG_file_type ]
!1786 = metadata !{metadata !1800}
!1800 = metadata !{metadata !"0x34\00badbit\00badbit\00badbit\00331\001\001", metadata !5, metadata !5, metadata !78, i32 1, metadata !77} ; [ DW_TAG_variable ]
!1801 = metadata !{metadata !"os_base.h", metadata !"/privite/tmp"}
!1802 = metadata !{metadata !"student2.cpp", metadata !"/privite/tmp"}
!1803 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
