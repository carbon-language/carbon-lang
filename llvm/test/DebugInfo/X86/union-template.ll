; RUN: llc -O0 -mtriple=x86_64-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Verify that we've emitted template arguments for the union
; CHECK: DW_TAG_union_type
; CHECK-NEXT: "Value<float>"
; CHECK: DW_TAG_template_type_parameter
; CHECK: "T"

%"union.PR15637::Value" = type { i32 }

@_ZN7PR156371fE = global %"union.PR15637::Value" zeroinitializer, align 4

define void @_ZN7PR156371gEf(float %value) #0 {
entry:
  %value.addr = alloca float, align 4
  %tempValue = alloca %"union.PR15637::Value", align 4
  store float %value, float* %value.addr, align 4
  call void @llvm.dbg.declare(metadata !{float* %value.addr}, metadata !23, metadata !{metadata !"0x102"}), !dbg !24
  call void @llvm.dbg.declare(metadata !{%"union.PR15637::Value"* %tempValue}, metadata !25, metadata !{metadata !"0x102"}), !dbg !26
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28}

!0 = metadata !{metadata !"0x11\004\00clang version 3.3 (trunk 178499) (llvm/trunk 178472)\000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !9,  metadata !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cc", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00g\00g\00_ZN7PR156371gEf\003\000\001\000\006\00256\000\003", metadata !1, metadata !5, metadata !6, null, void (float)* @_ZN7PR156371gEf, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 3] [def] [g]
!5 = metadata !{metadata !"0x39\00PR15637\001", metadata !1, null} ; [ DW_TAG_namespace ] [PR15637] [line 1]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x34\00f\00f\00_ZN7PR156371fE\006\000\001", metadata !5, metadata !11, metadata !12, %"union.PR15637::Value"* @_ZN7PR156371fE, null} ; [ DW_TAG_variable ] [f] [line 6] [def]
!11 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/foo.cc]
!12 = metadata !{metadata !"0x17\00Value<float>\002\0032\0032\000\000\000", metadata !1, metadata !5, null, metadata !13, null, metadata !21, null} ; [ DW_TAG_union_type ] [Value<float>] [line 2, size 32, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !16}
!14 = metadata !{metadata !"0xd\00a\002\0032\0032\000\000", metadata !1, metadata !12, metadata !15} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!15 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!16 = metadata !{metadata !"0x2e\00Value\00Value\00\002\000\000\000\006\00320\000\002", metadata !1, metadata !12, metadata !17, null, null, null, i32 0, metadata !20} ; [ DW_TAG_subprogram ] [line 2] [Value]
!17 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19}
!19 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from Value<float>]
!20 = metadata !{i32 786468}
!21 = metadata !{metadata !22}
!22 = metadata !{metadata !"0x2f\00T\000\000", null, metadata !8, null} ; [ DW_TAG_template_type_parameter ]
!23 = metadata !{metadata !"0x101\00value\0016777219\000", metadata !4, metadata !11, metadata !8} ; [ DW_TAG_arg_variable ] [value] [line 3]
!24 = metadata !{i32 3, i32 0, metadata !4, null}
!25 = metadata !{metadata !"0x100\00tempValue\004\000", metadata !4, metadata !11, metadata !12} ; [ DW_TAG_auto_variable ] [tempValue] [line 4]
!26 = metadata !{i32 4, i32 0, metadata !4, null}
!27 = metadata !{i32 5, i32 0, metadata !4, null}
!28 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
