; RUN: llc -O0 < %s | FileCheck %s
; Radar 9223880
; CHECK:         .loc    1 17 64
; CHECK:        movl    $0, %esi

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%class.LanguageRuntime = type opaque
%class.Process = type { i8 }

define zeroext i1 @_Z15SetDynamicValuev() uwtable ssp {
entry:
  %retval = alloca i1, align 1
  %process = alloca %class.Process*, align 8
  %cpp_runtime = alloca %class.LanguageRuntime*, align 8
  %objc_runtime = alloca %class.LanguageRuntime*, align 8
  %call = call zeroext i1 @_Z24IsPointerOrReferenceTypev(), !dbg !15
  br i1 %call, label %if.end, label %if.then, !dbg !15

if.then:                                          ; preds = %entry
  store i1 false, i1* %retval, !dbg !17
  br label %return, !dbg !17

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata !{%class.Process** %process}, metadata !18), !dbg !20
  %call1 = call %class.Process* @_Z10GetProcessv(), !dbg !21
  store %class.Process* %call1, %class.Process** %process, align 8, !dbg !21
  %tmp = load %class.Process** %process, align 8, !dbg !22
  %tobool = icmp ne %class.Process* %tmp, null, !dbg !22
  br i1 %tobool, label %if.end3, label %if.then2, !dbg !22

if.then2:                                         ; preds = %if.end
  store i1 false, i1* %retval, !dbg !23
  br label %return, !dbg !23

if.end3:                                          ; preds = %if.end
  call void @llvm.dbg.declare(metadata !{%class.LanguageRuntime** %cpp_runtime}, metadata !24), !dbg !25
  %tmp5 = load %class.Process** %process, align 8, !dbg !26
  %call6 = call %class.LanguageRuntime* @_ZN7Process18GetLanguageRuntimeEi(%class.Process* %tmp5, i32 0), !dbg !26
  store %class.LanguageRuntime* %call6, %class.LanguageRuntime** %cpp_runtime, align 8, !dbg !26
  %tmp7 = load %class.LanguageRuntime** %cpp_runtime, align 8, !dbg !27
  %tobool8 = icmp ne %class.LanguageRuntime* %tmp7, null, !dbg !27
  br i1 %tobool8, label %if.then9, label %if.end10, !dbg !27

if.then9:                                         ; preds = %if.end3
  store i1 true, i1* %retval, !dbg !28
  br label %return, !dbg !28

if.end10:                                         ; preds = %if.end3
  call void @llvm.dbg.declare(metadata !{%class.LanguageRuntime** %objc_runtime}, metadata !30), !dbg !31
  %tmp12 = load %class.Process** %process, align 8, !dbg !32
  %call13 = call %class.LanguageRuntime* @_ZN7Process18GetLanguageRuntimeEi(%class.Process* %tmp12, i32 1), !dbg !32
  store %class.LanguageRuntime* %call13, %class.LanguageRuntime** %objc_runtime, align 8, !dbg !32
  %tmp14 = load %class.LanguageRuntime** %objc_runtime, align 8, !dbg !33
  %tobool15 = icmp ne %class.LanguageRuntime* %tmp14, null, !dbg !33
  br i1 %tobool15, label %if.then16, label %if.end17, !dbg !33

if.then16:                                        ; preds = %if.end10
  store i1 true, i1* %retval, !dbg !34
  br label %return, !dbg !34

if.end17:                                         ; preds = %if.end10
  store i1 false, i1* %retval, !dbg !36
  br label %return, !dbg !36

return:                                           ; preds = %if.end17, %if.then16, %if.then9, %if.then2, %if.then
  %0 = load i1* %retval, !dbg !37
  ret i1 %0, !dbg !37
}

declare zeroext i1 @_Z24IsPointerOrReferenceTypev()

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare %class.Process* @_Z10GetProcessv()

declare %class.LanguageRuntime* @_ZN7Process18GetLanguageRuntimeEi(%class.Process*, i32)

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1, !6}

!0 = metadata !{i32 589841, i32 0, i32 4, metadata !"my_vo.cpp", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 133629)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589870, i32 0, metadata !2, metadata !"SetDynamicValue", metadata !"SetDynamicValue", metadata !"_Z15SetDynamicValuev", metadata !2, i32 9, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i1 ()* @_Z15SetDynamicValuev, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 589865, metadata !"my_vo.cpp", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 589845, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !0, metadata !"bool", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 589870, i32 0, metadata !7, metadata !"GetLanguageRuntime", metadata !"GetLanguageRuntime", metadata !"_ZN7Process18GetLanguageRuntimeEi", metadata !2, i32 4, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 589826, metadata !0, metadata !"Process", metadata !2, i32 2, i64 8, i64 8, i32 0, i32 0, null, metadata !8, i32 0, null, null} ; [ DW_TAG_class_type ]
!8 = metadata !{metadata !6}
!9 = metadata !{i32 589845, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !10, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!10 = metadata !{metadata !11, metadata !13, metadata !14}
!11 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 589843, metadata !0, metadata !"LanguageRuntime", metadata !2, i32 1, i64 0, i64 0, i32 0, i32 4, i32 0, null, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!13 = metadata !{i32 589839, metadata !0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !7} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 589860, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!15 = metadata !{i32 10, i32 3, metadata !16, null}
!16 = metadata !{i32 589835, metadata !1, i32 9, i32 24, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!17 = metadata !{i32 11, i32 5, metadata !16, null}
!18 = metadata !{i32 590080, metadata !16, metadata !"process", metadata !2, i32 13, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!19 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !7} ; [ DW_TAG_pointer_type ]
!20 = metadata !{i32 13, i32 12, metadata !16, null}
!21 = metadata !{i32 13, i32 34, metadata !16, null}
!22 = metadata !{i32 14, i32 3, metadata !16, null}
!23 = metadata !{i32 15, i32 5, metadata !16, null}
!24 = metadata !{i32 590080, metadata !16, metadata !"cpp_runtime", metadata !2, i32 17, metadata !11, i32 0} ; [ DW_TAG_auto_variable ]
!25 = metadata !{i32 17, i32 20, metadata !16, null}
!26 = metadata !{i32 17, i32 64, metadata !16, null}
!27 = metadata !{i32 18, i32 3, metadata !16, null}
!28 = metadata !{i32 19, i32 5, metadata !29, null}
!29 = metadata !{i32 589835, metadata !16, i32 18, i32 20, metadata !2, i32 1} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 590080, metadata !16, metadata !"objc_runtime", metadata !2, i32 22, metadata !11, i32 0} ; [ DW_TAG_auto_variable ]
!31 = metadata !{i32 22, i32 20, metadata !16, null}
!32 = metadata !{i32 22, i32 65, metadata !16, null}
!33 = metadata !{i32 23, i32 3, metadata !16, null}
!34 = metadata !{i32 24, i32 5, metadata !35, null}
!35 = metadata !{i32 589835, metadata !16, i32 23, i32 21, metadata !2, i32 2} ; [ DW_TAG_lexical_block ]
!36 = metadata !{i32 26, i32 3, metadata !16, null}
!37 = metadata !{i32 27, i32 1, metadata !16, null}
