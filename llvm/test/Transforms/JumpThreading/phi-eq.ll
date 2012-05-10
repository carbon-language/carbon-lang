; RUN: llvm-as < %s | opt -jump-threading | llvm-dis | FileCheck %s
; Test whether two consecutive switches with identical structures assign the
; proper value to the proper variable.  This is really testing 
; Instruction::isIdenticalToWhenDefined, as previously that function was 
; returning true if the value part of the operands of two phis were identical, 
; even if the incoming blocks were not.
; NB: this function should be pruned down more.

%struct._GList = type { i8*, %struct._GList*, %struct._GList* }
%struct.filter_def = type { i8*, i8* }

@capture_filters = external hidden global %struct._GList*, align 8
@display_filters = external hidden global %struct._GList*, align 8
@.str2 = external hidden unnamed_addr constant [10 x i8], align 1
@__PRETTY_FUNCTION__.copy_filter_list = external hidden unnamed_addr constant [62 x i8], align 1
@.str12 = external hidden unnamed_addr constant [22 x i8], align 1
@.str13 = external hidden unnamed_addr constant [31 x i8], align 1
@capture_edited_filters = external hidden global %struct._GList*, align 8
@display_edited_filters = external hidden global %struct._GList*, align 8
@__PRETTY_FUNCTION__.get_filter_list = external hidden unnamed_addr constant [44 x i8], align 1

declare void @g_assertion_message(i8*, i8*, i32, i8*, i8*) noreturn

declare void @g_free(i8*)

declare %struct._GList* @g_list_first(%struct._GList*)

declare noalias i8* @g_malloc(i64)

define void @copy_filter_list(i32 %dest_type, i32 %src_type) nounwind uwtable ssp {
entry:
  call void @llvm.dbg.value(metadata !{i32 %dest_type}, i64 0, metadata !89), !dbg !90
  call void @llvm.dbg.value(metadata !{i32 %src_type}, i64 0, metadata !91), !dbg !92
  br label %do.body, !dbg !93

do.body:                                          ; preds = %entry
  %cmp = icmp ne i32 %dest_type, %src_type, !dbg !95
  br i1 %cmp, label %if.then, label %if.else, !dbg !95

if.then:                                          ; preds = %do.body
  br label %if.end, !dbg !95

if.else:                                          ; preds = %do.body
  call void @g_assertion_message_expr(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 581, i8* getelementptr inbounds ([62 x i8]* @__PRETTY_FUNCTION__.copy_filter_list, i32 0, i32 0), i8* getelementptr inbounds ([22 x i8]* @.str12, i32 0, i32 0)) noreturn, !dbg !95
  unreachable, !dbg !95

if.end:                                           ; preds = %if.then
  br label %do.end, !dbg !95

do.end:                                           ; preds = %if.end
  call void @llvm.dbg.value(metadata !{i32 %dest_type}, i64 0, metadata !97) nounwind, !dbg !99
  switch i32 %dest_type, label %sw.default.i [
    i32 0, label %sw.bb.i
    i32 1, label %sw.bb1.i
    i32 2, label %sw.bb2.i
    i32 3, label %sw.bb3.i
  ], !dbg !100

sw.bb.i:                                          ; preds = %do.end
  call void @llvm.dbg.value(metadata !102, i64 0, metadata !103) nounwind, !dbg !104
  br label %get_filter_list.exit, !dbg !106

sw.bb1.i:                                         ; preds = %do.end
  call void @llvm.dbg.value(metadata !107, i64 0, metadata !103) nounwind, !dbg !108
  br label %get_filter_list.exit, !dbg !109

sw.bb2.i:                                         ; preds = %do.end
  call void @llvm.dbg.value(metadata !110, i64 0, metadata !103) nounwind, !dbg !111
  br label %get_filter_list.exit, !dbg !112

sw.bb3.i:                                         ; preds = %do.end
  call void @llvm.dbg.value(metadata !113, i64 0, metadata !103) nounwind, !dbg !114
  br label %get_filter_list.exit, !dbg !115

sw.default.i:                                     ; preds = %do.end
  call void @g_assertion_message(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 408, i8* getelementptr inbounds ([44 x i8]* @__PRETTY_FUNCTION__.get_filter_list, i32 0, i32 0), i8* null) noreturn nounwind, !dbg !116
  unreachable, !dbg !116

get_filter_list.exit:                             ; preds = %sw.bb3.i, %sw.bb2.i, %sw.bb1.i, %sw.bb.i
  %0 = phi %struct._GList** [ @display_edited_filters, %sw.bb3.i ], [ @capture_edited_filters, %sw.bb2.i ], [ @display_filters, %sw.bb1.i ], [ @capture_filters, %sw.bb.i ]
  call void @llvm.dbg.value(metadata !{%struct._GList** %0}, i64 0, metadata !118), !dbg !98
  call void @llvm.dbg.value(metadata !{i32 %src_type}, i64 0, metadata !119) nounwind, !dbg !121
  switch i32 %src_type, label %sw.default.i5 [
    i32 0, label %sw.bb.i1
    i32 1, label %sw.bb1.i2
    i32 2, label %sw.bb2.i3
    i32 3, label %sw.bb3.i4
  ], !dbg !122

sw.bb.i1:                                         ; preds = %get_filter_list.exit
  call void @llvm.dbg.value(metadata !102, i64 0, metadata !123) nounwind, !dbg !124
  br label %get_filter_list.exit6, !dbg !125

sw.bb1.i2:                                        ; preds = %get_filter_list.exit
  call void @llvm.dbg.value(metadata !107, i64 0, metadata !123) nounwind, !dbg !126
  br label %get_filter_list.exit6, !dbg !127

sw.bb2.i3:                                        ; preds = %get_filter_list.exit
  call void @llvm.dbg.value(metadata !110, i64 0, metadata !123) nounwind, !dbg !128
  br label %get_filter_list.exit6, !dbg !129

sw.bb3.i4:                                        ; preds = %get_filter_list.exit
  call void @llvm.dbg.value(metadata !113, i64 0, metadata !123) nounwind, !dbg !130
  br label %get_filter_list.exit6, !dbg !131

sw.default.i5:                                    ; preds = %get_filter_list.exit
  call void @g_assertion_message(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 408, i8* getelementptr inbounds ([44 x i8]* @__PRETTY_FUNCTION__.get_filter_list, i32 0, i32 0), i8* null) noreturn nounwind, !dbg !132
  unreachable, !dbg !132

; CHECK: get_filter_list.exit
get_filter_list.exit6:                            ; preds = %sw.bb3.i4, %sw.bb2.i3, %sw.bb1.i2, %sw.bb.i1
  %1 = phi %struct._GList** [ @display_edited_filters, %sw.bb3.i4 ], [ @capture_edited_filters, %sw.bb2.i3 ], [ @display_filters, %sw.bb1.i2 ], [ @capture_filters, %sw.bb.i1 ]
  call void @llvm.dbg.value(metadata !{%struct._GList** %1}, i64 0, metadata !133), !dbg !120
; CHECK: %2 = load
  %2 = load %struct._GList** %1, align 8, !dbg !134
  call void @llvm.dbg.value(metadata !{%struct._GList* %2}, i64 0, metadata !135), !dbg !134
; We should have jump-threading insert an additional load here for the value
; coming out of the first switch, which is picked up by a subsequent phi
; CHECK: {{%\.pr = load %[^%]* %0}}
; CHECK-NEXT:  br label %while.cond
  br label %while.cond, !dbg !136

; CHECK: while.cond
while.cond:                                       ; preds = %while.body, %get_filter_list.exit6
; CHECK: {{= phi .*%.pr}}
  %3 = load %struct._GList** %0, align 8, !dbg !136
; CHECK: tobool
  %tobool = icmp ne %struct._GList* %3, null, !dbg !136
  br i1 %tobool, label %while.body, label %while.end, !dbg !136

while.body:                                       ; preds = %while.cond
  %4 = load %struct._GList** %0, align 8, !dbg !137
  %5 = load %struct._GList** %0, align 8, !dbg !139
  %call2 = call %struct._GList* @g_list_first(%struct._GList* %5), !dbg !139
  call void @llvm.dbg.value(metadata !{%struct._GList* %4}, i64 0, metadata !140) nounwind, !dbg !141
  call void @llvm.dbg.value(metadata !{%struct._GList* %call2}, i64 0, metadata !142) nounwind, !dbg !143
  %data.i = getelementptr inbounds %struct._GList* %call2, i32 0, i32 0, !dbg !144
  %6 = load i8** %data.i, align 8, !dbg !144
  %7 = bitcast i8* %6 to %struct.filter_def*, !dbg !144
  call void @llvm.dbg.value(metadata !{%struct.filter_def* %7}, i64 0, metadata !146) nounwind, !dbg !144
  %name.i = getelementptr inbounds %struct.filter_def* %7, i32 0, i32 0, !dbg !153
  %8 = load i8** %name.i, align 8, !dbg !153
  call void @g_free(i8* %8) nounwind, !dbg !153
  %strval.i = getelementptr inbounds %struct.filter_def* %7, i32 0, i32 1, !dbg !154
  %9 = load i8** %strval.i, align 8, !dbg !154
  call void @g_free(i8* %9) nounwind, !dbg !154
  %10 = bitcast %struct.filter_def* %7 to i8*, !dbg !155
  call void @g_free(i8* %10) nounwind, !dbg !155
  %call.i = call %struct._GList* @g_list_remove_link(%struct._GList* %4, %struct._GList* %call2) nounwind, !dbg !156
  store %struct._GList* %call.i, %struct._GList** %0, align 8, !dbg !139
  br label %while.cond, !dbg !157

while.end:                                        ; preds = %while.cond
  br label %do.body4, !dbg !158

do.body4:                                         ; preds = %while.end
  %11 = load %struct._GList** %0, align 8, !dbg !159
  %call5 = call i32 @g_list_length(%struct._GList* %11), !dbg !159
  %cmp6 = icmp eq i32 %call5, 0, !dbg !159
  br i1 %cmp6, label %if.then7, label %if.else8, !dbg !159

if.then7:                                         ; preds = %do.body4
  br label %if.end9, !dbg !159

if.else8:                                         ; preds = %do.body4
  call void @g_assertion_message_expr(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 600, i8* getelementptr inbounds ([62 x i8]* @__PRETTY_FUNCTION__.copy_filter_list, i32 0, i32 0), i8* getelementptr inbounds ([31 x i8]* @.str13, i32 0, i32 0)) noreturn, !dbg !159
  unreachable, !dbg !159

if.end9:                                          ; preds = %if.then7
  br label %do.end10, !dbg !159

do.end10:                                         ; preds = %if.end9
  br label %while.cond11, !dbg !161

while.cond11:                                     ; preds = %cond.end, %do.end10
  %cond10 = phi %struct._GList* [ %cond, %cond.end ], [ %2, %do.end10 ]
  %tobool12 = icmp ne %struct._GList* %cond10, null, !dbg !161
  br i1 %tobool12, label %while.body13, label %while.end16, !dbg !161

while.body13:                                     ; preds = %while.cond11
  %data = getelementptr inbounds %struct._GList* %cond10, i32 0, i32 0, !dbg !162
  %12 = load i8** %data, align 8, !dbg !162
  %13 = bitcast i8* %12 to %struct.filter_def*, !dbg !162
  call void @llvm.dbg.value(metadata !{%struct.filter_def* %13}, i64 0, metadata !164), !dbg !162
  %14 = load %struct._GList** %0, align 8, !dbg !165
  %name = getelementptr inbounds %struct.filter_def* %13, i32 0, i32 0, !dbg !165
  %15 = load i8** %name, align 8, !dbg !165
  %strval = getelementptr inbounds %struct.filter_def* %13, i32 0, i32 1, !dbg !165
  %16 = load i8** %strval, align 8, !dbg !165
  call void @llvm.dbg.value(metadata !{%struct._GList* %14}, i64 0, metadata !166) nounwind, !dbg !167
  call void @llvm.dbg.value(metadata !{i8* %15}, i64 0, metadata !168) nounwind, !dbg !169
  call void @llvm.dbg.value(metadata !{i8* %16}, i64 0, metadata !170) nounwind, !dbg !171
  %call.i7 = call noalias i8* @g_malloc(i64 16) nounwind, !dbg !172
  %17 = bitcast i8* %call.i7 to %struct.filter_def*, !dbg !172
  call void @llvm.dbg.value(metadata !{%struct.filter_def* %17}, i64 0, metadata !174) nounwind, !dbg !172
  %call1.i = call noalias i8* @g_strdup(i8* %15) nounwind, !dbg !175
  %name.i8 = getelementptr inbounds %struct.filter_def* %17, i32 0, i32 0, !dbg !175
  store i8* %call1.i, i8** %name.i8, align 8, !dbg !175
  %call2.i = call noalias i8* @g_strdup(i8* %16) nounwind, !dbg !176
  %strval.i9 = getelementptr inbounds %struct.filter_def* %17, i32 0, i32 1, !dbg !176
  store i8* %call2.i, i8** %strval.i9, align 8, !dbg !176
  %18 = bitcast %struct.filter_def* %17 to i8*, !dbg !177
  %call3.i = call %struct._GList* @g_list_append(%struct._GList* %14, i8* %18) nounwind, !dbg !177
  store %struct._GList* %call3.i, %struct._GList** %0, align 8, !dbg !165
  %tobool15 = icmp ne %struct._GList* %cond10, null, !dbg !178
  br i1 %tobool15, label %cond.true, label %cond.false, !dbg !178

cond.true:                                        ; preds = %while.body13
  %next = getelementptr inbounds %struct._GList* %cond10, i32 0, i32 1, !dbg !178
  %19 = load %struct._GList** %next, align 8, !dbg !178
  br label %cond.end, !dbg !178

cond.false:                                       ; preds = %while.body13
  br label %cond.end, !dbg !178

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi %struct._GList* [ %19, %cond.true ], [ null, %cond.false ], !dbg !178
  call void @llvm.dbg.value(metadata !{%struct._GList* %cond}, i64 0, metadata !135), !dbg !178
  br label %while.cond11, !dbg !179

while.end16:                                      ; preds = %while.cond11
  ret void, !dbg !180
}

declare void @g_assertion_message_expr(i8*, i8*, i32, i8*, i8*) noreturn

declare i32 @g_list_length(%struct._GList*)

declare noalias i8* @g_strdup(i8*)

declare %struct._GList* @g_list_append(%struct._GList*, i8*)

declare %struct._GList* @g_list_remove_link(%struct._GList*, %struct._GList*)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 12, metadata !"filters.c", metadata !"/sw/src/fink.build/wireshark-1.6.5-1/wireshark-1.6.5", metadata !"clang version 3.2 (trunk 155986)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !22, metadata !24, metadata !83} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{metadata !3, metadata !10, metadata !10, metadata !10, metadata !10, metadata !10, metadata !10, metadata !3, metadata !3, metadata !3, metadata !3}
!3 = metadata !{i32 786436, null, metadata !"", metadata !4, i32 29, i64 32, i64 32, i32 0, i32 0, null, metadata !5, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!4 = metadata !{i32 786473, metadata !"./filters.h", metadata !"/sw/src/fink.build/wireshark-1.6.5-1/wireshark-1.6.5", null} ; [ DW_TAG_file_type ]
!5 = metadata !{metadata !6, metadata !7, metadata !8, metadata !9}
!6 = metadata !{i32 786472, metadata !"CFILTER_LIST", i64 0} ; [ DW_TAG_enumerator ]
!7 = metadata !{i32 786472, metadata !"DFILTER_LIST", i64 1} ; [ DW_TAG_enumerator ]
!8 = metadata !{i32 786472, metadata !"CFILTER_EDITED_LIST", i64 2} ; [ DW_TAG_enumerator ]
!9 = metadata !{i32 786472, metadata !"DFILTER_EDITED_LIST", i64 3} ; [ DW_TAG_enumerator ]
!10 = metadata !{i32 786436, null, metadata !"", metadata !11, i32 57, i64 32, i64 32, i32 0, i32 0, null, metadata !12, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!11 = metadata !{i32 786473, metadata !"/sw/include/glib-2.0/glib/gmessages.h", metadata !"/sw/src/fink.build/wireshark-1.6.5-1/wireshark-1.6.5", null} ; [ DW_TAG_file_type ]
!12 = metadata !{metadata !13, metadata !14, metadata !15, metadata !16, metadata !17, metadata !18, metadata !19, metadata !20, metadata !21}
!13 = metadata !{i32 786472, metadata !"G_LOG_FLAG_RECURSION", i64 1} ; [ DW_TAG_enumerator ]
!14 = metadata !{i32 786472, metadata !"G_LOG_FLAG_FATAL", i64 2} ; [ DW_TAG_enumerator ]
!15 = metadata !{i32 786472, metadata !"G_LOG_LEVEL_ERROR", i64 4} ; [ DW_TAG_enumerator ]
!16 = metadata !{i32 786472, metadata !"G_LOG_LEVEL_CRITICAL", i64 8} ; [ DW_TAG_enumerator ]
!17 = metadata !{i32 786472, metadata !"G_LOG_LEVEL_WARNING", i64 16} ; [ DW_TAG_enumerator ]
!18 = metadata !{i32 786472, metadata !"G_LOG_LEVEL_MESSAGE", i64 32} ; [ DW_TAG_enumerator ]
!19 = metadata !{i32 786472, metadata !"G_LOG_LEVEL_INFO", i64 64} ; [ DW_TAG_enumerator ]
!20 = metadata !{i32 786472, metadata !"G_LOG_LEVEL_DEBUG", i64 128} ; [ DW_TAG_enumerator ]
!21 = metadata !{i32 786472, metadata !"G_LOG_LEVEL_MASK", i64 4294967292} ; [ DW_TAG_enumerator ]
!22 = metadata !{metadata !23}
!23 = metadata !{i32 0}
!24 = metadata !{metadata !25}
!25 = metadata !{metadata !26, metadata !36, metadata !51, metadata !56, metadata !59, metadata !60, metadata !63, metadata !67, metadata !70, metadata !74, metadata !79, metadata !80}
!26 = metadata !{i32 786478, i32 0, metadata !27, metadata !"read_filter_list", metadata !"read_filter_list", metadata !"", metadata !27, i32 115, metadata !28, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 117} ; [ DW_TAG_subprogram ]
!27 = metadata !{i32 786473, metadata !"filters.c", metadata !"/sw/src/fink.build/wireshark-1.6.5-1/wireshark-1.6.5", null} ; [ DW_TAG_file_type ]
!28 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !29, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!29 = metadata !{null, metadata !30, metadata !31, metadata !34}
!30 = metadata !{i32 786454, null, metadata !"filter_list_type_t", metadata !27, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !3} ; [ DW_TAG_typedef ]
!31 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !32} ; [ DW_TAG_pointer_type ]
!32 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !33} ; [ DW_TAG_pointer_type ]
!33 = metadata !{i32 786468, null, metadata !"char", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!34 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !35} ; [ DW_TAG_pointer_type ]
!35 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!36 = metadata !{i32 786478, i32 0, metadata !27, metadata !"get_filter_list_first", metadata !"get_filter_list_first", metadata !"", metadata !27, i32 418, metadata !37, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 419} ; [ DW_TAG_subprogram ]
!37 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !38, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!38 = metadata !{metadata !39, metadata !30}
!39 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !40} ; [ DW_TAG_pointer_type ]
!40 = metadata !{i32 786454, null, metadata !"GList", metadata !27, i32 38, i64 0, i64 0, i64 0, i32 0, metadata !41} ; [ DW_TAG_typedef ]
!41 = metadata !{i32 786451, null, metadata !"_GList", metadata !42, i32 40, i64 192, i64 64, i32 0, i32 0, null, metadata !43, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!42 = metadata !{i32 786473, metadata !"/sw/include/glib-2.0/glib/glist.h", metadata !"/sw/src/fink.build/wireshark-1.6.5-1/wireshark-1.6.5", null} ; [ DW_TAG_file_type ]
!43 = metadata !{metadata !44, metadata !47, metadata !50}
!44 = metadata !{i32 786445, metadata !41, metadata !"data", metadata !42, i32 42, i64 64, i64 64, i64 0, i32 0, metadata !45} ; [ DW_TAG_member ]
!45 = metadata !{i32 786454, null, metadata !"gpointer", metadata !42, i32 77, i64 0, i64 0, i64 0, i32 0, metadata !46} ; [ DW_TAG_typedef ]
!46 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!47 = metadata !{i32 786445, metadata !41, metadata !"next", metadata !42, i32 43, i64 64, i64 64, i64 64, i32 0, metadata !48} ; [ DW_TAG_member ]
!48 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !49} ; [ DW_TAG_pointer_type ]
!49 = metadata !{i32 786454, null, metadata !"GList", metadata !42, i32 38, i64 0, i64 0, i64 0, i32 0, metadata !41} ; [ DW_TAG_typedef ]
!50 = metadata !{i32 786445, metadata !41, metadata !"prev", metadata !42, i32 44, i64 64, i64 64, i64 128, i32 0, metadata !48} ; [ DW_TAG_member ]
!51 = metadata !{i32 786478, i32 0, metadata !27, metadata !"add_to_filter_list", metadata !"add_to_filter_list", metadata !"", metadata !27, i32 431, metadata !52, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 433} ; [ DW_TAG_subprogram ]
!52 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !53, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!53 = metadata !{metadata !39, metadata !30, metadata !54, metadata !54}
!54 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !55} ; [ DW_TAG_pointer_type ]
!55 = metadata !{i32 786470, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !33} ; [ DW_TAG_const_type ]
!56 = metadata !{i32 786478, i32 0, metadata !27, metadata !"remove_from_filter_list", metadata !"remove_from_filter_list", metadata !"", metadata !27, i32 446, metadata !57, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 447} ; [ DW_TAG_subprogram ]
!57 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !58, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!58 = metadata !{null, metadata !30, metadata !39}
!59 = metadata !{i32 786478, i32 0, metadata !27, metadata !"save_filter_list", metadata !"save_filter_list", metadata !"", metadata !27, i32 463, metadata !28, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 465} ; [ DW_TAG_subprogram ]
!60 = metadata !{i32 786478, i32 0, metadata !27, metadata !"copy_filter_list", metadata !"copy_filter_list", metadata !"", metadata !27, i32 574, metadata !61, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32, i32)* @copy_filter_list, null, null, metadata !22, i32 575} ; [ DW_TAG_subprogram ]
!61 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !62, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!62 = metadata !{null, metadata !30, metadata !30}
!63 = metadata !{i32 786478, i32 0, metadata !27, metadata !"get_filter_list", metadata !"get_filter_list", metadata !"", metadata !27, i32 385, metadata !64, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 386} ; [ DW_TAG_subprogram ]
!64 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !65, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!65 = metadata !{metadata !66, metadata !30}
!66 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !39} ; [ DW_TAG_pointer_type ]
!67 = metadata !{i32 786478, i32 0, metadata !27, metadata !"add_filter_entry", metadata !"add_filter_entry", metadata !"", metadata !27, i32 92, metadata !68, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 93} ; [ DW_TAG_subprogram ]
!68 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !69, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!69 = metadata !{metadata !39, metadata !39, metadata !54, metadata !54}
!70 = metadata !{i32 786478, i32 0, metadata !71, metadata !"isspace", metadata !"isspace", metadata !"", metadata !71, i32 284, metadata !72, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 285} ; [ DW_TAG_subprogram ]
!71 = metadata !{i32 786473, metadata !"/usr/include/ctype.h", metadata !"/sw/src/fink.build/wireshark-1.6.5-1/wireshark-1.6.5", null} ; [ DW_TAG_file_type ]
!72 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !73, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!73 = metadata !{metadata !35, metadata !35}
!74 = metadata !{i32 786478, i32 0, metadata !71, metadata !"__istype", metadata !"__istype", metadata !"", metadata !71, i32 170, metadata !75, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 171} ; [ DW_TAG_subprogram ]
!75 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !76, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!76 = metadata !{metadata !35, metadata !77, metadata !78}
!77 = metadata !{i32 786454, null, metadata !"__darwin_ct_rune_t", metadata !71, i32 70, i64 0, i64 0, i64 0, i32 0, metadata !35} ; [ DW_TAG_typedef ]
!78 = metadata !{i32 786468, null, metadata !"long unsigned int", null, i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!79 = metadata !{i32 786478, i32 0, metadata !71, metadata !"isascii", metadata !"isascii", metadata !"", metadata !71, i32 152, metadata !72, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 153} ; [ DW_TAG_subprogram ]
!80 = metadata !{i32 786478, i32 0, metadata !27, metadata !"remove_filter_entry", metadata !"remove_filter_entry", metadata !"", metadata !27, i32 103, metadata !81, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !22, i32 104} ; [ DW_TAG_subprogram ]
!81 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !82, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!82 = metadata !{metadata !39, metadata !39, metadata !39}
!83 = metadata !{metadata !84}
!84 = metadata !{metadata !85, metadata !86, metadata !87, metadata !88}
!85 = metadata !{i32 786484, i32 0, null, metadata !"display_edited_filters", metadata !"display_edited_filters", metadata !"", metadata !27, i32 78, metadata !39, i32 1, i32 1, %struct._GList** @display_edited_filters} ; [ DW_TAG_variable ]
!86 = metadata !{i32 786484, i32 0, null, metadata !"capture_edited_filters", metadata !"capture_edited_filters", metadata !"", metadata !27, i32 73, metadata !39, i32 1, i32 1, %struct._GList** @capture_edited_filters} ; [ DW_TAG_variable ]
!87 = metadata !{i32 786484, i32 0, null, metadata !"display_filters", metadata !"display_filters", metadata !"", metadata !27, i32 68, metadata !39, i32 1, i32 1, %struct._GList** @display_filters} ; [ DW_TAG_variable ]
!88 = metadata !{i32 786484, i32 0, null, metadata !"capture_filters", metadata !"capture_filters", metadata !"", metadata !27, i32 63, metadata !39, i32 1, i32 1, %struct._GList** @capture_filters} ; [ DW_TAG_variable ]
!89 = metadata !{i32 786689, metadata !60, metadata !"dest_type", metadata !27, i32 16777790, metadata !30, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!90 = metadata !{i32 574, i32 42, metadata !60, null}
!91 = metadata !{i32 786689, metadata !60, metadata !"src_type", metadata !27, i32 33555006, metadata !30, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!92 = metadata !{i32 574, i32 72, metadata !60, null}
!93 = metadata !{i32 581, i32 5, metadata !94, null}
!94 = metadata !{i32 786443, metadata !60, i32 575, i32 1, metadata !27, i32 51} ; [ DW_TAG_lexical_block ]
!95 = metadata !{i32 581, i32 5, metadata !96, null}
!96 = metadata !{i32 786443, metadata !94, i32 581, i32 5, metadata !27, i32 52} ; [ DW_TAG_lexical_block ]
!97 = metadata !{i32 786689, metadata !63, metadata !"list_type", metadata !27, i32 16777601, metadata !30, i32 0, metadata !98} ; [ DW_TAG_arg_variable ]
!98 = metadata !{i32 583, i32 17, metadata !94, null}
!99 = metadata !{i32 385, i32 36, metadata !63, metadata !98}
!100 = metadata !{i32 389, i32 3, metadata !101, metadata !98}
!101 = metadata !{i32 786443, metadata !63, i32 386, i32 1, metadata !27, i32 56} ; [ DW_TAG_lexical_block ]
!102 = metadata !{%struct._GList** @capture_filters}
!103 = metadata !{i32 786688, metadata !101, metadata !"flpp", metadata !27, i32 387, metadata !66, i32 0, metadata !98} ; [ DW_TAG_auto_variable ]
!104 = metadata !{i32 392, i32 5, metadata !105, metadata !98}
!105 = metadata !{i32 786443, metadata !101, i32 389, i32 22, metadata !27, i32 57} ; [ DW_TAG_lexical_block ]
!106 = metadata !{i32 393, i32 5, metadata !105, metadata !98}
!107 = metadata !{%struct._GList** @display_filters}
!108 = metadata !{i32 396, i32 5, metadata !105, metadata !98}
!109 = metadata !{i32 397, i32 5, metadata !105, metadata !98}
!110 = metadata !{%struct._GList** @capture_edited_filters}
!111 = metadata !{i32 400, i32 5, metadata !105, metadata !98}
!112 = metadata !{i32 401, i32 5, metadata !105, metadata !98}
!113 = metadata !{%struct._GList** @display_edited_filters}
!114 = metadata !{i32 404, i32 5, metadata !105, metadata !98}
!115 = metadata !{i32 405, i32 5, metadata !105, metadata !98}
!116 = metadata !{i32 408, i32 5, metadata !117, metadata !98}
!117 = metadata !{i32 786443, metadata !105, i32 408, i32 5, metadata !27, i32 58} ; [ DW_TAG_lexical_block ]
!118 = metadata !{i32 786688, metadata !94, metadata !"flpp_dest", metadata !27, i32 576, metadata !66, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!119 = metadata !{i32 786689, metadata !63, metadata !"list_type", metadata !27, i32 16777601, metadata !30, i32 0, metadata !120} ; [ DW_TAG_arg_variable ]
!120 = metadata !{i32 584, i32 16, metadata !94, null}
!121 = metadata !{i32 385, i32 36, metadata !63, metadata !120}
!122 = metadata !{i32 389, i32 3, metadata !101, metadata !120}
!123 = metadata !{i32 786688, metadata !101, metadata !"flpp", metadata !27, i32 387, metadata !66, i32 0, metadata !120} ; [ DW_TAG_auto_variable ]
!124 = metadata !{i32 392, i32 5, metadata !105, metadata !120}
!125 = metadata !{i32 393, i32 5, metadata !105, metadata !120}
!126 = metadata !{i32 396, i32 5, metadata !105, metadata !120}
!127 = metadata !{i32 397, i32 5, metadata !105, metadata !120}
!128 = metadata !{i32 400, i32 5, metadata !105, metadata !120}
!129 = metadata !{i32 401, i32 5, metadata !105, metadata !120}
!130 = metadata !{i32 404, i32 5, metadata !105, metadata !120}
!131 = metadata !{i32 405, i32 5, metadata !105, metadata !120}
!132 = metadata !{i32 408, i32 5, metadata !117, metadata !120}
!133 = metadata !{i32 786688, metadata !94, metadata !"flpp_src", metadata !27, i32 577, metadata !66, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!134 = metadata !{i32 585, i32 5, metadata !94, null}
!135 = metadata !{i32 786688, metadata !94, metadata !"flp_src", metadata !27, i32 578, metadata !39, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!136 = metadata !{i32 597, i32 5, metadata !94, null}
!137 = metadata !{i32 598, i32 22, metadata !138, null}
!138 = metadata !{i32 786443, metadata !94, i32 597, i32 23, metadata !27, i32 53} ; [ DW_TAG_lexical_block ]
!139 = metadata !{i32 598, i32 54, metadata !138, null}
!140 = metadata !{i32 786689, metadata !80, metadata !"fl", metadata !27, i32 16777319, metadata !39, i32 0, metadata !139} ; [ DW_TAG_arg_variable ]
!141 = metadata !{i32 103, i32 28, metadata !80, metadata !139}
!142 = metadata !{i32 786689, metadata !80, metadata !"fl_entry", metadata !27, i32 33554535, metadata !39, i32 0, metadata !139} ; [ DW_TAG_arg_variable ]
!143 = metadata !{i32 103, i32 39, metadata !80, metadata !139}
!144 = metadata !{i32 107, i32 3, metadata !145, metadata !139}
!145 = metadata !{i32 786443, metadata !80, i32 104, i32 1, metadata !27, i32 63} ; [ DW_TAG_lexical_block ]
!146 = metadata !{i32 786688, metadata !145, metadata !"filt", metadata !27, i32 105, metadata !147, i32 0, metadata !139} ; [ DW_TAG_auto_variable ]
!147 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !148} ; [ DW_TAG_pointer_type ]
!148 = metadata !{i32 786454, null, metadata !"filter_def", metadata !27, i32 42, i64 0, i64 0, i64 0, i32 0, metadata !149} ; [ DW_TAG_typedef ]
!149 = metadata !{i32 786451, null, metadata !"", metadata !4, i32 39, i64 128, i64 64, i32 0, i32 0, null, metadata !150, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!150 = metadata !{metadata !151, metadata !152}
!151 = metadata !{i32 786445, metadata !149, metadata !"name", metadata !4, i32 40, i64 64, i64 64, i64 0, i32 0, metadata !32} ; [ DW_TAG_member ]
!152 = metadata !{i32 786445, metadata !149, metadata !"strval", metadata !4, i32 41, i64 64, i64 64, i64 64, i32 0, metadata !32} ; [ DW_TAG_member ]
!153 = metadata !{i32 108, i32 3, metadata !145, metadata !139}
!154 = metadata !{i32 109, i32 3, metadata !145, metadata !139}
!155 = metadata !{i32 110, i32 3, metadata !145, metadata !139}
!156 = metadata !{i32 111, i32 10, metadata !145, metadata !139}
!157 = metadata !{i32 599, i32 5, metadata !138, null}
!158 = metadata !{i32 600, i32 5, metadata !94, null}
!159 = metadata !{i32 600, i32 5, metadata !160, null}
!160 = metadata !{i32 786443, metadata !94, i32 600, i32 5, metadata !27, i32 54} ; [ DW_TAG_lexical_block ]
!161 = metadata !{i32 603, i32 5, metadata !94, null}
!162 = metadata !{i32 604, i32 9, metadata !163, null}
!163 = metadata !{i32 786443, metadata !94, i32 603, i32 20, metadata !27, i32 55} ; [ DW_TAG_lexical_block ]
!164 = metadata !{i32 786688, metadata !94, metadata !"filt", metadata !27, i32 579, metadata !147, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!165 = metadata !{i32 606, i32 22, metadata !163, null}
!166 = metadata !{i32 786689, metadata !67, metadata !"fl", metadata !27, i32 16777308, metadata !39, i32 0, metadata !165} ; [ DW_TAG_arg_variable ]
!167 = metadata !{i32 92, i32 25, metadata !67, metadata !165}
!168 = metadata !{i32 786689, metadata !67, metadata !"filt_name", metadata !27, i32 33554524, metadata !54, i32 0, metadata !165} ; [ DW_TAG_arg_variable ]
!169 = metadata !{i32 92, i32 41, metadata !67, metadata !165}
!170 = metadata !{i32 786689, metadata !67, metadata !"filt_expr", metadata !27, i32 50331740, metadata !54, i32 0, metadata !165} ; [ DW_TAG_arg_variable ]
!171 = metadata !{i32 92, i32 64, metadata !67, metadata !165}
!172 = metadata !{i32 96, i32 35, metadata !173, metadata !165}
!173 = metadata !{i32 786443, metadata !67, i32 93, i32 1, metadata !27, i32 59} ; [ DW_TAG_lexical_block ]
!174 = metadata !{i32 786688, metadata !173, metadata !"filt", metadata !27, i32 94, metadata !147, i32 0, metadata !165} ; [ DW_TAG_auto_variable ]
!175 = metadata !{i32 97, i32 20, metadata !173, metadata !165}
!176 = metadata !{i32 98, i32 20, metadata !173, metadata !165}
!177 = metadata !{i32 99, i32 12, metadata !173, metadata !165}
!178 = metadata !{i32 607, i32 9, metadata !163, null}
!179 = metadata !{i32 608, i32 5, metadata !163, null}
!180 = metadata !{i32 609, i32 1, metadata !94, null}
