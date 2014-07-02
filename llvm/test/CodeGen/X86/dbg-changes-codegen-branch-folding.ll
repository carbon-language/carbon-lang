; RUN: llc -march=x86-64 -mtriple=x86_64-linux < %s | FileCheck %s
; RUN: opt -strip-debug < %s | llc -march=x86-64 -mtriple=x86_64-linux | FileCheck %s
; http://llvm.org/PR19051. Minor code-motion difference with -g.
; Presence of debug info shouldn't affect the codegen. Make sure that
; we generated the same code sequence with and without debug info. 
;
; CHECK:      callq   _Z3fooPcjPKc
; CHECK:      callq   _Z3fooPcjPKc
; CHECK:      leaq    (%rsp), %rdi
; CHECK:      movl    $4, %esi
; CHECK:      testl   {{%[a-z]+}}, {{%[a-z]+}}
; CHECK:      je     .LBB0_4

; Regenerate test with this command: 
;   clang -emit-llvm -S -O2 -g
; from this source:
;
; extern void foo(char *dst,unsigned siz,const char *src);
; extern const char * i2str(int);
;
; struct AAA3 {
;  AAA3(const char *value) { foo(text,sizeof(text),value);}
;  void operator=(const char *value) { foo(text,sizeof(text),value);}
;  operator const char*() const { return text;}
;  char text[4];
; };
;
; void bar (int param1,int param2)  {
;   const char * temp(0);
;
;   if (param2) {
;     temp = i2str(param2);
;   }
;   AAA3 var1("");
;   AAA3 var2("");
;
;   if (param1)
;     var2 = "+";
;   else
;     var2 = "-";
;   var1 = "";
; }

%struct.AAA3 = type { [4 x i8] }

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str1 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@.str2 = private unnamed_addr constant [2 x i8] c"-\00", align 1

; Function Attrs: uwtable
define void @_Z3barii(i32 %param1, i32 %param2) #0 {
entry:
  %var1 = alloca %struct.AAA3, align 1
  %var2 = alloca %struct.AAA3, align 1
  tail call void @llvm.dbg.value(metadata !{i32 %param1}, i64 0, metadata !30), !dbg !47
  tail call void @llvm.dbg.value(metadata !{i32 %param2}, i64 0, metadata !31), !dbg !47
  tail call void @llvm.dbg.value(metadata !48, i64 0, metadata !32), !dbg !49
  %tobool = icmp eq i32 %param2, 0, !dbg !50
  br i1 %tobool, label %if.end, label %if.then, !dbg !50

if.then:                                          ; preds = %entry
  %call = tail call i8* @_Z5i2stri(i32 %param2), !dbg !52
  tail call void @llvm.dbg.value(metadata !{i8* %call}, i64 0, metadata !32), !dbg !49
  br label %if.end, !dbg !54

if.end:                                           ; preds = %entry, %if.then
  tail call void @llvm.dbg.value(metadata !{%struct.AAA3* %var1}, i64 0, metadata !33), !dbg !55
  tail call void @llvm.dbg.value(metadata !{%struct.AAA3* %var1}, i64 0, metadata !56), !dbg !57
  tail call void @llvm.dbg.value(metadata !58, i64 0, metadata !59), !dbg !60
  %arraydecay.i = getelementptr inbounds %struct.AAA3* %var1, i64 0, i32 0, i64 0, !dbg !61
  call void @_Z3fooPcjPKc(i8* %arraydecay.i, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !61
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var2}, i64 0, metadata !34), !dbg !63
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var2}, i64 0, metadata !64), !dbg !65
  call void @llvm.dbg.value(metadata !58, i64 0, metadata !66), !dbg !67
  %arraydecay.i5 = getelementptr inbounds %struct.AAA3* %var2, i64 0, i32 0, i64 0, !dbg !68
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !68
  %tobool1 = icmp eq i32 %param1, 0, !dbg !69
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var2}, i64 0, metadata !34), !dbg !63
  br i1 %tobool1, label %if.else, label %if.then2, !dbg !69

if.then2:                                         ; preds = %if.end
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var2}, i64 0, metadata !71), !dbg !73
  call void @llvm.dbg.value(metadata !74, i64 0, metadata !75), !dbg !76
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([2 x i8]* @.str1, i64 0, i64 0)), !dbg !76
  br label %if.end3, !dbg !72

if.else:                                          ; preds = %if.end
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var2}, i64 0, metadata !77), !dbg !79
  call void @llvm.dbg.value(metadata !80, i64 0, metadata !81), !dbg !82
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([2 x i8]* @.str2, i64 0, i64 0)), !dbg !82
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var1}, i64 0, metadata !33), !dbg !55
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var1}, i64 0, metadata !83), !dbg !85
  call void @llvm.dbg.value(metadata !58, i64 0, metadata !86), !dbg !87
  call void @_Z3fooPcjPKc(i8* %arraydecay.i, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !87
  ret void, !dbg !88
}

declare i8* @_Z5i2stri(i32) #1

declare void @_Z3fooPcjPKc(i8*, i32, i8*) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!44, !45}
!llvm.ident = !{!46}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !3, metadata !23, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"dbg-changes-codegen-branch-folding.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"AAA3", i32 4, i64 32, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS4AAA3"} ; [ DW_TAG_structure_type ] [AAA3] [line 4, size 32, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !11, metadata !17, metadata !18}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS4AAA3", metadata !"text", i32 8, i64 32, i64 8, i64 0, i32 0, metadata !7} ; [ DW_TAG_member ] [text] [line 8, size 32, align 8, offset 0] [from ]
!7 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 32, i64 8, i32 0, i32 0, metadata !8, metadata !9, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32, align 8, offset 0] [from char]
!8 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786465, i64 0, i64 4}        ; [ DW_TAG_subrange_type ] [0, 3]
!11 = metadata !{i32 786478, metadata !1, metadata !"_ZTS4AAA3", metadata !"AAA3", metadata !"AAA3", metadata !"", i32 5, metadata !12, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, null, i32 5} ; [ DW_TAG_subprogram ] [line 5] [AAA3]
!12 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{null, metadata !14, metadata !15}
!14 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS4AAA3"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS4AAA3]
!15 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!16 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
!17 = metadata !{i32 786478, metadata !1, metadata !"_ZTS4AAA3", metadata !"operator=", metadata !"operator=", metadata !"_ZN4AAA3aSEPKc", i32 6, metadata !12, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, null, i32 6} ; [ DW_TAG_subprogram ] [line 6] [operator=]
!18 = metadata !{i32 786478, metadata !1, metadata !"_ZTS4AAA3", metadata !"operator const char *", metadata !"operator const char *", metadata !"_ZNK4AAA3cvPKcEv", i32 7, metadata !19, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, null, i32 7} ; [ DW_TAG_subprogram ] [line 7] [operator const char *]
!19 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !20, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = metadata !{metadata !15, metadata !21}
!21 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !22} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from ]
!22 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !"_ZTS4AAA3"} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _ZTS4AAA3]
!23 = metadata !{metadata !24, metadata !35, metadata !40}
!24 = metadata !{i32 786478, metadata !1, metadata !25, metadata !"bar", metadata !"bar", metadata !"_Z3barii", i32 11, metadata !26, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i32, i32)* @_Z3barii, null, null, metadata !29, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [bar]
!25 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!26 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !27, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{null, metadata !28, metadata !28}
!28 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!29 = metadata !{metadata !30, metadata !31, metadata !32, metadata !33, metadata !34}
!30 = metadata !{i32 786689, metadata !24, metadata !"param1", metadata !25, i32 16777227, metadata !28, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [param1] [line 11]
!31 = metadata !{i32 786689, metadata !24, metadata !"param2", metadata !25, i32 33554443, metadata !28, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [param2] [line 11]
!32 = metadata !{i32 786688, metadata !24, metadata !"temp", metadata !25, i32 12, metadata !15, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [temp] [line 12]
!33 = metadata !{i32 786688, metadata !24, metadata !"var1", metadata !25, i32 17, metadata !"_ZTS4AAA3", i32 0, i32 0} ; [ DW_TAG_auto_variable ] [var1] [line 17]
!34 = metadata !{i32 786688, metadata !24, metadata !"var2", metadata !25, i32 18, metadata !"_ZTS4AAA3", i32 0, i32 0} ; [ DW_TAG_auto_variable ] [var2] [line 18]
!35 = metadata !{i32 786478, metadata !1, metadata !"_ZTS4AAA3", metadata !"operator=", metadata !"operator=", metadata !"_ZN4AAA3aSEPKc", i32 6, metadata !12, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !17, metadata !36, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [operator=]
!36 = metadata !{metadata !37, metadata !39}
!37 = metadata !{i32 786689, metadata !35, metadata !"this", null, i32 16777216, metadata !38, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!38 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS4AAA3"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS4AAA3]
!39 = metadata !{i32 786689, metadata !35, metadata !"value", metadata !25, i32 33554438, metadata !15, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [value] [line 6]
!40 = metadata !{i32 786478, metadata !1, metadata !"_ZTS4AAA3", metadata !"AAA3", metadata !"AAA3", metadata !"_ZN4AAA3C2EPKc", i32 5, metadata !12, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !11, metadata !41, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [AAA3]
!41 = metadata !{metadata !42, metadata !43}
!42 = metadata !{i32 786689, metadata !40, metadata !"this", null, i32 16777216, metadata !38, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!43 = metadata !{i32 786689, metadata !40, metadata !"value", metadata !25, i32 33554437, metadata !15, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [value] [line 5]
!44 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!45 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!46 = metadata !{metadata !"clang version 3.5.0 "}
!47 = metadata !{i32 11, i32 0, metadata !24, null}
!48 = metadata !{i8* null}
!49 = metadata !{i32 12, i32 0, metadata !24, null}
!50 = metadata !{i32 14, i32 0, metadata !51, null}
!51 = metadata !{i32 786443, metadata !1, metadata !24, i32 14, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!52 = metadata !{i32 15, i32 0, metadata !53, null}
!53 = metadata !{i32 786443, metadata !1, metadata !51, i32 14, i32 0, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!54 = metadata !{i32 16, i32 0, metadata !53, null}
!55 = metadata !{i32 17, i32 0, metadata !24, null}
!56 = metadata !{i32 786689, metadata !40, metadata !"this", null, i32 16777216, metadata !38, i32 1088, metadata !55} ; [ DW_TAG_arg_variable ] [this] [line 0]
!57 = metadata !{i32 0, i32 0, metadata !40, metadata !55}
!58 = metadata !{i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)}
!59 = metadata !{i32 786689, metadata !40, metadata !"value", metadata !25, i32 33554437, metadata !15, i32 0, metadata !55} ; [ DW_TAG_arg_variable ] [value] [line 5]
!60 = metadata !{i32 5, i32 0, metadata !40, metadata !55}
!61 = metadata !{i32 5, i32 0, metadata !62, metadata !55}
!62 = metadata !{i32 786443, metadata !1, metadata !40, i32 5, i32 0, i32 0, i32 3} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!63 = metadata !{i32 18, i32 0, metadata !24, null}
!64 = metadata !{i32 786689, metadata !40, metadata !"this", null, i32 16777216, metadata !38, i32 1088, metadata !63} ; [ DW_TAG_arg_variable ] [this] [line 0]
!65 = metadata !{i32 0, i32 0, metadata !40, metadata !63}
!66 = metadata !{i32 786689, metadata !40, metadata !"value", metadata !25, i32 33554437, metadata !15, i32 0, metadata !63} ; [ DW_TAG_arg_variable ] [value] [line 5]
!67 = metadata !{i32 5, i32 0, metadata !40, metadata !63}
!68 = metadata !{i32 5, i32 0, metadata !62, metadata !63}
!69 = metadata !{i32 20, i32 0, metadata !70, null}
!70 = metadata !{i32 786443, metadata !1, metadata !24, i32 20, i32 0, i32 0, i32 2} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!71 = metadata !{i32 786689, metadata !35, metadata !"this", null, i32 16777216, metadata !38, i32 1088, metadata !72} ; [ DW_TAG_arg_variable ] [this] [line 0]
!72 = metadata !{i32 21, i32 0, metadata !70, null}
!73 = metadata !{i32 0, i32 0, metadata !35, metadata !72}
!74 = metadata !{i8* getelementptr inbounds ([2 x i8]* @.str1, i64 0, i64 0)}
!75 = metadata !{i32 786689, metadata !35, metadata !"value", metadata !25, i32 33554438, metadata !15, i32 0, metadata !72} ; [ DW_TAG_arg_variable ] [value] [line 6]
!76 = metadata !{i32 6, i32 0, metadata !35, metadata !72}
!77 = metadata !{i32 786689, metadata !35, metadata !"this", null, i32 16777216, metadata !38, i32 1088, metadata !78} ; [ DW_TAG_arg_variable ] [this] [line 0]
!78 = metadata !{i32 23, i32 0, metadata !70, null}
!79 = metadata !{i32 0, i32 0, metadata !35, metadata !78}
!80 = metadata !{i8* getelementptr inbounds ([2 x i8]* @.str2, i64 0, i64 0)}
!81 = metadata !{i32 786689, metadata !35, metadata !"value", metadata !25, i32 33554438, metadata !15, i32 0, metadata !78} ; [ DW_TAG_arg_variable ] [value] [line 6]
!82 = metadata !{i32 6, i32 0, metadata !35, metadata !78}
!83 = metadata !{i32 786689, metadata !35, metadata !"this", null, i32 16777216, metadata !38, i32 1088, metadata !84} ; [ DW_TAG_arg_variable ] [this] [line 0]
!84 = metadata !{i32 24, i32 0, metadata !24, null}
!85 = metadata !{i32 0, i32 0, metadata !35, metadata !84}
!86 = metadata !{i32 786689, metadata !35, metadata !"value", metadata !25, i32 33554438, metadata !15, i32 0, metadata !84} ; [ DW_TAG_arg_variable ] [value] [line 6]
!87 = metadata !{i32 6, i32 0, metadata !35, metadata !84}
!88 = metadata !{i32 25, i32 0, metadata !24, null}
