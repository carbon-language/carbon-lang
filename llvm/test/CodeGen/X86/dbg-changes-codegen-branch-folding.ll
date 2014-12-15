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
  tail call void @llvm.dbg.value(metadata i32 %param1, i64 0, metadata !30, metadata !{!"0x102"}), !dbg !47
  tail call void @llvm.dbg.value(metadata i32 %param2, i64 0, metadata !31, metadata !{!"0x102"}), !dbg !47
  tail call void @llvm.dbg.value(metadata i8* null, i64 0, metadata !32, metadata !{!"0x102"}), !dbg !49
  %tobool = icmp eq i32 %param2, 0, !dbg !50
  br i1 %tobool, label %if.end, label %if.then, !dbg !50

if.then:                                          ; preds = %entry
  %call = tail call i8* @_Z5i2stri(i32 %param2), !dbg !52
  tail call void @llvm.dbg.value(metadata i8* %call, i64 0, metadata !32, metadata !{!"0x102"}), !dbg !49
  br label %if.end, !dbg !54

if.end:                                           ; preds = %entry, %if.then
  tail call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !33, metadata !{!"0x102"}), !dbg !55
  tail call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !56, metadata !{!"0x102"}), !dbg !57
  tail call void @llvm.dbg.value(metadata !58, i64 0, metadata !59, metadata !{!"0x102"}), !dbg !60
  %arraydecay.i = getelementptr inbounds %struct.AAA3* %var1, i64 0, i32 0, i64 0, !dbg !61
  call void @_Z3fooPcjPKc(i8* %arraydecay.i, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !61
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !34, metadata !{!"0x102"}), !dbg !63
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !64, metadata !{!"0x102"}), !dbg !65
  call void @llvm.dbg.value(metadata !58, i64 0, metadata !66, metadata !{!"0x102"}), !dbg !67
  %arraydecay.i5 = getelementptr inbounds %struct.AAA3* %var2, i64 0, i32 0, i64 0, !dbg !68
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !68
  %tobool1 = icmp eq i32 %param1, 0, !dbg !69
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !34, metadata !{!"0x102"}), !dbg !63
  br i1 %tobool1, label %if.else, label %if.then2, !dbg !69

if.then2:                                         ; preds = %if.end
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !71, metadata !{!"0x102"}), !dbg !73
  call void @llvm.dbg.value(metadata !74, i64 0, metadata !75, metadata !{!"0x102"}), !dbg !76
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([2 x i8]* @.str1, i64 0, i64 0)), !dbg !76
  br label %if.end3, !dbg !72

if.else:                                          ; preds = %if.end
  call void @llvm.dbg.value(metadata %struct.AAA3* %var2, i64 0, metadata !77, metadata !{!"0x102"}), !dbg !79
  call void @llvm.dbg.value(metadata !80, i64 0, metadata !81, metadata !{!"0x102"}), !dbg !82
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([2 x i8]* @.str2, i64 0, i64 0)), !dbg !82
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2
  call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !33, metadata !{!"0x102"}), !dbg !55
  call void @llvm.dbg.value(metadata %struct.AAA3* %var1, i64 0, metadata !83, metadata !{!"0x102"}), !dbg !85
  call void @llvm.dbg.value(metadata !58, i64 0, metadata !86, metadata !{!"0x102"}), !dbg !87
  call void @_Z3fooPcjPKc(i8* %arraydecay.i, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !87
  ret void, !dbg !88
}

declare i8* @_Z5i2stri(i32) #1

declare void @_Z3fooPcjPKc(i8*, i32, i8*) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!44, !45}
!llvm.ident = !{!46}

!0 = !{!"0x11\004\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !3, !23, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"dbg-changes-codegen-branch-folding.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00AAA3\004\0032\008\000\000\000", !1, null, null, !5, null, null, !"_ZTS4AAA3"} ; [ DW_TAG_structure_type ] [AAA3] [line 4, size 32, align 8, offset 0] [def] [from ]
!5 = !{!6, !11, !17, !18}
!6 = !{!"0xd\00text\008\0032\008\000\000", !1, !"_ZTS4AAA3", !7} ; [ DW_TAG_member ] [text] [line 8, size 32, align 8, offset 0] [from ]
!7 = !{!"0x1\00\000\0032\008\000\000", null, null, !8, !9, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32, align 8, offset 0] [from char]
!8 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!9 = !{!10}
!10 = !{!"0x21\000\004"}        ; [ DW_TAG_subrange_type ] [0, 3]
!11 = !{!"0x2e\00AAA3\00AAA3\00\005\000\000\000\006\00256\001\005", !1, !"_ZTS4AAA3", !12, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 5] [AAA3]
!12 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = !{null, !14, !15}
!14 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS4AAA3"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS4AAA3]
!15 = !{!"0xf\00\000\0064\0064\000\000", null, null, !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!16 = !{!"0x26\00\000\000\000\000\000", null, null, !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
!17 = !{!"0x2e\00operator=\00operator=\00_ZN4AAA3aSEPKc\006\000\000\000\006\00256\001\006", !1, !"_ZTS4AAA3", !12, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 6] [operator=]
!18 = !{!"0x2e\00operator const char *\00operator const char *\00_ZNK4AAA3cvPKcEv\007\000\000\000\006\00256\001\007", !1, !"_ZTS4AAA3", !19, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 7] [operator const char *]
!19 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !20, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = !{!15, !21}
!21 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !22} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from ]
!22 = !{!"0x26\00\000\000\000\000\000", null, null, !"_ZTS4AAA3"} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _ZTS4AAA3]
!23 = !{!24, !35, !40}
!24 = !{!"0x2e\00bar\00bar\00_Z3barii\0011\000\001\000\006\00256\001\0011", !1, !25, !26, null, void (i32, i32)* @_Z3barii, null, null, !29} ; [ DW_TAG_subprogram ] [line 11] [def] [bar]
!25 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!26 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !27, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = !{null, !28, !28}
!28 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!29 = !{!30, !31, !32, !33, !34}
!30 = !{!"0x101\00param1\0016777227\000", !24, !25, !28} ; [ DW_TAG_arg_variable ] [param1] [line 11]
!31 = !{!"0x101\00param2\0033554443\000", !24, !25, !28} ; [ DW_TAG_arg_variable ] [param2] [line 11]
!32 = !{!"0x100\00temp\0012\000", !24, !25, !15} ; [ DW_TAG_auto_variable ] [temp] [line 12]
!33 = !{!"0x100\00var1\0017\000", !24, !25, !"_ZTS4AAA3"} ; [ DW_TAG_auto_variable ] [var1] [line 17]
!34 = !{!"0x100\00var2\0018\000", !24, !25, !"_ZTS4AAA3"} ; [ DW_TAG_auto_variable ] [var2] [line 18]
!35 = !{!"0x2e\00operator=\00operator=\00_ZN4AAA3aSEPKc\006\000\001\000\006\00256\001\006", !1, !"_ZTS4AAA3", !12, null, null, null, !17, !36} ; [ DW_TAG_subprogram ] [line 6] [def] [operator=]
!36 = !{!37, !39}
!37 = !{!"0x101\00this\0016777216\001088", !35, null, !38} ; [ DW_TAG_arg_variable ] [this] [line 0]
!38 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS4AAA3"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS4AAA3]
!39 = !{!"0x101\00value\0033554438\000", !35, !25, !15} ; [ DW_TAG_arg_variable ] [value] [line 6]
!40 = !{!"0x2e\00AAA3\00AAA3\00_ZN4AAA3C2EPKc\005\000\001\000\006\00256\001\005", !1, !"_ZTS4AAA3", !12, null, null, null, !11, !41} ; [ DW_TAG_subprogram ] [line 5] [def] [AAA3]
!41 = !{!42, !43}
!42 = !{!"0x101\00this\0016777216\001088", !40, null, !38} ; [ DW_TAG_arg_variable ] [this] [line 0]
!43 = !{!"0x101\00value\0033554437\000", !40, !25, !15} ; [ DW_TAG_arg_variable ] [value] [line 5]
!44 = !{i32 2, !"Dwarf Version", i32 4}
!45 = !{i32 2, !"Debug Info Version", i32 2}
!46 = !{!"clang version 3.5.0 "}
!47 = !{i32 11, i32 0, !24, null}
!48 = !{i8* null}
!49 = !{i32 12, i32 0, !24, null}
!50 = !{i32 14, i32 0, !51, null}
!51 = !{!"0xb\0014\000\000", !1, !24} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!52 = !{i32 15, i32 0, !53, null}
!53 = !{!"0xb\0014\000\000", !1, !51} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!54 = !{i32 16, i32 0, !53, null}
!55 = !{i32 17, i32 0, !24, null}
!56 = !{!"0x101\00this\0016777216\001088", !40, null, !38, !55} ; [ DW_TAG_arg_variable ] [this] [line 0]
!57 = !{i32 0, i32 0, !40, !55}
!58 = !{i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)}
!59 = !{!"0x101\00value\0033554437\000", !40, !25, !15, !55} ; [ DW_TAG_arg_variable ] [value] [line 5]
!60 = !{i32 5, i32 0, !40, !55}
!61 = !{i32 5, i32 0, !62, !55}
!62 = !{!"0xb\005\000\000", !1, !40} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!63 = !{i32 18, i32 0, !24, null}
!64 = !{!"0x101\00this\0016777216\001088", !40, null, !38, !63} ; [ DW_TAG_arg_variable ] [this] [line 0]
!65 = !{i32 0, i32 0, !40, !63}
!66 = !{!"0x101\00value\0033554437\000", !40, !25, !15, !63} ; [ DW_TAG_arg_variable ] [value] [line 5]
!67 = !{i32 5, i32 0, !40, !63}
!68 = !{i32 5, i32 0, !62, !63}
!69 = !{i32 20, i32 0, !70, null}
!70 = !{!"0xb\0020\000\000", !1, !24} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/dbg-changes-codegen-branch-folding.cpp]
!71 = !{!"0x101\00this\0016777216\001088", !35, null, !38, !72} ; [ DW_TAG_arg_variable ] [this] [line 0]
!72 = !{i32 21, i32 0, !70, null}
!73 = !{i32 0, i32 0, !35, !72}
!74 = !{i8* getelementptr inbounds ([2 x i8]* @.str1, i64 0, i64 0)}
!75 = !{!"0x101\00value\0033554438\000", !35, !25, !15, !72} ; [ DW_TAG_arg_variable ] [value] [line 6]
!76 = !{i32 6, i32 0, !35, !72}
!77 = !{!"0x101\00this\0016777216\001088", !35, null, !38, !78} ; [ DW_TAG_arg_variable ] [this] [line 0]
!78 = !{i32 23, i32 0, !70, null}
!79 = !{i32 0, i32 0, !35, !78}
!80 = !{i8* getelementptr inbounds ([2 x i8]* @.str2, i64 0, i64 0)}
!81 = !{!"0x101\00value\0033554438\000", !35, !25, !15, !78} ; [ DW_TAG_arg_variable ] [value] [line 6]
!82 = !{i32 6, i32 0, !35, !78}
!83 = !{!"0x101\00this\0016777216\001088", !35, null, !38, !84} ; [ DW_TAG_arg_variable ] [this] [line 0]
!84 = !{i32 24, i32 0, !24, null}
!85 = !{i32 0, i32 0, !35, !84}
!86 = !{!"0x101\00value\0033554438\000", !35, !25, !15, !84} ; [ DW_TAG_arg_variable ] [value] [line 6]
!87 = !{i32 6, i32 0, !35, !84}
!88 = !{i32 25, i32 0, !24, null}
