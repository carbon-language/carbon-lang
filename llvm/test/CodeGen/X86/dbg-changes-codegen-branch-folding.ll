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
  %tobool = icmp eq i32 %param2, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = call i8* @_Z5i2stri(i32 %param2)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  call void @llvm.dbg.value(metadata !{%struct.AAA3* %var1}, i64 0, metadata !60)
  call void @llvm.dbg.value(metadata !62, i64 0, metadata !63)
  %arraydecay.i = getelementptr inbounds %struct.AAA3* %var1, i64 0, i32 0, i64 0
  call void @_Z3fooPcjPKc(i8* %arraydecay.i, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0))
  call void @llvm.dbg.declare(metadata !{%struct.AAA3* %var2}, metadata !38)
  %arraydecay.i5 = getelementptr inbounds %struct.AAA3* %var2, i64 0, i32 0, i64 0
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0))
  %tobool1 = icmp eq i32 %param1, 0
  br i1 %tobool1, label %if.else, label %if.then2

if.then2:                                         ; preds = %if.end
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([2 x i8]* @.str1, i64 0, i64 0))
  br label %if.end3

if.else:                                          ; preds = %if.end
  call void @_Z3fooPcjPKc(i8* %arraydecay.i5, i32 4, i8* getelementptr inbounds ([2 x i8]* @.str2, i64 0, i64 0))
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2
  call void @_Z3fooPcjPKc(i8* %arraydecay.i, i32 4, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0))
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare i8* @_Z5i2stri(i32) #2

declare void @_Z3fooPcjPKc(i8*, i32, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!48, !49}
!llvm.ident = !{!50}

!38 = metadata !{i32 786688, null, metadata !"var2", null, i32 20, null, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [var2] [line 20]
!48 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!49 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!50 = metadata !{metadata !"clang version 3.5 (202418)"}
!60 = metadata !{i32 786689, null, metadata !"this", null, i32 16777216, null, i32 1088, null} ; [ DW_TAG_arg_variable ] [this] [line 0]
!62 = metadata !{i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)}
!63 = metadata !{i32 786689, null, metadata !"value", null, i32 33554439, null, i32 0, null} ; [ DW_TAG_arg_variable ] [value] [line 7]
