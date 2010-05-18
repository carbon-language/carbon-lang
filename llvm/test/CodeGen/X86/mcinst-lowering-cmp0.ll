; RUN: llc --show-mc-encoding -relocation-model=pic -disable-fp-elim -O3 < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0.0"

%struct.NSConstantString = type { i32*, i32, i8*, i32 }
%struct._objc_module = type { i32, i32, i8*, %struct._objc_symtab* }
%struct._objc_symtab = type { i32, i8*, i16, i16, [0 x i8*] }

@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__OBJC, __image_info,regular" ; <[2 x i32]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_" = internal global [4 x i8] c"foo\00", section "__TEXT,__cstring,cstring_literals", align 1 ; <[4 x i8]*> [#uses=1]
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4 ; <i8**> [#uses=3]
@__CFConstantStringClassReference = external global [0 x i32] ; <[0 x i32]*> [#uses=1]
@.str = private constant [3 x i8] c"||\00"        ; <[3 x i8]*> [#uses=1]
@_unnamed_cfstring_ = private constant %struct.NSConstantString { i32* getelementptr inbounds ([0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0), i32 2 }, section "__DATA,__cfstring" ; <%struct.NSConstantString*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_1" = internal global [5 x i8] c"baz:\00", section "__TEXT,__cstring,cstring_literals", align 1 ; <[5 x i8]*> [#uses=1]
@"\01L_OBJC_SELECTOR_REFERENCES_2" = internal global i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4 ; <i8**> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_3" = internal global [4 x i8] c"bar\00", section "__TEXT,__cstring,cstring_literals", align 1 ; <[4 x i8]*> [#uses=1]
@"\01L_OBJC_SELECTOR_REFERENCES_4" = internal global i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_3", i32 0, i32 0), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4 ; <i8**> [#uses=2]
@"\01L_OBJC_CLASS_NAME_" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 1 ; <[1 x i8]*> [#uses=1]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module { i32 7, i32 16, i8* getelementptr inbounds ([1 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct._objc_symtab* null }, section "__OBJC,__module_info,regular,no_dead_strip", align 4 ; <%struct._objc_module*> [#uses=1]
@llvm.used = appending global [9 x i8*] [i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_" to i8*), i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_2" to i8*), i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_3", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_4" to i8*), i8* getelementptr inbounds ([1 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*)], section "llvm.metadata" ; <[9 x i8*]*> [#uses=0]

define void @f0(i8* nocapture %a, i8* nocapture %b) nounwind optsize ssp {
entry:
  %call = tail call i32 (...)* @get_name() nounwind optsize ; <i32> [#uses=2]
  %conv = inttoptr i32 %call to i8*               ; <i8*> [#uses=1]
  %call1 = tail call i32 (...)* @get_dict() nounwind optsize ; <i32> [#uses=2]
  %conv2 = inttoptr i32 %call1 to i8*             ; <i8*> [#uses=2]

; Check that we lower to the short form of cmpl, which has an 8-bit immediate.
;
; CHECK: cmpl  $0, -16(%ebp)           ## 4-byte Folded Reload
; CHECK:                               ## encoding: [0x83,0x7d,0xf0,0x00]
; rdar://7999130
  %cmp = icmp eq i32 %call1, 0                    ; <i1> [#uses=1]
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %tmp5 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_" ; <i8*> [#uses=1]
  %call6 = tail call i8* (i8*, i8*, ...)* @objc_msgSend(i8* %conv2, i8* %tmp5) nounwind optsize ; <i8*> [#uses=1]
  %tmp7 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_2" ; <i8*> [#uses=1]
  %call820 = tail call i8* (i8*, i8*, ...)* @objc_msgSend(i8* %call6, i8* %tmp7, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring_ to i8*)) nounwind optsize ; <i8*> [#uses=0]
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %tmp10 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_" ; <i8*> [#uses=1]
  %call11 = tail call i8* (i8*, i8*, ...)* @objc_msgSend(i8* %conv2, i8* %tmp10) nounwind optsize ; <i8*> [#uses=1]
  %tmp12 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_4" ; <i8*> [#uses=1]
  %call13 = tail call i8* (i8*, i8*, ...)* @objc_msgSend(i8* %call11, i8* %tmp12) nounwind optsize ; <i8*> [#uses=0]
  %cmp15 = icmp eq i32 %call, 0                   ; <i1> [#uses=1]
  br i1 %cmp15, label %if.end19, label %if.then17

if.then17:                                        ; preds = %if.end
  tail call void (...)* @f1(i8* %conv) nounwind optsize
  ret void

if.end19:                                         ; preds = %if.end
  ret void
}

declare i32 @get_name(...) optsize

declare i32 @get_dict(...) optsize

declare i8* @objc_msgSend(i8*, i8*, ...)

declare void @f1(...) optsize
