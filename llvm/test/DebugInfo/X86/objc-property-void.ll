; RUN: llc -filetype=obj -o %t.o < %s >/dev/null 2>&1
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s

; CHECK: DW_TAG_structure_type
; CHECK:  DW_AT_APPLE_objc_complete_type
; CHECK:  DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9a-fA-F]+}}] = "Foo")
; CHECK: DW_TAG_APPLE_property
; CHECK:  DW_AT_APPLE_property_name [DW_FORM_strp] ( .debug_str[0x{{[0-9a-fA-F]+}}] = "foo")

; generated from:
; @interface Foo
; @property (nonatomic,assign,readonly) void foo;
; @end
; @implementation Foo
; - (void)foo {}
; @end
;
; with:
; clang -S -emit-llvm -O0 -g

; ModuleID = '-'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%0 = type opaque
%struct._objc_cache = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8** }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i64*, i8*, i8*, i32, i32 }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_Foo" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Foo", %struct._class_t* null, %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null, %struct._class_ro_t* @"\01l_OBJC_CLASS_RO_$_Foo" }, section "__DATA, __objc_data", align 8
@"OBJC_METACLASS_$_Foo" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Foo", %struct._class_t* @"OBJC_CLASS_$_Foo", %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null, %struct._class_ro_t* @"\01l_OBJC_METACLASS_RO_$_Foo" }, section "__DATA, __objc_data", align 8
@"\01L_OBJC_CLASS_NAME_" = internal global [4 x i8] c"Foo\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"\01l_OBJC_METACLASS_RO_$_Foo" = internal global %struct._class_ro_t { i32 3, i32 40, i32 40, i8* null, i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* null, %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [4 x i8] c"foo\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [8 x i8] c"v16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_METHODS_Foo" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast (void (%0*, i8*)* @"\01-[Foo foo]" to i8*) }] }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_PROP_NAME_ATTR_" = internal global [4 x i8] c"foo\00", section "__TEXT,__cstring,cstring_literals", align 1
@"\01L_OBJC_PROP_NAME_ATTR_1" = internal global [7 x i8] c"Tv,R,N\00", section "__TEXT,__cstring,cstring_literals", align 1
@"\01l_OBJC_$_PROP_LIST_Foo" = internal global { i32, i32, [1 x %struct._prop_t] } { i32 16, i32 1, [1 x %struct._prop_t] [%struct._prop_t { i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_", i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_1", i32 0, i32 0) }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_Foo" = internal global %struct._class_ro_t { i32 2, i32 0, i32 0, i8* null, i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Foo" to %struct.__method_list_t*), %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* bitcast ({ i32, i32, [1 x %struct._prop_t] }* @"\01l_OBJC_$_PROP_LIST_Foo" to %struct._prop_list_t*) }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_LABEL_CLASS_$" = internal global [1 x i8*] [i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_Foo" to i8*)], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.used = appending global [8 x i8*] [i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Foo" to i8*), i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_", i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_1", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._prop_t] }* @"\01l_OBJC_$_PROP_LIST_Foo" to i8*), i8* bitcast ([1 x i8*]* @"\01L_OBJC_LABEL_CLASS_$" to i8*)], section "llvm.metadata"

; Function Attrs: ssp uwtable
define internal void @"\01-[Foo foo]"(%0* %self, i8* %_cmd) #0 {
entry:
  %self.addr = alloca %0*, align 8
  %_cmd.addr = alloca i8*, align 8
  store %0* %self, %0** %self.addr, align 8
  call void @llvm.dbg.declare(metadata !{%0** %self.addr}, metadata !24), !dbg !26
  store i8* %_cmd, i8** %_cmd.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8** %_cmd.addr}, metadata !27), !dbg !26
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19, !20, !21, !22}
!llvm.ident = !{!23}

!0 = metadata !{i32 786449, metadata !1, i32 16, metadata !"", i1 false, metadata !"", i32 2, metadata !2, metadata !3, metadata !9, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [] [DW_LANG_ObjC]
!1 = metadata !{metadata !"-", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"Foo", i32 1, i64 0, i64 8, i32 0, i32 512, null, metadata !7, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [Foo] [line 1, size 0, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"<stdin>", metadata !""}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] []
!7 = metadata !{metadata !8}
!8 = metadata !{i32 803328, metadata !"foo", metadata !6, i32 2, metadata !"", metadata !"", i32 2117, null} ; [ DW_TAG_APPLE_property ] [foo] [line 2, properties 2117]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"-[Foo foo]", metadata !"-[Foo foo]", metadata !"", i32 5, metadata !11, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%0*, i8*)* @"\01-[Foo foo]", null, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [local] [def] [-[Foo foo]]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13, metadata !14}
!13 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from Foo]
!14 = metadata !{i32 786454, metadata !5, null, metadata !"SEL", i32 5, i64 0, i64 0, i64 0, i32 64, metadata !15} ; [ DW_TAG_typedef ] [SEL] [line 5, size 0, align 0, offset 0] [artificial] [from ]
!15 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_selector]
!16 = metadata !{i32 786451, metadata !1, null, metadata !"objc_selector", i32 0, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_selector] [line 0, size 0, align 0, offset 0] [decl] [from ]
!17 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!18 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!19 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!20 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!23 = metadata !{metadata !""}
!24 = metadata !{i32 786689, metadata !10, metadata !"self", null, i32 16777216, metadata !25, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [self] [line 0]
!25 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from Foo]
!26 = metadata !{i32 0, i32 0, metadata !10, null}
!27 = metadata !{i32 786689, metadata !10, metadata !"_cmd", null, i32 33554432, metadata !28, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [_cmd] [line 0]
!28 = metadata !{i32 786454, metadata !5, null, metadata !"SEL", i32 5, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_typedef ] [SEL] [line 5, size 0, align 0, offset 0] [from ]
!29 = metadata !{i32 5, i32 0, metadata !10, null}
