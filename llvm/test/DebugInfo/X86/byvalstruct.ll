; RUN: llc  -mtriple=x86_64-apple-macosx10.8.0 -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; Test that we generate debug info for by-value struct args that are not used.
;
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "info"
;
; generated from
;
; typedef unsigned long NSUInteger;
; typedef struct
; {
;  NSUInteger width;
;  NSUInteger height;
;  double pixelAspect;
; } ImageInfo;
; @implementation Bitmap
; - (id)initWithCopy:(Bitmap *)otherBitmap
;            andInfo:(ImageInfo)info
;        andLength:(NSUInteger)length
; {
; }
; @end

; ModuleID = 't.mm'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%0 = type opaque
%struct._objc_cache = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8** }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i64*, i8*, i8*, i32, i32 }
%struct.ImageInfo = type { i64, i64, double }

@_objc_empty_cache = external global %struct._objc_cache
@_objc_empty_vtable = external global i8* (i8*, i8*)*
@"OBJC_CLASS_$_Bitmap" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Bitmap", %struct._class_t* null, %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** @_objc_empty_vtable, %struct._class_ro_t* @"\01l_OBJC_CLASS_RO_$_Bitmap" }, section "__DATA, __objc_data", align 8
@"OBJC_METACLASS_$_Bitmap" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Bitmap", %struct._class_t* @"OBJC_CLASS_$_Bitmap", %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** @_objc_empty_vtable, %struct._class_ro_t* @"\01l_OBJC_METACLASS_RO_$_Bitmap" }, section "__DATA, __objc_data", align 8
@"\01L_OBJC_CLASS_NAME_" = internal global [7 x i8] c"Bitmap\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"\01l_OBJC_METACLASS_RO_$_Bitmap" = internal global %struct._class_ro_t { i32 3, i32 40, i32 40, i8* null, i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* null, %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [32 x i8] c"initWithCopy:andInfo:andLength:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [23 x i8] c"@56@0:8@16{?=QQd}24Q48\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_METHODS_Bitmap" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([32 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([23 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast (i8* (%0*, i8*, %0*, %struct.ImageInfo*, i64)* @"\01-[Bitmap initWithCopy:andInfo:andLength:]" to i8*) }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_Bitmap" = internal global %struct._class_ro_t { i32 2, i32 0, i32 0, i8* null, i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Bitmap" to %struct.__method_list_t*), %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_LABEL_CLASS_$" = internal global [1 x i8*] [i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_Bitmap" to i8*)], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.used = appending global [5 x i8*] [i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([32 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([23 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Bitmap" to i8*), i8* bitcast ([1 x i8*]* @"\01L_OBJC_LABEL_CLASS_$" to i8*)], section "llvm.metadata"

; Function Attrs: ssp uwtable
define internal i8* @"\01-[Bitmap initWithCopy:andInfo:andLength:]"(%0* %self, i8* %_cmd, %0* %otherBitmap, %struct.ImageInfo* byval align 8 %info, i64 %length) #0 {
entry:
  %retval = alloca i8*, align 8
  %self.addr = alloca %0*, align 8
  %_cmd.addr = alloca i8*, align 8
  %otherBitmap.addr = alloca %0*, align 8
  %length.addr = alloca i64, align 8
  store %0* %self, %0** %self.addr, align 8
  call void @llvm.dbg.declare(metadata !{%0** %self.addr}, metadata !28, metadata !{metadata !"0x102"}), !dbg !29
  store i8* %_cmd, i8** %_cmd.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8** %_cmd.addr}, metadata !30, metadata !{metadata !"0x102"}), !dbg !29
  store %0* %otherBitmap, %0** %otherBitmap.addr, align 8
  call void @llvm.dbg.declare(metadata !{%0** %otherBitmap.addr}, metadata !32, metadata !{metadata !"0x102"}), !dbg !29
  call void @llvm.dbg.declare(metadata !{%struct.ImageInfo* %info}, metadata !33, metadata !{metadata !"0x102"}), !dbg !34
  store i64 %length, i64* %length.addr, align 8
  call void @llvm.dbg.declare(metadata !{i64* %length.addr}, metadata !35, metadata !{metadata !"0x102"}), !dbg !36
  %0 = load i8** %retval, !dbg !37
  ret i8* %0, !dbg !37
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25, !26, !27, !38}

!0 = metadata !{metadata !"0x11\0017\00clang version 3.4 \000\00\002\00\000", metadata !1, metadata !2, metadata !3, metadata !6, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/t.mm] [DW_LANG_ObjC_plus_plus]
!1 = metadata !{metadata !"t.mm", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00Bitmap\008\008\008\000\00512\0017", metadata !1, metadata !5, null, metadata !2, null, null, null} ; [ DW_TAG_structure_type ] [Bitmap] [line 8, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/t.mm]
!6 = metadata !{metadata !7}
!7 = metadata !{metadata !"0x2e\00-[Bitmap initWithCopy:andInfo:andLength:]\00-[Bitmap initWithCopy:andInfo:andLength:]\00\009\001\001\000\006\00256\000\009", metadata !1, metadata !5, metadata !8, null, i8* (%0*, i8*, %0*, %struct.ImageInfo*, i64)* @"\01-[Bitmap initWithCopy:andInfo:andLength:]", null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 9] [local] [def] [-[Bitmap initWithCopy:andInfo:andLength:]]
!8 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{metadata !4, metadata !10, metadata !11, metadata !14, metadata !15, metadata !19}
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from Bitmap]
!11 = metadata !{metadata !"0x16\00SEL\009\000\000\000\0064", metadata !1, null, metadata !12} ; [ DW_TAG_typedef ] [SEL] [line 9, size 0, align 0, offset 0] [artificial] [from ]
!12 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_selector]
!13 = metadata !{metadata !"0x13\00objc_selector\000\000\000\000\004\000", metadata !1, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [objc_selector] [line 0, size 0, align 0, offset 0] [decl] [from ]
!14 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from Bitmap]
!15 = metadata !{metadata !"0x16\00ImageInfo\007\000\000\000\000", metadata !1, null, metadata !16} ; [ DW_TAG_typedef ] [ImageInfo] [line 7, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !"0x13\00\002\00192\0064\000\000\000", metadata !1, null, null, metadata !17, null, null, null} ; [ DW_TAG_structure_type ] [line 2, size 192, align 64, offset 0] [def] [from ]
!17 = metadata !{metadata !18, metadata !21, metadata !22}
!18 = metadata !{metadata !"0xd\00width\004\0064\0064\000\000", metadata !1, metadata !16, metadata !19} ; [ DW_TAG_member ] [width] [line 4, size 64, align 64, offset 0] [from NSUInteger]
!19 = metadata !{metadata !"0x16\00NSUInteger\001\000\000\000\000", metadata !1, null, metadata !20} ; [ DW_TAG_typedef ] [NSUInteger] [line 1, size 0, align 0, offset 0] [from long unsigned int]
!20 = metadata !{metadata !"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!21 = metadata !{metadata !"0xd\00height\005\0064\0064\0064\000", metadata !1, metadata !16, metadata !19} ; [ DW_TAG_member ] [height] [line 5, size 64, align 64, offset 64] [from NSUInteger]
!22 = metadata !{metadata !"0xd\00pixelAspect\006\0064\0064\00128\000", metadata !1, metadata !16, metadata !23} ; [ DW_TAG_member ] [pixelAspect] [line 6, size 64, align 64, offset 128] [from double]
!23 = metadata !{metadata !"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!24 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!25 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!26 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!27 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
!28 = metadata !{metadata !"0x101\00self\0016777225\001088", metadata !7, metadata !5, metadata !14} ; [ DW_TAG_arg_variable ] [self] [line 9]
!29 = metadata !{i32 9, i32 0, metadata !7, null}
!30 = metadata !{metadata !"0x101\00_cmd\0033554441\0064", metadata !7, metadata !5, metadata !31} ; [ DW_TAG_arg_variable ] [_cmd] [line 9]
!31 = metadata !{metadata !"0x16\00SEL\009\000\000\000\000", metadata !1, null, metadata !12} ; [ DW_TAG_typedef ] [SEL] [line 9, size 0, align 0, offset 0] [from ]
!32 = metadata !{metadata !"0x101\00otherBitmap\0050331657\000", metadata !7, metadata !5, metadata !14} ; [ DW_TAG_arg_variable ] [otherBitmap] [line 9]
!33 = metadata !{metadata !"0x101\00info\0067108874\000", metadata !7, metadata !5, metadata !15} ; [ DW_TAG_arg_variable ] [info] [line 10]
!34 = metadata !{i32 10, i32 0, metadata !7, null}
!35 = metadata !{metadata !"0x101\00length\0083886091\000", metadata !7, metadata !5, metadata !19} ; [ DW_TAG_arg_variable ] [length] [line 11]
!36 = metadata !{i32 11, i32 0, metadata !7, null}
!37 = metadata !{i32 13, i32 0, metadata !7, null}
!38 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
