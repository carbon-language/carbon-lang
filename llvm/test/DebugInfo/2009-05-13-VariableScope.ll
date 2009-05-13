; RUN: llvm-as < %s | llc -O0

module asm "\09.lazy_reference .objc_class_name_NSTextFieldCell"
module asm "\09.objc_class_name_DVIconAndTextCell=0"
module asm "\09.globl .objc_class_name_DVIconAndTextCell"
module asm ""
module asm ""
	type { i32, [2 x %struct._objc_ivar] }		; type %0
	type { i8*, i32, [23 x %struct._objc_method] }		; type %1
	type { i32, %struct.objc_selector*, i16, i16, [1 x i8*] }		; type %2
	type opaque		; type %3
	type opaque		; type %4
	type opaque		; type %5
	type opaque		; type %6
	type opaque		; type %7
	type opaque		; type %8
	type opaque		; type %9
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
	%struct.CGPoint = type <{ double, double }>
	%struct.CGRect = type <{ %struct.CGPoint, %struct.CGPoint }>
	%struct.CGSize = type <{ double, double }>
	%struct.NSConstantString = type <{ i32*, i32, i8*, i32 }>
	%struct._NSZone = type opaque
	%struct._objc_cache = type opaque
	%struct._objc_category = type { i8*, i8*, %struct._objc_method_list*, %struct._objc_method_list*, %struct._objc_protocol_list*, i32, %struct._prop_list_t* }
	%struct._objc_class = type { %struct._objc_class*, %struct._objc_class*, i8*, i32, i32, i32, %struct._objc_ivar_list*, %struct._objc_method_list*, %struct._objc_cache*, %struct._objc_protocol_list*, i8*, %struct._objc_class_extension* }
	%struct._objc_class_extension = type { i32, i8*, %struct._prop_list_t* }
	%struct._objc_exception_data = type { [18 x i32], [4 x i8*] }
	%struct._objc_ivar = type { i8*, i8*, i32 }
	%struct._objc_ivar_list = type opaque
	%struct._objc_method = type { %struct.objc_selector*, i8*, i8* }
	%struct._objc_method_description = type { %struct.objc_selector*, i8* }
	%struct._objc_method_description_list = type { i32, [0 x %struct._objc_method_description] }
	%struct._objc_method_list = type opaque
	%struct._objc_module = type { i32, i32, i8*, %struct._objc_symtab* }
	%struct._objc_protocol = type { %struct._objc_protocol_extension*, i8*, %struct._objc_protocol_list*, %struct._objc_method_description_list*, %struct._objc_method_description_list* }
	%struct._objc_protocol_extension = type { i32, %struct._objc_method_description_list*, %struct._objc_method_description_list*, %struct._prop_list_t* }
	%struct._objc_protocol_list = type { %struct._objc_protocol_list*, i32, [0 x %struct._objc_protocol] }
	%struct._objc_super = type <{ %struct.objc_object*, %struct.objc_class* }>
	%struct._objc_symtab = type { i32, %struct.objc_selector*, i16, i16, [0 x i8*] }
	%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
	%struct._prop_t = type { i8*, i8* }
	%struct.objc_class = type opaque
	%struct.objc_object = type <{ %struct.objc_class* }>
	%struct.objc_selector = type opaque
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__OBJC, __image_info,regular"		; <[2 x i32]*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [20 x i8] c"DVIconAndTextCell.m\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@.str1 = internal constant [103 x i8] c"/Volumes/Data/ddunbar/private/GarnetXcodeIDE/XcodeIDE/Frameworks/DocSetManagementViewing/DocSetViewing\00", section "llvm.metadata"		; <[103 x i8]*> [#uses=1]
@.str2 = internal constant [10 x i8] c"clang 1.0\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([20 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([103 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [7 x i8] c"objc.h\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str4 = internal constant [18 x i8] c"/usr/include/objc\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.compile_unit5 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([7 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str6 = internal constant [12 x i8] c"objc_object\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str7 = internal constant [11 x i8] c"objc_class\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.composite8 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str7, i32 0, i32 0), { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite8 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str9 = internal constant [6 x i8] c"Class\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype10 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str9, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to { }*), i32 35, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str11 = internal constant [4 x i8] c"isa\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.derivedtype12 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str11, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to { }*), i32 37, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype10 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite13 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str6, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to { }*), i32 36, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype14 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite13 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str15 = internal constant [3 x i8] c"id\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.derivedtype16 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([3 x i8]* @.str15, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to { }*), i32 38, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype14 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str17 = internal constant [36 x i8] c"\01-[DVIconAndTextCell initTextCell:]\00", section "llvm.metadata"		; <[36 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([36 x i8]* @.str17, i32 0, i32 0), i8* getelementptr ([36 x i8]* @.str17, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 15, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str18 = internal constant [10 x i8] c"<unknown>\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str19 = internal constant [22 x i8] c"/Volumes/Sandbox/llvm\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.compile_unit20 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([10 x i8]* @.str18, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str19, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str21 = internal constant [20 x i8] c"DVIconAndTextCell.h\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.compile_unit22 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([20 x i8]* @.str21, i32 0, i32 0), i8* getelementptr ([103 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str23 = internal constant [18 x i8] c"DVIconAndTextCell\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@.str24 = internal constant [18 x i8] c"NSTextFieldCell.h\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@.str25 = internal constant [52 x i8] c"/System/Library/Frameworks/AppKit.framework/Headers\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit26 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([18 x i8]* @.str24, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str27 = internal constant [16 x i8] c"NSTextFieldCell\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str29 = internal constant [15 x i8] c"NSActionCell.h\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.compile_unit30 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([15 x i8]* @.str29, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str31 = internal constant [13 x i8] c"NSActionCell\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str33 = internal constant [9 x i8] c"NSCell.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit34 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([9 x i8]* @.str33, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str35 = internal constant [7 x i8] c"NSCell\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str37 = internal constant [11 x i8] c"NSObject.h\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str38 = internal constant [56 x i8] c"/System/Library/Frameworks/Foundation.framework/Headers\00", section "llvm.metadata"		; <[56 x i8]*> [#uses=1]
@llvm.dbg.compile_unit39 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([11 x i8]* @.str37, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str40 = internal constant [9 x i8] c"NSObject\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype42 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([4 x i8]* @.str11, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit39 to { }*), i32 66, i64 32, i64 32, i64 0, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype10 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array43 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype42 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite44 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str40, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit39 to { }*), i32 65, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array43 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype45 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str46 = internal constant [10 x i8] c"_contents\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype47 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str46, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 168, i64 32, i64 32, i64 32, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str48 = internal constant [9 x i8] c"__CFlags\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str50 = internal constant [13 x i8] c"unsigned int\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str50, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str51 = internal constant [6 x i8] c"state\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype52 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([6 x i8]* @.str51, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 112, i64 1, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str53 = internal constant [12 x i8] c"highlighted\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype54 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([12 x i8]* @.str53, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 113, i64 1, i64 32, i64 1, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str55 = internal constant [9 x i8] c"disabled\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype56 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str55, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 114, i64 1, i64 32, i64 2, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str57 = internal constant [9 x i8] c"editable\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype58 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str57, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 115, i64 1, i64 32, i64 3, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str59 = internal constant [14 x i8] c"unsigned long\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.basictype60 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str59, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str61 = internal constant [16 x i8] c"NSObjCRuntime.h\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.compile_unit62 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([16 x i8]* @.str61, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str63 = internal constant [11 x i8] c"NSUInteger\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype64 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str63, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit62 to { }*), i32 161, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype60 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str65 = internal constant [11 x i8] c"NSCellType\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype66 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str65, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 31, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype64 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str67 = internal constant [5 x i8] c"type\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype68 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([5 x i8]* @.str67, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 116, i64 2, i64 32, i64 4, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype66 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str69 = internal constant [10 x i8] c"vCentered\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype70 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str69, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 117, i64 1, i64 32, i64 6, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str71 = internal constant [10 x i8] c"hCentered\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype72 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str71, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 118, i64 1, i64 32, i64 7, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str73 = internal constant [9 x i8] c"bordered\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype74 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str73, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 119, i64 1, i64 32, i64 8, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str75 = internal constant [8 x i8] c"bezeled\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype76 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str75, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 120, i64 1, i64 32, i64 9, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str77 = internal constant [11 x i8] c"selectable\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype78 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str77, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 121, i64 1, i64 32, i64 10, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str79 = internal constant [11 x i8] c"scrollable\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype80 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str79, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 122, i64 1, i64 32, i64 11, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str81 = internal constant [11 x i8] c"continuous\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype82 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str81, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 123, i64 1, i64 32, i64 12, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str83 = internal constant [15 x i8] c"actOnMouseDown\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype84 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([15 x i8]* @.str83, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 124, i64 1, i64 32, i64 13, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str85 = internal constant [7 x i8] c"isLeaf\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype86 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str85, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 125, i64 1, i64 32, i64 14, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str87 = internal constant [19 x i8] c"invalidObjectValue\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype88 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([19 x i8]* @.str87, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 126, i64 1, i64 32, i64 15, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str89 = internal constant [12 x i8] c"invalidFont\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype90 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([12 x i8]* @.str89, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 127, i64 1, i64 32, i64 16, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str91 = internal constant [19 x i8] c"NSParagraphStyle.h\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.compile_unit92 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([19 x i8]* @.str91, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str93 = internal constant [16 x i8] c"NSLineBreakMode\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype94 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str93, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit92 to { }*), i32 28, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype64 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str95 = internal constant [14 x i8] c"lineBreakMode\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype96 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str95, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 128, i64 3, i64 32, i64 17, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype94 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str97 = internal constant [14 x i8] c"cellReserved1\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype98 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str97, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 129, i64 2, i64 32, i64 20, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str99 = internal constant [15 x i8] c"singleLineMode\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype100 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([15 x i8]* @.str99, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 130, i64 1, i64 32, i64 22, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str101 = internal constant [18 x i8] c"actOnMouseDragged\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype102 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([18 x i8]* @.str101, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 131, i64 1, i64 32, i64 23, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str103 = internal constant [9 x i8] c"isLoaded\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype104 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str103, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 132, i64 1, i64 32, i64 24, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str105 = internal constant [17 x i8] c"truncateLastLine\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype106 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str105, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 133, i64 1, i64 32, i64 25, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str107 = internal constant [17 x i8] c"dontActOnMouseUp\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype108 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str107, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 134, i64 1, i64 32, i64 26, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str109 = internal constant [8 x i8] c"isWhite\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype110 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str109, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 135, i64 1, i64 32, i64 27, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str111 = internal constant [21 x i8] c"useUserKeyEquivalent\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype112 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([21 x i8]* @.str111, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 136, i64 1, i64 32, i64 28, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str113 = internal constant [20 x i8] c"showsFirstResponder\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype114 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([20 x i8]* @.str113, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 137, i64 1, i64 32, i64 29, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str115 = internal constant [14 x i8] c"focusRingType\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype116 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str115, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 138, i64 2, i64 32, i64 30, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str117 = internal constant [14 x i8] c"wasSelectable\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype118 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str117, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 139, i64 1, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str119 = internal constant [17 x i8] c"hasInvalidObject\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype120 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str119, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 140, i64 1, i64 32, i64 33, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str121 = internal constant [28 x i8] c"allowsEditingTextAttributes\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.derivedtype122 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([28 x i8]* @.str121, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 141, i64 1, i64 32, i64 34, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str123 = internal constant [16 x i8] c"importsGraphics\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype124 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str123, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 142, i64 1, i64 32, i64 35, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str125 = internal constant [9 x i8] c"NSText.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit126 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([9 x i8]* @.str125, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str127 = internal constant [16 x i8] c"NSTextAlignment\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype128 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str127, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit126 to { }*), i32 36, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype64 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str129 = internal constant [10 x i8] c"alignment\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype130 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str129, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 143, i64 3, i64 32, i64 36, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype128 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str131 = internal constant [19 x i8] c"layoutDirectionRTL\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype132 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([19 x i8]* @.str131, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 144, i64 1, i64 32, i64 39, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str133 = internal constant [16 x i8] c"backgroundStyle\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype134 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str133, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 145, i64 3, i64 32, i64 40, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str135 = internal constant [14 x i8] c"cellReserved2\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype136 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str135, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 146, i64 4, i64 32, i64 43, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str137 = internal constant [22 x i8] c"refusesFirstResponder\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.derivedtype138 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([22 x i8]* @.str137, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 147, i64 1, i64 32, i64 47, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str139 = internal constant [21 x i8] c"needsHighlightedText\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype140 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([21 x i8]* @.str139, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 148, i64 1, i64 32, i64 48, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str141 = internal constant [15 x i8] c"dontAllowsUndo\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype142 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([15 x i8]* @.str141, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 149, i64 1, i64 32, i64 49, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str143 = internal constant [17 x i8] c"currentlyEditing\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype144 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str143, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 150, i64 1, i64 32, i64 50, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str145 = internal constant [17 x i8] c"allowsMixedState\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype146 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str145, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 151, i64 1, i64 32, i64 51, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str147 = internal constant [13 x i8] c"inMixedState\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype148 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str147, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 152, i64 1, i64 32, i64 52, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str149 = internal constant [24 x i8] c"sendsActionOnEndEditing\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.derivedtype150 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([24 x i8]* @.str149, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 153, i64 1, i64 32, i64 53, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str151 = internal constant [13 x i8] c"inSendAction\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype152 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str151, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 154, i64 1, i64 32, i64 54, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str153 = internal constant [11 x i8] c"menuWasSet\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype154 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str153, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 155, i64 1, i64 32, i64 55, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str155 = internal constant [12 x i8] c"controlTint\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype156 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([12 x i8]* @.str155, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 156, i64 3, i64 32, i64 56, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str157 = internal constant [12 x i8] c"controlSize\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype158 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([12 x i8]* @.str157, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 157, i64 2, i64 32, i64 59, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str159 = internal constant [20 x i8] c"branchImageDisabled\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype160 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([20 x i8]* @.str159, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 158, i64 1, i64 32, i64 61, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str161 = internal constant [20 x i8] c"drawingInRevealover\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype162 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([20 x i8]* @.str161, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 159, i64 1, i64 32, i64 62, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str163 = internal constant [25 x i8] c"needsHighlightedTextHint\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.derivedtype164 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([25 x i8]* @.str163, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 160, i64 1, i64 32, i64 63, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array165 = internal constant [49 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype52 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype54 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype56 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype58 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype68 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype70 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype72 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype74 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype76 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype78 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype82 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype84 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype86 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype88 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype90 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype96 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype98 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype100 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype102 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype104 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype106 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype108 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype110 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype112 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype114 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype116 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype118 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype120 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype122 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype124 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype130 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype132 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype134 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype136 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype138 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype140 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype142 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype144 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype146 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype148 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype150 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype152 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype154 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype156 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype158 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype160 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype162 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype164 to { }*)], section "llvm.metadata"		; <[49 x { }*]*> [#uses=1]
@llvm.dbg.composite166 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str48, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 111, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([49 x { }*]* @llvm.dbg.array165 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str167 = internal constant [8 x i8] c"_CFlags\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype168 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str167, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 161, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite166 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str169 = internal constant [8 x i8] c"_cFlags\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype170 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str169, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 169, i64 64, i64 32, i64 64, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype168 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str171 = internal constant [9 x i8] c"_support\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype172 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str171, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 173, i64 32, i64 32, i64 128, i32 1, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array173 = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype45 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype47 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype170 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype172 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite174 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str35, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit34 to { }*), i32 165, i64 160, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array173 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype175 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite174 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str176 = internal constant [5 x i8] c"long\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.basictype177 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([5 x i8]* @.str176, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str178 = internal constant [10 x i8] c"NSInteger\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype179 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str178, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit62 to { }*), i32 160, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype177 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str180 = internal constant [5 x i8] c"_tag\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype181 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([5 x i8]* @.str180, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 13, i64 32, i64 32, i64 160, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype179 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str182 = internal constant [8 x i8] c"_target\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype183 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str182, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 14, i64 32, i64 32, i64 192, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str184 = internal constant [14 x i8] c"objc_selector\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.composite185 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str184, i32 0, i32 0), { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype186 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite185 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str187 = internal constant [4 x i8] c"SEL\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.derivedtype188 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([4 x i8]* @.str187, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to { }*), i32 41, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype186 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str189 = internal constant [8 x i8] c"_action\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype190 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str189, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 15, i64 32, i64 32, i64 224, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str191 = internal constant [13 x i8] c"_controlView\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype192 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str191, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 16, i64 32, i64 32, i64 256, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array193 = internal constant [5 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype175 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype181 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype183 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype190 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype192 to { }*)], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite194 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str31, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 10, i64 288, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array193 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype195 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite194 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str196 = internal constant [10 x i8] c"NSColor.h\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.compile_unit197 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([10 x i8]* @.str196, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str198 = internal constant [8 x i8] c"NSColor\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype200 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array201 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype200 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite202 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str198, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit197 to { }*), i32 43, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array201 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype203 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite202 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str204 = internal constant [17 x i8] c"_backgroundColor\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype205 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str204, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 22, i64 32, i64 32, i64 288, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype203 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str206 = internal constant [11 x i8] c"_textColor\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype207 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str206, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 23, i64 32, i64 32, i64 320, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype203 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str208 = internal constant [10 x i8] c"__tfFlags\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str210 = internal constant [16 x i8] c"drawsBackground\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype211 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str210, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 25, i64 1, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str212 = internal constant [11 x i8] c"bezelStyle\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype213 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str212, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 26, i64 3, i64 32, i64 1, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str214 = internal constant [17 x i8] c"thcSortDirection\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype215 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str214, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 27, i64 2, i64 32, i64 4, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str216 = internal constant [16 x i8] c"thcSortPriority\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype217 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str216, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 28, i64 4, i64 32, i64 6, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str218 = internal constant [5 x i8] c"mini\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype219 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([5 x i8]* @.str218, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 29, i64 1, i64 32, i64 10, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str220 = internal constant [34 x i8] c"textColorIgnoresNormalDisableFlag\00", section "llvm.metadata"		; <[34 x i8]*> [#uses=1]
@llvm.dbg.derivedtype221 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([34 x i8]* @.str220, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 30, i64 1, i64 32, i64 11, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str222 = internal constant [21 x i8] c"textColorDisableFlag\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype223 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([21 x i8]* @.str222, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 31, i64 1, i64 32, i64 12, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str224 = internal constant [25 x i8] c"thcForceHighlightForSort\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.derivedtype225 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([25 x i8]* @.str224, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 32, i64 1, i64 32, i64 13, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str226 = internal constant [17 x i8] c"invalidTextColor\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype227 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([17 x i8]* @.str226, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 33, i64 1, i64 32, i64 14, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str228 = internal constant [26 x i8] c"notificationForMarkedText\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.derivedtype229 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([26 x i8]* @.str228, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 34, i64 1, i64 32, i64 15, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str230 = internal constant [22 x i8] c"reservedTextFieldCell\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.derivedtype231 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([22 x i8]* @.str230, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 35, i64 16, i64 32, i64 16, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array232 = internal constant [11 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype211 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype213 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype215 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype217 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype219 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype221 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype223 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype225 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype227 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype229 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype231 to { }*)], section "llvm.metadata"		; <[11 x { }*]*> [#uses=1]
@llvm.dbg.composite233 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str208, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 24, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([11 x { }*]* @llvm.dbg.array232 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str234 = internal constant [9 x i8] c"_tfFlags\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype235 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str234, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 36, i64 32, i64 32, i64 352, i32 2, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite233 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array236 = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype195 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype205 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype207 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype235 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite237 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str27, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit26 to { }*), i32 20, i64 384, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array236 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype238 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite237 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str239 = internal constant [10 x i8] c"NSImage.h\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.compile_unit240 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([10 x i8]* @.str239, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str241 = internal constant [8 x i8] c"NSImage\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype243 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str244 = internal constant [11 x i8] c"NSString.h\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.compile_unit245 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([11 x i8]* @.str244, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str246 = internal constant [9 x i8] c"NSString\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype248 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array249 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype248 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite250 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str246, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit245 to { }*), i32 84, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array249 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype251 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite250 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str252 = internal constant [6 x i8] c"_name\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype253 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([6 x i8]* @.str252, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 50, i64 32, i64 32, i64 32, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype251 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str254 = internal constant [13 x i8] c"CGGeometry.h\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str255 = internal constant [110 x i8] c"/System/Library/Frameworks/ApplicationServices.framework/Headers/../Frameworks/CoreGraphics.framework/Headers\00", section "llvm.metadata"		; <[110 x i8]*> [#uses=1]
@llvm.dbg.compile_unit256 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([13 x i8]* @.str254, i32 0, i32 0), i8* getelementptr ([110 x i8]* @.str255, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str257 = internal constant [7 x i8] c"CGSize\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str259 = internal constant [7 x i8] c"double\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.basictype260 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str259, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i32 0, i64 64, i64 32, i64 0, i32 0, i32 4 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str261 = internal constant [9 x i8] c"CGBase.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit262 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([9 x i8]* @.str261, i32 0, i32 0), i8* getelementptr ([110 x i8]* @.str255, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str263 = internal constant [8 x i8] c"CGFloat\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype264 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str263, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit262 to { }*), i32 105, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype260 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str265 = internal constant [6 x i8] c"width\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype266 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([6 x i8]* @.str265, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 22, i64 64, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str267 = internal constant [7 x i8] c"height\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype268 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str267, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 23, i64 64, i64 32, i64 64, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array269 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype266 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype268 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite270 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str257, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 21, i64 128, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array269 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype271 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str257, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 25, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite270 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str272 = internal constant [13 x i8] c"NSGeometry.h\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.compile_unit273 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([13 x i8]* @.str272, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str274 = internal constant [7 x i8] c"NSSize\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype275 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str274, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 26, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype271 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str276 = internal constant [6 x i8] c"_size\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype277 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([6 x i8]* @.str276, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 51, i64 128, i64 32, i64 64, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str278 = internal constant [13 x i8] c"__imageFlags\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str280 = internal constant [9 x i8] c"scalable\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype281 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([9 x i8]* @.str280, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 53, i64 1, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str282 = internal constant [13 x i8] c"dataRetained\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype283 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str282, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 54, i64 1, i64 32, i64 1, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str284 = internal constant [13 x i8] c"uniqueWindow\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype285 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str284, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 55, i64 1, i64 32, i64 2, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str286 = internal constant [21 x i8] c"sizeWasExplicitlySet\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype287 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([21 x i8]* @.str286, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 56, i64 1, i64 32, i64 3, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str288 = internal constant [8 x i8] c"builtIn\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype289 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str288, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 57, i64 1, i64 32, i64 4, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str290 = internal constant [14 x i8] c"needsToExpand\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype291 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str290, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 58, i64 1, i64 32, i64 5, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str292 = internal constant [27 x i8] c"useEPSOnResolutionMismatch\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.derivedtype293 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([27 x i8]* @.str292, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 59, i64 1, i64 32, i64 6, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str294 = internal constant [20 x i8] c"colorMatchPreferred\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype295 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([20 x i8]* @.str294, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 60, i64 1, i64 32, i64 7, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str296 = internal constant [27 x i8] c"multipleResolutionMatching\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.derivedtype297 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([27 x i8]* @.str296, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 61, i64 1, i64 32, i64 8, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str298 = internal constant [21 x i8] c"focusedWhilePrinting\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype299 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([21 x i8]* @.str298, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 62, i64 1, i64 32, i64 9, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str300 = internal constant [14 x i8] c"archiveByName\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype301 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([14 x i8]* @.str300, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 63, i64 1, i64 32, i64 10, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str302 = internal constant [20 x i8] c"unboundedCacheDepth\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype303 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([20 x i8]* @.str302, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 64, i64 1, i64 32, i64 11, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str304 = internal constant [8 x i8] c"flipped\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype305 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str304, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 65, i64 1, i64 32, i64 12, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str306 = internal constant [8 x i8] c"aliased\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype307 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str306, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 66, i64 1, i64 32, i64 13, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str308 = internal constant [8 x i8] c"dirtied\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype309 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str308, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 67, i64 1, i64 32, i64 14, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str310 = internal constant [10 x i8] c"cacheMode\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype311 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str310, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 68, i64 2, i64 32, i64 15, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str312 = internal constant [11 x i8] c"sampleMode\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype313 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str312, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 69, i64 3, i64 32, i64 17, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str314 = internal constant [10 x i8] c"reserved2\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype315 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str314, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 70, i64 1, i64 32, i64 20, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str316 = internal constant [11 x i8] c"isTemplate\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype317 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([11 x i8]* @.str316, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 71, i64 1, i64 32, i64 21, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str318 = internal constant [15 x i8] c"failedToExpand\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype319 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([15 x i8]* @.str318, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 72, i64 1, i64 32, i64 22, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str320 = internal constant [10 x i8] c"reserved1\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype321 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([10 x i8]* @.str320, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 73, i64 9, i64 32, i64 23, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array322 = internal constant [21 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype281 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype283 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype287 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype289 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype291 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype293 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype295 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype297 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype299 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype301 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype303 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype305 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype307 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype309 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype311 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype313 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype315 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype317 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype319 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype321 to { }*)], section "llvm.metadata"		; <[21 x { }*]*> [#uses=1]
@llvm.dbg.composite323 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([13 x i8]* @.str278, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 52, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([21 x { }*]* @llvm.dbg.array322 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str324 = internal constant [7 x i8] c"_flags\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype325 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 74, i64 32, i64 32, i64 192, i32 2, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite323 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str326 = internal constant [6 x i8] c"_reps\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype327 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([6 x i8]* @.str326, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 75, i64 32, i64 32, i64 224, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str328 = internal constant [18 x i8] c"_NSImageAuxiliary\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.composite329 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([18 x i8]* @.str328, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 40, i64 0, i64 0, i64 0, i32 0, { }* null, { }* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype330 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite329 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str331 = internal constant [16 x i8] c"_imageAuxiliary\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype332 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([16 x i8]* @.str331, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 76, i64 32, i64 32, i64 256, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype330 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array333 = internal constant [6 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype243 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype253 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype277 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype325 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype327 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype332 to { }*)], section "llvm.metadata"		; <[6 x { }*]*> [#uses=1]
@llvm.dbg.composite334 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([8 x i8]* @.str241, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit240 to { }*), i32 42, i64 288, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([6 x { }*]* @llvm.dbg.array333 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype335 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite334 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str336 = internal constant [5 x i8] c"icon\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype337 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([5 x i8]* @.str336, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit22 to { }*), i32 15, i64 32, i64 32, i64 384, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype335 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str338 = internal constant [18 x i8] c"preferredIconSize\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype339 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([18 x i8]* @.str338, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit22 to { }*), i32 16, i64 128, i64 32, i64 416, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array340 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype238 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype337 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype339 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite341 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* getelementptr ([18 x i8]* @.str23, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit22 to { }*), i32 12, i64 544, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array340 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype342 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit20 to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite341 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str343 = internal constant [5 x i8] c"self\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str344 = internal constant [5 x i8] c"_cmd\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable345 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str346 = internal constant [7 x i8] c"string\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.variable347 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([7 x i8]* @.str346, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 15, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype251 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_" = internal global [16 x i8] c"NSTextFieldCell\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[16 x i8]*> [#uses=2]
@"\01L_OBJC_CLASS_REFERENCES_" = internal global %struct._objc_class* bitcast ([16 x i8]* @"\01L_OBJC_CLASS_NAME_" to %struct._objc_class*), section "__OBJC,__cls_refs,literal_pointers,no_dead_strip", align 4		; <%struct._objc_class**> [#uses=6]
@"\01L_OBJC_METH_VAR_NAME_" = internal global [14 x i8] c"initTextCell:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[14 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global %struct.objc_selector* bitcast ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@"\01L_OBJC_METH_VAR_NAME_348" = internal global [18 x i8] c"setLineBreakMode:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[18 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_349" = internal global %struct.objc_selector* bitcast ([18 x i8]* @"\01L_OBJC_METH_VAR_NAME_348" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_350" = internal global [18 x i8] c"setFocusRingType:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[18 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_351" = internal global %struct.objc_selector* bitcast ([18 x i8]* @"\01L_OBJC_METH_VAR_NAME_350" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@.str352 = internal constant [27 x i8] c"\01-[DVIconAndTextCell init]\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram353 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([27 x i8]* @.str352, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str352, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 23, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable354 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram353 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable355 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram353 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@__CFConstantStringClassReference = external global [0 x i32]		; <[0 x i32]*> [#uses=1]
@"\01LC" = internal constant [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals"		; <[1 x i8]*> [#uses=1]
@"\01LC356" = internal constant %struct.NSConstantString <{ i32* getelementptr ([0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr ([1 x i8]* @"\01LC", i32 0, i32 0), i32 0 }>, section "__DATA,__cfstring"		; <%struct.NSConstantString*> [#uses=1]
@.str357 = internal constant [36 x i8] c"\01-[DVIconAndTextCell copyWithZone:]\00", section "llvm.metadata"		; <[36 x i8]*> [#uses=1]
@llvm.dbg.subprogram358 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([36 x i8]* @.str357, i32 0, i32 0), i8* getelementptr ([36 x i8]* @.str357, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 27, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable359 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram358 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable360 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram358 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str361 = internal constant [9 x i8] c"NSZone.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit362 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([9 x i8]* @.str361, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str363 = internal constant [8 x i8] c"_NSZone\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str363, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit362 to { }*), i32 10, i64 0, i64 0, i64 0, i32 0, { }* null, { }* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str364 = internal constant [7 x i8] c"NSZone\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype365 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str364, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit362 to { }*), i32 10, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype366 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype365 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str367 = internal constant [5 x i8] c"zone\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable368 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram358 to { }*), i8* getelementptr ([5 x i8]* @.str367, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 27, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype366 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str369 = internal constant [5 x i8] c"copy\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable370 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram358 to { }*), i8* getelementptr ([5 x i8]* @.str369, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 28, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_371" = internal global [14 x i8] c"copyWithZone:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[14 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_372" = internal global %struct.objc_selector* bitcast ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_371" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_373" = internal global [7 x i8] c"retain\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[7 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_374" = internal global %struct.objc_selector* bitcast ([7 x i8]* @"\01L_OBJC_METH_VAR_NAME_373" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@.str375 = internal constant [30 x i8] c"\01-[DVIconAndTextCell dealloc]\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram376 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([30 x i8]* @.str375, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str375, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 34, { }* null, i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable377 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram376 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable378 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram376 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_379" = internal global [8 x i8] c"release\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[8 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_380" = internal global %struct.objc_selector* bitcast ([8 x i8]* @"\01L_OBJC_METH_VAR_NAME_379" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@"\01L_OBJC_METH_VAR_NAME_381" = internal global [8 x i8] c"dealloc\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[8 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_382" = internal global %struct.objc_selector* bitcast ([8 x i8]* @"\01L_OBJC_METH_VAR_NAME_381" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@.str383 = internal constant [31 x i8] c"\01-[DVIconAndTextCell setIcon:]\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram384 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([31 x i8]* @.str383, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str383, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 39, { }* null, i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable385 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram384 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable386 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram384 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str387 = internal constant [8 x i8] c"newIcon\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.variable388 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram384 to { }*), i8* getelementptr ([8 x i8]* @.str387, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 39, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype335 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str389 = internal constant [27 x i8] c"\01-[DVIconAndTextCell icon]\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram390 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([27 x i8]* @.str389, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str389, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 46, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype335 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable391 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram390 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable392 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram390 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str393 = internal constant [40 x i8] c"\01-[DVIconAndTextCell preferredIconSize]\00", section "llvm.metadata"		; <[40 x i8]*> [#uses=1]
@llvm.dbg.subprogram394 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([40 x i8]* @.str393, i32 0, i32 0), i8* getelementptr ([40 x i8]* @.str393, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 50, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable395 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram394 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable396 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram394 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str397 = internal constant [44 x i8] c"\01-[DVIconAndTextCell setPreferredIconSize:]\00", section "llvm.metadata"		; <[44 x i8]*> [#uses=1]
@llvm.dbg.subprogram398 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([44 x i8]* @.str397, i32 0, i32 0), i8* getelementptr ([44 x i8]* @.str397, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 54, { }* null, i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable399 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram398 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable400 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram398 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str401 = internal constant [5 x i8] c"size\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable402 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram398 to { }*), i8* getelementptr ([5 x i8]* @.str401, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 54, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str403 = internal constant [41 x i8] c"\01-[DVIconAndTextCell iconSizeForBounds:]\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram404 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([41 x i8]* @.str403, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str403, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 58, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable405 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram404 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable406 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram404 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str407 = internal constant [7 x i8] c"CGRect\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str409 = internal constant [8 x i8] c"CGPoint\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str411 = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype412 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str411, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 14, i64 64, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str413 = internal constant [2 x i8] c"y\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype414 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str413, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 15, i64 64, i64 32, i64 64, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array415 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype412 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype414 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite416 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str409, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 13, i64 128, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array415 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype417 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str409, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 17, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite416 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str418 = internal constant [7 x i8] c"origin\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype419 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str418, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 30, i64 128, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype417 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype420 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str401, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 31, i64 128, i64 32, i64 128, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype271 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array421 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype419 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype420 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite422 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str407, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 29, i64 256, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array421 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype423 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str407, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit256 to { }*), i32 33, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite422 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str424 = internal constant [7 x i8] c"NSRect\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype425 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str424, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 31, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype423 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str426 = internal constant [7 x i8] c"bounds\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.variable427 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram404 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 58, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str428 = internal constant [9 x i8] c"iconSize\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.variable429 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram404 to { }*), i8* getelementptr ([9 x i8]* @.str428, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 59, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_430" = internal global [5 x i8] c"icon\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[5 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_431" = internal global %struct.objc_selector* bitcast ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_430" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=4]
@"\01L_OBJC_METH_VAR_NAME_432" = internal global [5 x i8] c"size\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[5 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_433" = internal global %struct.objc_selector* bitcast ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_432" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@NSZeroSize = external constant %struct.CGPoint		; <%struct.CGPoint*> [#uses=1]
@.str434 = internal constant [42 x i8] c"\01-[DVIconAndTextCell iconInsetForBounds:]\00", section "llvm.metadata"		; <[42 x i8]*> [#uses=1]
@llvm.dbg.subprogram435 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([42 x i8]* @.str434, i32 0, i32 0), i8* getelementptr ([42 x i8]* @.str434, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 66, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable436 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram435 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable437 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram435 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable438 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram435 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 66, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_439" = internal global [19 x i8] c"iconSizeForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[19 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_440" = internal global %struct.objc_selector* bitcast ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_439" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=6]
@.str441 = internal constant [42 x i8] c"\01-[DVIconAndTextCell textInsetForBounds:]\00", section "llvm.metadata"		; <[42 x i8]*> [#uses=1]
@llvm.dbg.subprogram442 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([42 x i8]* @.str441, i32 0, i32 0), i8* getelementptr ([42 x i8]* @.str441, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 70, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable443 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram442 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable444 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram442 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable445 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram442 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 70, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str446 = internal constant [21 x i8] c"NSAttributedString.h\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.compile_unit447 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([21 x i8]* @.str446, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str448 = internal constant [19 x i8] c"NSAttributedString\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype450 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array451 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype450 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite452 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str448, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit447 to { }*), i32 8, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array451 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype453 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite452 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str454 = internal constant [38 x i8] c"\01-[DVIconAndTextCell attributedTitle]\00", section "llvm.metadata"		; <[38 x i8]*> [#uses=1]
@llvm.dbg.subprogram455 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([38 x i8]* @.str454, i32 0, i32 0), i8* getelementptr ([38 x i8]* @.str454, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 74, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype453 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable456 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram455 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable457 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram455 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str458 = internal constant [26 x i8] c"NSMutableAttributedString\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.derivedtype460 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite452 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array461 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype460 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite462 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([26 x i8]* @.str458, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit447 to { }*), i32 43, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array461 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype463 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite462 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str464 = internal constant [6 x i8] c"title\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.variable465 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram455 to { }*), i8* getelementptr ([6 x i8]* @.str464, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 75, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype463 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_466" = internal global [22 x i8] c"attributedStringValue\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[22 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_467" = internal global %struct.objc_selector* bitcast ([22 x i8]* @"\01L_OBJC_METH_VAR_NAME_466" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_468" = internal global [12 x i8] c"mutableCopy\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[12 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_469" = internal global %struct.objc_selector* bitcast ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_468" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_470" = internal global [12 x i8] c"autorelease\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[12 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_471" = internal global %struct.objc_selector* bitcast ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_470" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@.str472 = internal constant [32 x i8] c"\01-[DVIconAndTextCell titleSize]\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram473 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([32 x i8]* @.str472, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str472, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 79, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable474 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram473 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable475 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram473 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_476" = internal global [16 x i8] c"attributedTitle\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[16 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_477" = internal global %struct.objc_selector* bitcast ([16 x i8]* @"\01L_OBJC_METH_VAR_NAME_476" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@.str478 = internal constant [49 x i8] c"\01-[DVIconAndTextCell titleAndIconRectForBounds:]\00", section "llvm.metadata"		; <[49 x i8]*> [#uses=1]
@llvm.dbg.subprogram479 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([49 x i8]* @.str478, i32 0, i32 0), i8* getelementptr ([49 x i8]* @.str478, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 83, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable480 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable481 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable482 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 83, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str483 = internal constant [10 x i8] c"iconInset\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.variable484 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([10 x i8]* @.str483, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 84, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_485" = internal global [20 x i8] c"iconInsetForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[20 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_486" = internal global %struct.objc_selector* bitcast ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_485" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@.str487 = internal constant [10 x i8] c"textInset\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.variable488 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([10 x i8]* @.str487, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 85, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_489" = internal global [20 x i8] c"textInsetForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[20 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_490" = internal global %struct.objc_selector* bitcast ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_489" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@llvm.dbg.variable491 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([9 x i8]* @.str428, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 86, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str492 = internal constant [9 x i8] c"textSize\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.variable493 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([9 x i8]* @.str492, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 87, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_494" = internal global [10 x i8] c"titleSize\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[10 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_495" = internal global %struct.objc_selector* bitcast ([10 x i8]* @"\01L_OBJC_METH_VAR_NAME_494" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@.str496 = internal constant [14 x i8] c"maxLegalWidth\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.variable497 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([14 x i8]* @.str496, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 88, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str498 = internal constant [17 x i8] c"titleAndIconSize\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.variable499 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([17 x i8]* @.str498, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 89, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable500 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([2 x i8]* @.str411, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 90, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable501 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*), i8* getelementptr ([2 x i8]* @.str413, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 90, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str502 = internal constant [41 x i8] c"\01-[DVIconAndTextCell iconRectForBounds:]\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram503 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([41 x i8]* @.str502, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str502, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 95, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable504 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable505 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable506 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 95, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable507 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*), i8* getelementptr ([9 x i8]* @.str428, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 96, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str508 = internal constant [10 x i8] c"unionRect\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.variable509 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*), i8* getelementptr ([10 x i8]* @.str508, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 97, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_510" = internal global [27 x i8] c"titleAndIconRectForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[27 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_511" = internal global %struct.objc_selector* bitcast ([27 x i8]* @"\01L_OBJC_METH_VAR_NAME_510" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=4]
@.str512 = internal constant [12 x i8] c"signed char\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.basictype513 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str512, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 6 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str514 = internal constant [5 x i8] c"BOOL\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype515 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str514, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to { }*), i32 43, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype513 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str516 = internal constant [8 x i8] c"shorter\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.variable517 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*), i8* getelementptr ([8 x i8]* @.str516, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 98, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype515 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable518 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*), i8* getelementptr ([2 x i8]* @.str413, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 99, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str519 = internal constant [42 x i8] c"\01-[DVIconAndTextCell titleRectForBounds:]\00", section "llvm.metadata"		; <[42 x i8]*> [#uses=1]
@llvm.dbg.subprogram520 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([42 x i8]* @.str519, i32 0, i32 0), i8* getelementptr ([42 x i8]* @.str519, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 103, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable521 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable522 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable523 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 103, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable524 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([10 x i8]* @.str487, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 104, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable525 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([9 x i8]* @.str428, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 105, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str526 = internal constant [10 x i8] c"titleSize\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.variable527 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([10 x i8]* @.str526, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 106, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable528 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([10 x i8]* @.str508, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 107, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str529 = internal constant [10 x i8] c"iconIndet\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.variable530 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([10 x i8]* @.str529, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 108, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable531 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([8 x i8]* @.str516, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 109, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype515 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable532 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([2 x i8]* @.str413, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 110, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable533 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*), i8* getelementptr ([7 x i8]* @.str267, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 111, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str534 = internal constant [54 x i8] c"\01-[DVIconAndTextCell expansionFrameWithFrame:inView:]\00", section "llvm.metadata"		; <[54 x i8]*> [#uses=1]
@llvm.dbg.subprogram535 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([54 x i8]* @.str534, i32 0, i32 0), i8* getelementptr ([54 x i8]* @.str534, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 115, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable536 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram535 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable537 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram535 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str538 = internal constant [10 x i8] c"cellFrame\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.variable539 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram535 to { }*), i8* getelementptr ([10 x i8]* @.str538, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 115, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str540 = internal constant [9 x i8] c"NSView.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit541 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([9 x i8]* @.str540, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str542 = internal constant [7 x i8] c"NSView\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str544 = internal constant [14 x i8] c"NSResponder.h\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.compile_unit545 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([14 x i8]* @.str544, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str546 = internal constant [12 x i8] c"NSResponder\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype548 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str549 = internal constant [15 x i8] c"_nextResponder\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype550 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str549, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit545 to { }*), i32 15, i64 32, i64 32, i64 32, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array551 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype548 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype550 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite552 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str546, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit545 to { }*), i32 12, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array551 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype553 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite552 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str554 = internal constant [7 x i8] c"_frame\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype555 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str554, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 128, i64 256, i64 32, i64 64, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str556 = internal constant [8 x i8] c"_bounds\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype557 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str556, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 129, i64 256, i64 32, i64 320, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str558 = internal constant [11 x i8] c"_superview\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype559 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str558, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 130, i64 32, i64 32, i64 576, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str560 = internal constant [10 x i8] c"_subviews\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype561 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str560, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 131, i64 32, i64 32, i64 608, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str562 = internal constant [11 x i8] c"NSWindow.h\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.compile_unit563 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([11 x i8]* @.str562, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str564 = internal constant [9 x i8] c"NSWindow\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype566 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite552 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype567 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str554, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 164, i64 256, i64 32, i64 64, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str568 = internal constant [13 x i8] c"_contentView\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype569 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str568, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 165, i64 32, i64 32, i64 320, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str570 = internal constant [10 x i8] c"_delegate\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype571 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str570, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 166, i64 32, i64 32, i64 352, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype572 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite552 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str573 = internal constant [16 x i8] c"_firstResponder\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype574 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str573, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 167, i64 32, i64 32, i64 384, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype572 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype575 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite873 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str576 = internal constant [13 x i8] c"_lastLeftHit\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype577 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str576, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 168, i64 32, i64 32, i64 416, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype575 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str578 = internal constant [14 x i8] c"_lastRightHit\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype579 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str578, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 169, i64 32, i64 32, i64 448, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype575 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str580 = internal constant [13 x i8] c"_counterpart\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype581 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str580, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 170, i64 32, i64 32, i64 480, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str582 = internal constant [13 x i8] c"_fieldEditor\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype583 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str582, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 171, i64 32, i64 32, i64 512, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str584 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype585 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str584, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str586 = internal constant [14 x i8] c"_winEventMask\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype587 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str586, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 172, i64 32, i64 32, i64 544, i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype585 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str588 = internal constant [11 x i8] c"_windowNum\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype589 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str588, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 173, i64 32, i64 32, i64 576, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype179 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str590 = internal constant [7 x i8] c"_level\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype591 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str590, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 174, i64 32, i64 32, i64 608, i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype585 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype592 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str204, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 175, i64 32, i64 32, i64 640, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype203 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str593 = internal constant [12 x i8] c"_borderView\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype594 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str593, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 176, i64 32, i64 32, i64 672, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str595 = internal constant [14 x i8] c"unsigned char\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.basictype596 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str595, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 8 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str597 = internal constant [17 x i8] c"_postingDisabled\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype598 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str597, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 177, i64 8, i64 8, i64 704, i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype596 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str599 = internal constant [11 x i8] c"_styleMask\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype600 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str599, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 178, i64 8, i64 8, i64 712, i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype596 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str601 = internal constant [15 x i8] c"_flushDisabled\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype602 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str601, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 179, i64 8, i64 8, i64 720, i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype596 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str603 = internal constant [17 x i8] c"_reservedWindow1\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype604 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str603, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 180, i64 8, i64 8, i64 728, i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype596 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype605 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str606 = internal constant [13 x i8] c"_cursorRects\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype607 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str606, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 181, i64 32, i64 32, i64 736, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype605 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str608 = internal constant [12 x i8] c"_trectTable\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype609 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str608, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 182, i64 32, i64 32, i64 768, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype605 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str610 = internal constant [10 x i8] c"_miniIcon\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype611 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str610, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 183, i64 32, i64 32, i64 800, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype335 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str612 = internal constant [8 x i8] c"_unused\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype613 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str612, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 184, i64 32, i64 32, i64 832, i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype585 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str614 = internal constant [8 x i8] c"NSSet.h\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.compile_unit615 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([8 x i8]* @.str614, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str616 = internal constant [13 x i8] c"NSMutableSet\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str618 = internal constant [6 x i8] c"NSSet\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype620 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array621 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype620 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite622 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str618, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit615 to { }*), i32 12, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array621 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype623 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite622 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array624 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype623 to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite625 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str616, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit615 to { }*), i32 67, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array624 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype626 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite625 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str627 = internal constant [11 x i8] c"_dragTypes\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype628 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str627, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 185, i64 32, i64 32, i64 864, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype626 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str629 = internal constant [8 x i8] c"NSURL.h\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.compile_unit630 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([8 x i8]* @.str629, i32 0, i32 0), i8* getelementptr ([56 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str631 = internal constant [6 x i8] c"NSURL\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype633 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str634 = internal constant [11 x i8] c"_urlString\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype635 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str634, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit630 to { }*), i32 39, i64 32, i64 32, i64 32, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype251 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype636 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite645 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str637 = internal constant [9 x i8] c"_baseURL\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype638 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str637, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit630 to { }*), i32 40, i64 32, i64 32, i64 64, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype636 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str639 = internal constant [9 x i8] c"_clients\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype640 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str639, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit630 to { }*), i32 41, i64 32, i64 32, i64 96, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype605 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype641 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str642 = internal constant [10 x i8] c"_reserved\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype643 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str642, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit630 to { }*), i32 42, i64 32, i64 32, i64 128, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype641 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array644 = internal constant [5 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype633 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype635 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype638 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype640 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype643 to { }*)], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite645 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str631, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit630 to { }*), i32 36, i64 160, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array644 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype646 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite645 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str647 = internal constant [16 x i8] c"_representedURL\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype648 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str647, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 186, i64 32, i64 32, i64 896, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype646 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype649 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str650 = internal constant [12 x i8] c"_sizeLimits\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype651 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str650, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 187, i64 32, i64 32, i64 928, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype649 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str652 = internal constant [15 x i8] c"_frameSaveName\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype653 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str652, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 188, i64 32, i64 32, i64 960, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype251 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype654 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite622 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str655 = internal constant [14 x i8] c"_regDragTypes\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype656 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str655, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 189, i64 32, i64 32, i64 992, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype654 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str657 = internal constant [9 x i8] c"__wFlags\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str659 = internal constant [8 x i8] c"backing\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype660 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str659, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 191, i64 2, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str661 = internal constant [8 x i8] c"visible\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype662 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str661, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 192, i64 1, i64 32, i64 2, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str663 = internal constant [13 x i8] c"isMainWindow\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype664 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str663, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 193, i64 1, i64 32, i64 3, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str665 = internal constant [12 x i8] c"isKeyWindow\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype666 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str665, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 194, i64 1, i64 32, i64 4, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str667 = internal constant [18 x i8] c"hidesOnDeactivate\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype668 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str667, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 195, i64 1, i64 32, i64 5, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str669 = internal constant [19 x i8] c"dontFreeWhenClosed\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype670 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str669, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 196, i64 1, i64 32, i64 6, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str671 = internal constant [8 x i8] c"oneShot\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype672 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str671, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 197, i64 1, i64 32, i64 7, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str673 = internal constant [9 x i8] c"deferred\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype674 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str673, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 198, i64 1, i64 32, i64 8, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str675 = internal constant [20 x i8] c"cursorRectsDisabled\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype676 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([20 x i8]* @.str675, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 199, i64 1, i64 32, i64 9, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str677 = internal constant [20 x i8] c"haveFreeCursorRects\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype678 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([20 x i8]* @.str677, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 200, i64 1, i64 32, i64 10, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str679 = internal constant [17 x i8] c"validCursorRects\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype680 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str679, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 201, i64 1, i64 32, i64 11, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str681 = internal constant [10 x i8] c"docEdited\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype682 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str681, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 202, i64 1, i64 32, i64 12, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str683 = internal constant [18 x i8] c"dynamicDepthLimit\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype684 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str683, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 203, i64 1, i64 32, i64 13, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str685 = internal constant [15 x i8] c"worksWhenModal\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype686 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str685, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 204, i64 1, i64 32, i64 14, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str687 = internal constant [17 x i8] c"limitedBecomeKey\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype688 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str687, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 205, i64 1, i64 32, i64 15, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str689 = internal constant [11 x i8] c"needsFlush\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype690 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str689, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 206, i64 1, i64 32, i64 16, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str691 = internal constant [17 x i8] c"viewsNeedDisplay\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype692 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str691, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 207, i64 1, i64 32, i64 17, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str693 = internal constant [18 x i8] c"ignoredFirstMouse\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype694 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str693, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 208, i64 1, i64 32, i64 18, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str695 = internal constant [19 x i8] c"repostedFirstMouse\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype696 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str695, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 209, i64 1, i64 32, i64 19, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str697 = internal constant [12 x i8] c"windowDying\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype698 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str697, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 210, i64 1, i64 32, i64 20, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str699 = internal constant [11 x i8] c"tempHidden\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype700 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str699, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 211, i64 1, i64 32, i64 21, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str701 = internal constant [14 x i8] c"floatingPanel\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype702 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str701, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 212, i64 1, i64 32, i64 22, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str703 = internal constant [22 x i8] c"wantsToBeOnMainScreen\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.derivedtype704 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([22 x i8]* @.str703, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 213, i64 1, i64 32, i64 23, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str705 = internal constant [19 x i8] c"optimizedDrawingOk\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype706 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str705, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 214, i64 1, i64 32, i64 24, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str707 = internal constant [16 x i8] c"optimizeDrawing\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype708 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str707, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 215, i64 1, i64 32, i64 25, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str709 = internal constant [27 x i8] c"titleIsRepresentedFilename\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.derivedtype710 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([27 x i8]* @.str709, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 216, i64 1, i64 32, i64 26, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str711 = internal constant [24 x i8] c"excludedFromWindowsMenu\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.derivedtype712 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([24 x i8]* @.str711, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 217, i64 1, i64 32, i64 27, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str713 = internal constant [11 x i8] c"depthLimit\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype714 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str713, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 218, i64 4, i64 32, i64 28, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str715 = internal constant [30 x i8] c"delegateReturnsValidRequestor\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.derivedtype716 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([30 x i8]* @.str715, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 219, i64 1, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str717 = internal constant [16 x i8] c"lmouseupPending\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype718 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str717, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 220, i64 1, i64 32, i64 33, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str719 = internal constant [16 x i8] c"rmouseupPending\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype720 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str719, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 221, i64 1, i64 32, i64 34, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str721 = internal constant [25 x i8] c"wantsToDestroyRealWindow\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.derivedtype722 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([25 x i8]* @.str721, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 222, i64 1, i64 32, i64 35, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str723 = internal constant [20 x i8] c"wantsToRegDragTypes\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype724 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([20 x i8]* @.str723, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 223, i64 1, i64 32, i64 36, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str725 = internal constant [29 x i8] c"sentInvalidateCursorRectsMsg\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.derivedtype726 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([29 x i8]* @.str725, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 224, i64 1, i64 32, i64 37, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str727 = internal constant [17 x i8] c"avoidsActivation\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype728 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str727, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 225, i64 1, i64 32, i64 38, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str729 = internal constant [21 x i8] c"frameSavedUsingTitle\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype730 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([21 x i8]* @.str729, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 226, i64 1, i64 32, i64 39, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str731 = internal constant [16 x i8] c"didRegDragTypes\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype732 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str731, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 227, i64 1, i64 32, i64 40, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str733 = internal constant [15 x i8] c"delayedOneShot\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype734 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str733, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 228, i64 1, i64 32, i64 41, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str735 = internal constant [23 x i8] c"postedNeedsDisplayNote\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.derivedtype736 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([23 x i8]* @.str735, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 229, i64 1, i64 32, i64 42, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str737 = internal constant [29 x i8] c"postedInvalidCursorRectsNote\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.derivedtype738 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([29 x i8]* @.str737, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 230, i64 1, i64 32, i64 43, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str739 = internal constant [29 x i8] c"initialFirstResponderTempSet\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.derivedtype740 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([29 x i8]* @.str739, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 231, i64 1, i64 32, i64 44, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str741 = internal constant [12 x i8] c"autodisplay\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype742 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str741, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 232, i64 1, i64 32, i64 45, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str743 = internal constant [17 x i8] c"tossedFirstEvent\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype744 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str743, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 233, i64 1, i64 32, i64 46, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str745 = internal constant [13 x i8] c"isImageCache\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype746 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str745, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 234, i64 1, i64 32, i64 47, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str747 = internal constant [15 x i8] c"interfaceStyle\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype748 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str747, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 235, i64 3, i64 32, i64 48, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str749 = internal constant [26 x i8] c"keyViewSelectionDirection\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.derivedtype750 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([26 x i8]* @.str749, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 236, i64 2, i64 32, i64 51, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str751 = internal constant [39 x i8] c"defaultButtonCellKETemporarilyDisabled\00", section "llvm.metadata"		; <[39 x i8]*> [#uses=1]
@llvm.dbg.derivedtype752 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([39 x i8]* @.str751, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 237, i64 1, i64 32, i64 53, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str753 = internal constant [28 x i8] c"defaultButtonCellKEDisabled\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.derivedtype754 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([28 x i8]* @.str753, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 238, i64 1, i64 32, i64 54, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str755 = internal constant [15 x i8] c"menuHasBeenSet\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype756 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str755, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 239, i64 1, i64 32, i64 55, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str757 = internal constant [15 x i8] c"wantsToBeModal\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype758 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str757, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 240, i64 1, i64 32, i64 56, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str759 = internal constant [18 x i8] c"showingModalFrame\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype760 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str759, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 241, i64 1, i64 32, i64 57, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str761 = internal constant [14 x i8] c"isTerminating\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype762 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str761, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 242, i64 1, i64 32, i64 58, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str763 = internal constant [31 x i8] c"win32MouseActivationInProgress\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.derivedtype764 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([31 x i8]* @.str763, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 243, i64 1, i64 32, i64 59, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str765 = internal constant [33 x i8] c"makingFirstResponderForMouseDown\00", section "llvm.metadata"		; <[33 x i8]*> [#uses=1]
@llvm.dbg.derivedtype766 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([33 x i8]* @.str765, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 244, i64 1, i64 32, i64 60, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str767 = internal constant [10 x i8] c"needsZoom\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype768 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str767, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 245, i64 1, i64 32, i64 61, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str769 = internal constant [26 x i8] c"sentWindowNeedsDisplayMsg\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.derivedtype770 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([26 x i8]* @.str769, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 246, i64 1, i64 32, i64 62, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str771 = internal constant [17 x i8] c"liveResizeActive\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype772 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str771, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 247, i64 1, i64 32, i64 63, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array773 = internal constant [57 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype660 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype662 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype664 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype666 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype668 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype670 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype672 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype674 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype676 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype678 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype680 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype682 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype684 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype686 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype688 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype690 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype692 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype694 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype696 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype698 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype700 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype702 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype704 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype706 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype708 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype710 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype712 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype714 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype716 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype718 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype720 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype722 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype724 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype726 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype728 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype730 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype732 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype734 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype736 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype738 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype740 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype742 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype744 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype746 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype748 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype750 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype752 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype754 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype756 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype758 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype760 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype762 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype764 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype766 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype768 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype770 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype772 to { }*)], section "llvm.metadata"		; <[57 x { }*]*> [#uses=1]
@llvm.dbg.composite774 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str657, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 190, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([57 x { }*]* @llvm.dbg.array773 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str775 = internal constant [8 x i8] c"_wFlags\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype776 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str775, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 248, i64 64, i64 32, i64 1024, i32 2, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite774 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str777 = internal constant [19 x i8] c"_defaultButtonCell\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype778 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str777, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 249, i64 32, i64 32, i64 1088, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str779 = internal constant [23 x i8] c"_initialFirstResponder\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.derivedtype780 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([23 x i8]* @.str779, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 250, i64 32, i64 32, i64 1120, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype575 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str781 = internal constant [18 x i8] c"NSWindowAuxiliary\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.composite782 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str781, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 152, i64 0, i64 0, i64 0, i32 0, { }* null, { }* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype783 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite782 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str784 = internal constant [18 x i8] c"_auxiliaryStorage\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype785 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str784, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 251, i64 32, i64 32, i64 1152, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype783 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array786 = internal constant [31 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype566 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype567 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype569 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype571 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype574 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype577 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype579 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype581 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype583 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype587 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype589 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype591 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype592 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype594 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype598 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype600 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype602 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype604 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype607 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype609 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype611 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype613 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype628 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype648 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype651 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype653 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype656 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype776 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype778 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype780 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype785 to { }*)], section "llvm.metadata"		; <[31 x { }*]*> [#uses=1]
@llvm.dbg.composite787 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str564, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit563 to { }*), i32 156, i64 1184, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([31 x { }*]* @llvm.dbg.array786 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype788 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite787 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str789 = internal constant [8 x i8] c"_window\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype790 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str789, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 132, i64 32, i64 32, i64 640, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype788 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str791 = internal constant [8 x i8] c"_gState\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype792 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str791, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 133, i64 32, i64 32, i64 672, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str793 = internal constant [13 x i8] c"_frameMatrix\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype794 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str793, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 134, i64 32, i64 32, i64 704, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str795 = internal constant [12 x i8] c"_drawMatrix\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype796 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str795, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 135, i64 32, i64 32, i64 736, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype797 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str627, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 136, i64 32, i64 32, i64 768, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str798 = internal constant [17 x i8] c"_NSViewAuxiliary\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.composite799 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str798, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 120, i64 0, i64 0, i64 0, i32 0, { }* null, { }* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype800 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite799 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str801 = internal constant [15 x i8] c"_viewAuxiliary\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype802 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str801, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 137, i64 32, i64 32, i64 800, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype800 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str803 = internal constant [9 x i8] c"__VFlags\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str805 = internal constant [14 x i8] c"aboutToResize\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype806 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str805, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 92, i64 1, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str807 = internal constant [19 x i8] c"retainCountOverMax\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype808 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str807, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 93, i64 1, i64 32, i64 1, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str809 = internal constant [12 x i8] c"retainCount\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype810 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str809, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 94, i64 6, i64 32, i64 2, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str811 = internal constant [16 x i8] c"interfaceStyle1\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype812 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str811, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 95, i64 1, i64 32, i64 8, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str813 = internal constant [17 x i8] c"specialArchiving\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype814 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str813, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 96, i64 1, i64 32, i64 9, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str815 = internal constant [22 x i8] c"needsDisplayForBounds\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.derivedtype816 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([22 x i8]* @.str815, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 97, i64 1, i64 32, i64 10, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str817 = internal constant [16 x i8] c"interfaceStyle0\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype818 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str817, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 98, i64 1, i64 32, i64 11, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str819 = internal constant [28 x i8] c"removingWithoutInvalidation\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.derivedtype820 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([28 x i8]* @.str819, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 99, i64 1, i64 32, i64 12, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str821 = internal constant [22 x i8] c"needsBoundsChangeNote\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.derivedtype822 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([22 x i8]* @.str821, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 100, i64 1, i64 32, i64 13, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str823 = internal constant [27 x i8] c"boundsChangeNotesSuspended\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.derivedtype824 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([27 x i8]* @.str823, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 101, i64 1, i64 32, i64 14, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str825 = internal constant [26 x i8] c"focusChangeNotesSuspended\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.derivedtype826 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([26 x i8]* @.str825, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 102, i64 1, i64 32, i64 15, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str827 = internal constant [21 x i8] c"needsFrameChangeNote\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype828 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([21 x i8]* @.str827, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 103, i64 1, i64 32, i64 16, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str829 = internal constant [26 x i8] c"frameChangeNotesSuspended\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.derivedtype830 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([26 x i8]* @.str829, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 104, i64 1, i64 32, i64 17, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str831 = internal constant [21 x i8] c"noVerticalAutosizing\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype832 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([21 x i8]* @.str831, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 105, i64 1, i64 32, i64 18, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str833 = internal constant [10 x i8] c"newGState\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype834 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str833, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 106, i64 1, i64 32, i64 19, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str835 = internal constant [12 x i8] c"validGState\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype836 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str835, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 107, i64 1, i64 32, i64 20, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str837 = internal constant [13 x i8] c"needsDisplay\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype838 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str837, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 108, i64 1, i64 32, i64 21, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str839 = internal constant [12 x i8] c"wantsGState\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype840 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str839, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 109, i64 1, i64 32, i64 22, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str841 = internal constant [19 x i8] c"autoresizeSubviews\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype842 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str841, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 110, i64 1, i64 32, i64 23, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str843 = internal constant [11 x i8] c"autosizing\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype844 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str843, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 111, i64 6, i64 32, i64 24, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str845 = internal constant [24 x i8] c"rotatedOrScaledFromBase\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.derivedtype846 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([24 x i8]* @.str845, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 112, i64 1, i64 32, i64 30, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str847 = internal constant [16 x i8] c"rotatedFromBase\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype848 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str847, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 113, i64 1, i64 32, i64 31, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array849 = internal constant [22 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype806 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype808 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype810 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype812 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype814 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype816 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype818 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype820 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype822 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype824 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype826 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype828 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype830 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype832 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype834 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype836 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype838 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype840 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype842 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype844 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype846 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype848 to { }*)], section "llvm.metadata"		; <[22 x { }*]*> [#uses=1]
@llvm.dbg.composite850 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str803, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 67, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([22 x { }*]* @llvm.dbg.array849 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str851 = internal constant [8 x i8] c"_VFlags\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype852 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str851, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 115, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite850 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str853 = internal constant [8 x i8] c"_vFlags\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype854 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str853, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 138, i64 32, i64 32, i64 832, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype852 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str855 = internal constant [10 x i8] c"__VFlags2\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str857 = internal constant [20 x i8] c"nextKeyViewRefCount\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype858 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([20 x i8]* @.str857, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 140, i64 14, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str859 = internal constant [24 x i8] c"previousKeyViewRefCount\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.derivedtype860 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([24 x i8]* @.str859, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 141, i64 14, i64 32, i64 14, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str861 = internal constant [14 x i8] c"isVisibleRect\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype862 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str861, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 142, i64 1, i64 32, i64 28, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str863 = internal constant [11 x i8] c"hasToolTip\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype864 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str863, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 143, i64 1, i64 32, i64 29, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str865 = internal constant [19 x i8] c"needsRealLockFocus\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype866 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str865, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 144, i64 1, i64 32, i64 30, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype867 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str153, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 145, i64 1, i64 32, i64 31, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array868 = internal constant [6 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype858 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype860 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype862 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype864 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype866 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype867 to { }*)], section "llvm.metadata"		; <[6 x { }*]*> [#uses=1]
@llvm.dbg.composite869 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str855, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 139, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([6 x { }*]* @llvm.dbg.array868 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str870 = internal constant [9 x i8] c"_vFlags2\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype871 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str870, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 146, i64 32, i64 32, i64 864, i32 2, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite869 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array872 = internal constant [13 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype553 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype555 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype557 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype559 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype561 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype790 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype792 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype794 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype796 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype797 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype802 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype854 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype871 to { }*)], section "llvm.metadata"		; <[13 x { }*]*> [#uses=1]
@llvm.dbg.composite873 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str542, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit541 to { }*), i32 122, i64 896, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([13 x { }*]* @llvm.dbg.array872 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype874 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite873 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str875 = internal constant [5 x i8] c"view\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable876 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram535 to { }*), i8* getelementptr ([5 x i8]* @.str875, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 115, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype874 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str877 = internal constant [15 x i8] c"expansionFrame\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.variable878 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram535 to { }*), i8* getelementptr ([15 x i8]* @.str877, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 116, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_879" = internal global [9 x i8] c"cellSize\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[9 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_880" = internal global %struct.objc_selector* bitcast ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_879" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@NSZeroRect = external constant %struct.CGRect		; <%struct.CGRect*> [#uses=1]
@.str881 = internal constant [41 x i8] c"\01-[DVIconAndTextCell cellSizeForBounds:]\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram882 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([41 x i8]* @.str881, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str881, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 125, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable883 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram882 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable884 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram882 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable885 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram882 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 125, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable886 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram882 to { }*), i8* getelementptr ([10 x i8]* @.str483, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 126, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str887 = internal constant [12 x i8] c"contentSize\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.variable888 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram882 to { }*), i8* getelementptr ([12 x i8]* @.str887, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 127, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str889 = internal constant [31 x i8] c"\01-[DVIconAndTextCell isOpaque]\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram890 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([31 x i8]* @.str889, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str889, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 131, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype515 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable891 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram890 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable892 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram890 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str893 = internal constant [53 x i8] c"\01-[DVIconAndTextCell drawWithExpansionFrame:inView:]\00", section "llvm.metadata"		; <[53 x i8]*> [#uses=1]
@llvm.dbg.subprogram894 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([53 x i8]* @.str893, i32 0, i32 0), i8* getelementptr ([53 x i8]* @.str893, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 135, { }* null, i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable895 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram894 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable896 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram894 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable897 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram894 to { }*), i8* getelementptr ([10 x i8]* @.str538, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 135, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable898 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram894 to { }*), i8* getelementptr ([5 x i8]* @.str875, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 135, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype874 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_899" = internal global [31 x i8] c"drawWithExpansionFrame:inView:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[31 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_900" = internal global %struct.objc_selector* bitcast ([31 x i8]* @"\01L_OBJC_METH_VAR_NAME_899" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@.str901 = internal constant [52 x i8] c"\01-[DVIconAndTextCell drawInteriorWithFrame:inView:]\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.subprogram902 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([52 x i8]* @.str901, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str901, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 141, { }* null, i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable903 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable904 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable905 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 141, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str906 = internal constant [12 x i8] c"controlView\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.variable907 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*), i8* getelementptr ([12 x i8]* @.str906, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 141, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype874 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_908" = internal global [19 x i8] c"iconRectForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[19 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_909" = internal global %struct.objc_selector* bitcast ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_908" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@"\01L_OBJC_METH_VAR_NAME_910" = internal global [10 x i8] c"isFlipped\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[10 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_911" = internal global %struct.objc_selector* bitcast ([10 x i8]* @"\01L_OBJC_METH_VAR_NAME_910" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_912" = internal global [38 x i8] c"drawInRect:operation:fraction:unflip:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[38 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_913" = internal global %struct.objc_selector* bitcast ([38 x i8]* @"\01L_OBJC_METH_VAR_NAME_912" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@.str914 = internal constant [16 x i8] c"attributedTitle\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.variable915 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*), i8* getelementptr ([16 x i8]* @.str914, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 145, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype453 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str916 = internal constant [10 x i8] c"titleRect\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.variable917 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*), i8* getelementptr ([10 x i8]* @.str916, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 146, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_918" = internal global [20 x i8] c"titleRectForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[20 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_919" = internal global %struct.objc_selector* bitcast ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_918" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=3]
@"\01L_OBJC_METH_VAR_NAME_920" = internal global [12 x i8] c"drawInRect:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[12 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_921" = internal global %struct.objc_selector* bitcast ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_920" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@.str922 = internal constant [49 x i8] c"\01-[DVIconAndTextCell titleEditingRectForBounds:]\00", section "llvm.metadata"		; <[49 x i8]*> [#uses=1]
@llvm.dbg.subprogram923 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([49 x i8]* @.str922, i32 0, i32 0), i8* getelementptr ([49 x i8]* @.str922, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 150, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable924 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram923 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable925 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram923 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable926 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram923 to { }*), i8* getelementptr ([7 x i8]* @.str426, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 150, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable927 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram923 to { }*), i8* getelementptr ([10 x i8]* @.str916, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 151, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str928 = internal constant [75 x i8] c"\01-[DVIconAndTextCell selectWithFrame:inView:editor:delegate:start:length:]\00", section "llvm.metadata"		; <[75 x i8]*> [#uses=1]
@llvm.dbg.subprogram929 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([75 x i8]* @.str928, i32 0, i32 0), i8* getelementptr ([75 x i8]* @.str928, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 155, { }* null, i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable930 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([5 x i8]* @.str343, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype342 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable931 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([5 x i8]* @.str344, i32 0, i32 0), { }* null, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype188 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str932 = internal constant [6 x i8] c"frame\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.variable933 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([6 x i8]* @.str932, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 155, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable934 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([12 x i8]* @.str906, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 155, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype874 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str935 = internal constant [7 x i8] c"NSText\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype937 = internal constant %llvm.dbg.derivedtype.type { i32 458780, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite873 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str938 = internal constant [7 x i8] c"_ivars\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype939 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str938, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit126 to { }*), i32 74, i64 32, i64 32, i64 896, i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array940 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype937 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype939 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite941 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str935, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit126 to { }*), i32 72, i64 928, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array940 to { }*), i32 1 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype942 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite941 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str943 = internal constant [5 x i8] c"text\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable944 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([5 x i8]* @.str943, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 155, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype942 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str945 = internal constant [9 x i8] c"delegate\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.variable946 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([9 x i8]* @.str945, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 155, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype16 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str947 = internal constant [6 x i8] c"start\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.variable948 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([6 x i8]* @.str947, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 155, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype179 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str949 = internal constant [7 x i8] c"length\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.variable950 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*), i8* getelementptr ([7 x i8]* @.str949, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 155, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype179 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_951" = internal global [27 x i8] c"titleEditingRectForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[27 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_952" = internal global %struct.objc_selector* bitcast ([27 x i8]* @"\01L_OBJC_METH_VAR_NAME_951" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_953" = internal global [53 x i8] c"selectWithFrame:inView:editor:delegate:start:length:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[53 x i8]*> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_954" = internal global %struct.objc_selector* bitcast ([53 x i8]* @"\01L_OBJC_METH_VAR_NAME_953" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [10 x i8] c"@12@0:4@8\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[10 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_955" = internal global [5 x i8] c"init\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[5 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_956" = internal global [7 x i8] c"@8@0:4\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[7 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_TYPE_957" = internal global [20 x i8] c"@12@0:4^{_NSZone=}8\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[20 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_TYPE_958" = internal global [7 x i8] c"v8@0:4\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[7 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_959" = internal global [9 x i8] c"setIcon:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[9 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_960" = internal global [10 x i8] c"v12@0:4@8\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[10 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_961" = internal global [18 x i8] c"preferredIconSize\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[18 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_962" = internal global [17 x i8] c"{CGSize=dd}8@0:4\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[17 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_963" = internal global [22 x i8] c"setPreferredIconSize:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[22 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_964" = internal global [20 x i8] c"v24@0:4{CGSize=dd}8\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[20 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_TYPE_965" = internal global [51 x i8] c"{CGSize=dd}40@0:4{CGRect={CGPoint=dd}{CGSize=dd}}8\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[51 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_TYPE_966" = internal global [41 x i8] c"d40@0:4{CGRect={CGPoint=dd}{CGSize=dd}}8\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[41 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_TYPE_967" = internal global [72 x i8] c"{CGRect={CGPoint=dd}{CGSize=dd}}40@0:4{CGRect={CGPoint=dd}{CGSize=dd}}8\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[72 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_968" = internal global [32 x i8] c"expansionFrameWithFrame:inView:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[32 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_969" = internal global [75 x i8] c"{CGRect={CGPoint=dd}{CGSize=dd}}44@0:4{CGRect={CGPoint=dd}{CGSize=dd}}8@40\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[75 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_970" = internal global [19 x i8] c"cellSizeForBounds:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[19 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_971" = internal global [9 x i8] c"isOpaque\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[9 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_972" = internal global [7 x i8] c"c8@0:4\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[7 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_TYPE_973" = internal global [44 x i8] c"v44@0:4{CGRect={CGPoint=dd}{CGSize=dd}}8@40\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[44 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_NAME_974" = internal global [30 x i8] c"drawInteriorWithFrame:inView:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[30 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_975" = internal global [56 x i8] c"v60@0:4{CGRect={CGPoint=dd}{CGSize=dd}}8@40@44@48l52l56\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[56 x i8]*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_976" = internal global [9 x i8] c"NSObject\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[9 x i8]*> [#uses=2]
@"\01L_OBJC_CLASS_NAME_977" = internal global [18 x i8] c"DVIconAndTextCell\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[18 x i8]*> [#uses=1]
@"\01L_OBJC_METACLASS_DVIconAndTextCell" = internal global %struct._objc_class { %struct._objc_class* bitcast ([9 x i8]* @"\01L_OBJC_CLASS_NAME_976" to %struct._objc_class*), %struct._objc_class* bitcast ([16 x i8]* @"\01L_OBJC_CLASS_NAME_" to %struct._objc_class*), i8* getelementptr ([18 x i8]* @"\01L_OBJC_CLASS_NAME_977", i32 0, i32 0), i32 0, i32 2, i32 48, %struct._objc_ivar_list* null, %struct._objc_method_list* null, %struct._objc_cache* null, %struct._objc_protocol_list* null, i8* null, %struct._objc_class_extension* null }, section "__OBJC,__meta_class,regular,no_dead_strip", align 4		; <%struct._objc_class*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_978" = internal global [11 x i8] c"@\22NSImage\22\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[11 x i8]*> [#uses=1]
@"\01L_OBJC_METH_VAR_TYPE_979" = internal global [27 x i8] c"{CGSize=\22width\22d\22height\22d}\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[27 x i8]*> [#uses=1]
@"\01L_OBJC_INSTANCE_VARIABLES_DVIconAndTextCell" = internal global %0 { i32 2, [2 x %struct._objc_ivar] [%struct._objc_ivar { i8* getelementptr ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_430", i32 0, i32 0), i8* getelementptr ([11 x i8]* @"\01L_OBJC_METH_VAR_TYPE_978", i32 0, i32 0), i32 48 }, %struct._objc_ivar { i8* getelementptr ([18 x i8]* @"\01L_OBJC_METH_VAR_NAME_961", i32 0, i32 0), i8* getelementptr ([27 x i8]* @"\01L_OBJC_METH_VAR_TYPE_979", i32 0, i32 0), i32 52 }] }, section "__OBJC,__instance_vars,regular,no_dead_strip", align 4		; <%0*> [#uses=2]
@"\01L_OBJC_INSTANCE_METHODS_DVIconAndTextCell" = internal global %1 { i8* null, i32 23, [23 x %struct._objc_method] [%struct._objc_method { %struct.objc_selector* bitcast ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_" to %struct.objc_selector*), i8* getelementptr ([10 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast (%struct.objc_object* (%3*, %struct.objc_selector*, %4*)* @"\01-[DVIconAndTextCell initTextCell:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_955" to %struct.objc_selector*), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_956", i32 0, i32 0), i8* bitcast (%struct.objc_object* (%3*, %struct.objc_selector*)* @"\01-[DVIconAndTextCell init]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_371" to %struct.objc_selector*), i8* getelementptr ([20 x i8]* @"\01L_OBJC_METH_VAR_TYPE_957", i32 0, i32 0), i8* bitcast (%struct.objc_object* (%3*, %struct.objc_selector*, %struct._NSZone*)* @"\01-[DVIconAndTextCell copyWithZone:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([8 x i8]* @"\01L_OBJC_METH_VAR_NAME_381" to %struct.objc_selector*), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_958", i32 0, i32 0), i8* bitcast (void (%3*, %struct.objc_selector*)* @"\01-[DVIconAndTextCell dealloc]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_959" to %struct.objc_selector*), i8* getelementptr ([10 x i8]* @"\01L_OBJC_METH_VAR_TYPE_960", i32 0, i32 0), i8* bitcast (void (%3*, %struct.objc_selector*, %5*)* @"\01-[DVIconAndTextCell setIcon:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_430" to %struct.objc_selector*), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_956", i32 0, i32 0), i8* bitcast (%5* (%3*, %struct.objc_selector*)* @"\01-[DVIconAndTextCell icon]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([18 x i8]* @"\01L_OBJC_METH_VAR_NAME_961" to %struct.objc_selector*), i8* getelementptr ([17 x i8]* @"\01L_OBJC_METH_VAR_TYPE_962", i32 0, i32 0), i8* bitcast (void (%struct.CGPoint*, %3*, %struct.objc_selector*)* @"\01-[DVIconAndTextCell preferredIconSize]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([22 x i8]* @"\01L_OBJC_METH_VAR_NAME_963" to %struct.objc_selector*), i8* getelementptr ([20 x i8]* @"\01L_OBJC_METH_VAR_TYPE_964", i32 0, i32 0), i8* bitcast (void (%3*, %struct.objc_selector*, double, double)* @"\01-[DVIconAndTextCell setPreferredIconSize:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_439" to %struct.objc_selector*), i8* getelementptr ([51 x i8]* @"\01L_OBJC_METH_VAR_TYPE_965", i32 0, i32 0), i8* bitcast (void (%struct.CGPoint*, %3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell iconSizeForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_485" to %struct.objc_selector*), i8* getelementptr ([41 x i8]* @"\01L_OBJC_METH_VAR_TYPE_966", i32 0, i32 0), i8* bitcast (double (%3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell iconInsetForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_489" to %struct.objc_selector*), i8* getelementptr ([41 x i8]* @"\01L_OBJC_METH_VAR_TYPE_966", i32 0, i32 0), i8* bitcast (double (%3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell textInsetForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([16 x i8]* @"\01L_OBJC_METH_VAR_NAME_476" to %struct.objc_selector*), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_956", i32 0, i32 0), i8* bitcast (%6* (%3*, %struct.objc_selector*)* @"\01-[DVIconAndTextCell attributedTitle]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([10 x i8]* @"\01L_OBJC_METH_VAR_NAME_494" to %struct.objc_selector*), i8* getelementptr ([17 x i8]* @"\01L_OBJC_METH_VAR_TYPE_962", i32 0, i32 0), i8* bitcast (void (%struct.CGPoint*, %3*, %struct.objc_selector*)* @"\01-[DVIconAndTextCell titleSize]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([27 x i8]* @"\01L_OBJC_METH_VAR_NAME_510" to %struct.objc_selector*), i8* getelementptr ([72 x i8]* @"\01L_OBJC_METH_VAR_TYPE_967", i32 0, i32 0), i8* bitcast (void (%struct.CGRect*, %3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell titleAndIconRectForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_908" to %struct.objc_selector*), i8* getelementptr ([72 x i8]* @"\01L_OBJC_METH_VAR_TYPE_967", i32 0, i32 0), i8* bitcast (void (%struct.CGRect*, %3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell iconRectForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_918" to %struct.objc_selector*), i8* getelementptr ([72 x i8]* @"\01L_OBJC_METH_VAR_TYPE_967", i32 0, i32 0), i8* bitcast (void (%struct.CGRect*, %3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell titleRectForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([32 x i8]* @"\01L_OBJC_METH_VAR_NAME_968" to %struct.objc_selector*), i8* getelementptr ([75 x i8]* @"\01L_OBJC_METH_VAR_TYPE_969", i32 0, i32 0), i8* bitcast (void (%struct.CGRect*, %3*, %struct.objc_selector*, %struct.CGRect*, %8*)* @"\01-[DVIconAndTextCell expansionFrameWithFrame:inView:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_970" to %struct.objc_selector*), i8* getelementptr ([51 x i8]* @"\01L_OBJC_METH_VAR_TYPE_965", i32 0, i32 0), i8* bitcast (void (%struct.CGPoint*, %3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell cellSizeForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_971" to %struct.objc_selector*), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_972", i32 0, i32 0), i8* bitcast (i8 (%3*, %struct.objc_selector*)* @"\01-[DVIconAndTextCell isOpaque]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([31 x i8]* @"\01L_OBJC_METH_VAR_NAME_899" to %struct.objc_selector*), i8* getelementptr ([44 x i8]* @"\01L_OBJC_METH_VAR_TYPE_973", i32 0, i32 0), i8* bitcast (void (%3*, %struct.objc_selector*, %struct.CGRect*, %8*)* @"\01-[DVIconAndTextCell drawWithExpansionFrame:inView:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([30 x i8]* @"\01L_OBJC_METH_VAR_NAME_974" to %struct.objc_selector*), i8* getelementptr ([44 x i8]* @"\01L_OBJC_METH_VAR_TYPE_973", i32 0, i32 0), i8* bitcast (void (%3*, %struct.objc_selector*, %struct.CGRect*, %8*)* @"\01-[DVIconAndTextCell drawInteriorWithFrame:inView:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([27 x i8]* @"\01L_OBJC_METH_VAR_NAME_951" to %struct.objc_selector*), i8* getelementptr ([72 x i8]* @"\01L_OBJC_METH_VAR_TYPE_967", i32 0, i32 0), i8* bitcast (void (%struct.CGRect*, %3*, %struct.objc_selector*, %struct.CGRect*)* @"\01-[DVIconAndTextCell titleEditingRectForBounds:]" to i8*) }, %struct._objc_method { %struct.objc_selector* bitcast ([53 x i8]* @"\01L_OBJC_METH_VAR_NAME_953" to %struct.objc_selector*), i8* getelementptr ([56 x i8]* @"\01L_OBJC_METH_VAR_TYPE_975", i32 0, i32 0), i8* bitcast (void (%3*, %struct.objc_selector*, %struct.CGRect*, %8*, %9*, %struct.objc_object*, i32, i32)* @"\01-[DVIconAndTextCell selectWithFrame:inView:editor:delegate:start:length:]" to i8*) }] }, section "__OBJC,__inst_meth,regular,no_dead_strip", align 4		; <%1*> [#uses=2]
@"\01L_OBJC_CLASS_DVIconAndTextCell" = internal global %struct._objc_class { %struct._objc_class* @"\01L_OBJC_METACLASS_DVIconAndTextCell", %struct._objc_class* bitcast ([16 x i8]* @"\01L_OBJC_CLASS_NAME_" to %struct._objc_class*), i8* getelementptr ([18 x i8]* @"\01L_OBJC_CLASS_NAME_977", i32 0, i32 0), i32 0, i32 1, i32 68, %struct._objc_ivar_list* bitcast (%0* @"\01L_OBJC_INSTANCE_VARIABLES_DVIconAndTextCell" to %struct._objc_ivar_list*), %struct._objc_method_list* bitcast (%1* @"\01L_OBJC_INSTANCE_METHODS_DVIconAndTextCell" to %struct._objc_method_list*), %struct._objc_cache* null, %struct._objc_protocol_list* null, i8* null, %struct._objc_class_extension* null }, section "__OBJC,__class,regular,no_dead_strip", align 4		; <%struct._objc_class*> [#uses=1]
@.str980 = internal constant [7 x i8] c"NSMaxX\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram981 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i8* getelementptr ([7 x i8]* @.str980, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str980, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 109, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str982 = internal constant [6 x i8] c"aRect\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.variable983 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram981 to { }*), i8* getelementptr ([6 x i8]* @.str982, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 109, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str984 = internal constant [11 x i8] c"NSMakeRect\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram985 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i8* getelementptr ([11 x i8]* @.str984, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str984, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 100, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable986 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*), i8* getelementptr ([2 x i8]* @.str411, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 100, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable987 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*), i8* getelementptr ([2 x i8]* @.str413, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 100, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str988 = internal constant [2 x i8] c"w\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable989 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*), i8* getelementptr ([2 x i8]* @.str988, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 100, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str990 = internal constant [2 x i8] c"h\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable991 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*), i8* getelementptr ([2 x i8]* @.str990, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 100, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str992 = internal constant [2 x i8] c"r\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable993 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*), i8* getelementptr ([2 x i8]* @.str992, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 101, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str994 = internal constant [7 x i8] c"NSMinX\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram995 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i8* getelementptr ([7 x i8]* @.str994, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str994, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 125, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable996 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram995 to { }*), i8* getelementptr ([6 x i8]* @.str982, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 125, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str997 = internal constant [7 x i8] c"NSMinY\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram998 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i8* getelementptr ([7 x i8]* @.str997, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str997, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 129, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable999 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram998 to { }*), i8* getelementptr ([6 x i8]* @.str982, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 129, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1000 = internal constant [9 x i8] c"NSHeight\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.subprogram1001 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i8* getelementptr ([9 x i8]* @.str1000, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str1000, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 137, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable1002 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*), i8* getelementptr ([6 x i8]* @.str982, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 137, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1003 = internal constant [11 x i8] c"NSMakeSize\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram1004 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i8* getelementptr ([11 x i8]* @.str1003, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1003, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 93, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable1005 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1004 to { }*), i8* getelementptr ([2 x i8]* @.str988, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 93, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable1006 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1004 to { }*), i8* getelementptr ([2 x i8]* @.str990, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 93, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1007 = internal constant [2 x i8] c"s\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable1008 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1004 to { }*), i8* getelementptr ([2 x i8]* @.str1007, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 94, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1009 = internal constant [8 x i8] c"NSWidth\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.subprogram1010 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i8* getelementptr ([8 x i8]* @.str1009, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1009, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 133, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype264 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable1011 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1010 to { }*), i8* getelementptr ([6 x i8]* @.str982, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*), i32 133, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype425 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_1012" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 1		; <[1 x i8]*> [#uses=1]
@"\01L_OBJC_SYMBOLS" = internal global %2 { i32 0, %struct.objc_selector* null, i16 1, i16 0, [1 x i8*] [i8* bitcast (%struct._objc_class* @"\01L_OBJC_CLASS_DVIconAndTextCell" to i8*)] }, section "__OBJC,__symbols,regular,no_dead_strip", align 4		; <%2*> [#uses=2]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module { i32 7, i32 16, i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_1012", i32 0, i32 0), %struct._objc_symtab* bitcast (%2* @"\01L_OBJC_SYMBOLS" to %struct._objc_symtab*) }, section "__OBJC,__module_info,regular,no_dead_strip", align 4		; <%struct._objc_module*> [#uses=1]
@llvm.used = appending global [90 x i8*] [i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* getelementptr ([16 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* bitcast (%struct._objc_class** @"\01L_OBJC_CLASS_REFERENCES_" to i8*), i8* getelementptr ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_" to i8*), i8* getelementptr ([18 x i8]* @"\01L_OBJC_METH_VAR_NAME_348", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_349" to i8*), i8* getelementptr ([18 x i8]* @"\01L_OBJC_METH_VAR_NAME_350", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_351" to i8*), i8* getelementptr ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_371", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_372" to i8*), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_NAME_373", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_374" to i8*), i8* getelementptr ([8 x i8]* @"\01L_OBJC_METH_VAR_NAME_379", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_380" to i8*), i8* getelementptr ([8 x i8]* @"\01L_OBJC_METH_VAR_NAME_381", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_382" to i8*), i8* getelementptr ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_430", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_431" to i8*), i8* getelementptr ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_432", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_433" to i8*), i8* getelementptr ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_439", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_440" to i8*), i8* getelementptr ([22 x i8]* @"\01L_OBJC_METH_VAR_NAME_466", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_467" to i8*), i8* getelementptr ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_468", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_469" to i8*), i8* getelementptr ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_470", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_471" to i8*), i8* getelementptr ([16 x i8]* @"\01L_OBJC_METH_VAR_NAME_476", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_477" to i8*), i8* getelementptr ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_485", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_486" to i8*), i8* getelementptr ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_489", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_490" to i8*), i8* getelementptr ([10 x i8]* @"\01L_OBJC_METH_VAR_NAME_494", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_495" to i8*), i8* getelementptr ([27 x i8]* @"\01L_OBJC_METH_VAR_NAME_510", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_511" to i8*), i8* getelementptr ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_879", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_880" to i8*), i8* getelementptr ([31 x i8]* @"\01L_OBJC_METH_VAR_NAME_899", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_900" to i8*), i8* getelementptr ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_908", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_909" to i8*), i8* getelementptr ([10 x i8]* @"\01L_OBJC_METH_VAR_NAME_910", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_911" to i8*), i8* getelementptr ([38 x i8]* @"\01L_OBJC_METH_VAR_NAME_912", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_913" to i8*), i8* getelementptr ([20 x i8]* @"\01L_OBJC_METH_VAR_NAME_918", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_919" to i8*), i8* getelementptr ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_920", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_921" to i8*), i8* getelementptr ([27 x i8]* @"\01L_OBJC_METH_VAR_NAME_951", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_952" to i8*), i8* getelementptr ([53 x i8]* @"\01L_OBJC_METH_VAR_NAME_953", i32 0, i32 0), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_954" to i8*), i8* getelementptr ([10 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* getelementptr ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_955", i32 0, i32 0), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_956", i32 0, i32 0), i8* getelementptr ([20 x i8]* @"\01L_OBJC_METH_VAR_TYPE_957", i32 0, i32 0), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_958", i32 0, i32 0), i8* getelementptr ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_959", i32 0, i32 0), i8* getelementptr ([10 x i8]* @"\01L_OBJC_METH_VAR_TYPE_960", i32 0, i32 0), i8* getelementptr ([18 x i8]* @"\01L_OBJC_METH_VAR_NAME_961", i32 0, i32 0), i8* getelementptr ([17 x i8]* @"\01L_OBJC_METH_VAR_TYPE_962", i32 0, i32 0), i8* getelementptr ([22 x i8]* @"\01L_OBJC_METH_VAR_NAME_963", i32 0, i32 0), i8* getelementptr ([20 x i8]* @"\01L_OBJC_METH_VAR_TYPE_964", i32 0, i32 0), i8* getelementptr ([51 x i8]* @"\01L_OBJC_METH_VAR_TYPE_965", i32 0, i32 0), i8* getelementptr ([41 x i8]* @"\01L_OBJC_METH_VAR_TYPE_966", i32 0, i32 0), i8* getelementptr ([72 x i8]* @"\01L_OBJC_METH_VAR_TYPE_967", i32 0, i32 0), i8* getelementptr ([32 x i8]* @"\01L_OBJC_METH_VAR_NAME_968", i32 0, i32 0), i8* getelementptr ([75 x i8]* @"\01L_OBJC_METH_VAR_TYPE_969", i32 0, i32 0), i8* getelementptr ([19 x i8]* @"\01L_OBJC_METH_VAR_NAME_970", i32 0, i32 0), i8* getelementptr ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_971", i32 0, i32 0), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_972", i32 0, i32 0), i8* getelementptr ([44 x i8]* @"\01L_OBJC_METH_VAR_TYPE_973", i32 0, i32 0), i8* getelementptr ([30 x i8]* @"\01L_OBJC_METH_VAR_NAME_974", i32 0, i32 0), i8* getelementptr ([56 x i8]* @"\01L_OBJC_METH_VAR_TYPE_975", i32 0, i32 0), i8* getelementptr ([9 x i8]* @"\01L_OBJC_CLASS_NAME_976", i32 0, i32 0), i8* getelementptr ([18 x i8]* @"\01L_OBJC_CLASS_NAME_977", i32 0, i32 0), i8* bitcast (%struct._objc_class* @"\01L_OBJC_METACLASS_DVIconAndTextCell" to i8*), i8* getelementptr ([11 x i8]* @"\01L_OBJC_METH_VAR_TYPE_978", i32 0, i32 0), i8* getelementptr ([27 x i8]* @"\01L_OBJC_METH_VAR_TYPE_979", i32 0, i32 0), i8* bitcast (%0* @"\01L_OBJC_INSTANCE_VARIABLES_DVIconAndTextCell" to i8*), i8* bitcast (%1* @"\01L_OBJC_INSTANCE_METHODS_DVIconAndTextCell" to i8*), i8* bitcast (%struct._objc_class* @"\01L_OBJC_CLASS_DVIconAndTextCell" to i8*), i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_1012", i32 0, i32 0), i8* bitcast (%2* @"\01L_OBJC_SYMBOLS" to i8*), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*)], section "llvm.metadata"		; <[90 x i8*]*> [#uses=0]

define internal %struct.objc_object* @"\01-[DVIconAndTextCell initTextCell:]"(%3* %self, %struct.objc_selector* %_cmd, %4* %string) nounwind {
entry:
	%retval = alloca %struct.objc_object*		; <%struct.objc_object**> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=7]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%string.addr = alloca %4*		; <%4**> [#uses=3]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable345 to { }*))
	store %4* %string, %4** %string.addr
	%2 = bitcast %4** %string.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable347 to { }*))
	call void @llvm.dbg.stoppoint(i32 16, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%self1 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp = load %4** %string.addr		; <%4*> [#uses=1]
	%objc_super = alloca %struct._objc_super		; <%struct._objc_super*> [#uses=3]
	%3 = bitcast %3* %self1 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%4 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 0		; <%struct.objc_object**> [#uses=1]
	store %struct.objc_object* %3, %struct.objc_object** %4
	%tmp2 = load %struct._objc_class** @"\01L_OBJC_CLASS_REFERENCES_"		; <%struct._objc_class*> [#uses=1]
	%5 = bitcast %struct._objc_class* %tmp2 to %struct.objc_class*		; <%struct.objc_class*> [#uses=1]
	%6 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 1		; <%struct.objc_class**> [#uses=1]
	store %struct.objc_class* %5, %struct.objc_class** %6
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_"		; <%struct.objc_selector*> [#uses=1]
	%call = call %struct.objc_object* bitcast (%struct.objc_object* (%struct._objc_super*, %struct.objc_selector*, ...)* @objc_msgSendSuper to %struct.objc_object* (%struct._objc_super*, %struct.objc_selector*, %4*)*)(%struct._objc_super* %objc_super, %struct.objc_selector* %tmp3, %4* %tmp)		; <%struct.objc_object*> [#uses=1]
	%conv = bitcast %struct.objc_object* %call to %3*		; <%3*> [#uses=2]
	store %3* %conv, %3** %self.addr
	%tobool = icmp ne %3* %conv, null		; <i1> [#uses=1]
	br i1 %tobool, label %if.then, label %if.end

if.then:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 17, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp4 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp5 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_349"		; <%struct.objc_selector*> [#uses=1]
	%tmp6 = bitcast %3* %tmp4 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.objc_object*, %struct.objc_selector*, i32)*)(%struct.objc_object* %tmp6, %struct.objc_selector* %tmp5, i32 4)
	call void @llvm.dbg.stoppoint(i32 18, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp7 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp8 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_351"		; <%struct.objc_selector*> [#uses=1]
	%tmp9 = bitcast %3* %tmp7 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.objc_object*, %struct.objc_selector*, i32)*)(%struct.objc_object* %tmp9, %struct.objc_selector* %tmp8, i32 2)
	br label %if.end

if.end:		; preds = %if.then, %entry
	call void @llvm.dbg.stoppoint(i32 20, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp10 = load %3** %self.addr		; <%3*> [#uses=1]
	%conv11 = bitcast %3* %tmp10 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	store %struct.objc_object* %conv11, %struct.objc_object** %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %7, %if.end
	call void @llvm.dbg.stoppoint(i32 21, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%8 = load %struct.objc_object** %retval		; <%struct.objc_object*> [#uses=1]
	ret %struct.objc_object* %8
}

declare void @llvm.dbg.func.start({ }*) nounwind readnone

declare void @llvm.dbg.declare({ }*, { }*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind readnone

declare %struct.objc_object* @objc_msgSendSuper(%struct._objc_super*, %struct.objc_selector*, ...)

declare %struct.objc_object* @objc_msgSend(%struct.objc_object*, %struct.objc_selector*, ...)

declare void @llvm.dbg.region.end({ }*) nounwind readnone

define internal %struct.objc_object* @"\01-[DVIconAndTextCell init]"(%3* %self, %struct.objc_selector* %_cmd) nounwind {
entry:
	%retval = alloca %struct.objc_object*		; <%struct.objc_object**> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram353 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable354 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable355 to { }*))
	call void @llvm.dbg.stoppoint(i32 24, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_"		; <%struct.objc_selector*> [#uses=1]
	%tmp2 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call = call %struct.objc_object* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, %4*)*)(%struct.objc_object* %tmp2, %struct.objc_selector* %tmp1, %4* bitcast (%struct.NSConstantString* @"\01LC356" to %4*))		; <%struct.objc_object*> [#uses=1]
	store %struct.objc_object* %call, %struct.objc_object** %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %2, %entry
	call void @llvm.dbg.stoppoint(i32 25, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram353 to { }*))
	%3 = load %struct.objc_object** %retval		; <%struct.objc_object*> [#uses=1]
	ret %struct.objc_object* %3
}

define internal %struct.objc_object* @"\01-[DVIconAndTextCell copyWithZone:]"(%3* %self, %struct.objc_selector* %_cmd, %struct._NSZone* %zone) nounwind {
entry:
	%retval = alloca %struct.objc_object*		; <%struct.objc_object**> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=5]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%zone.addr = alloca %struct._NSZone*		; <%struct._NSZone**> [#uses=3]
	%copy = alloca %3*, align 4		; <%3**> [#uses=5]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram358 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable359 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable360 to { }*))
	store %struct._NSZone* %zone, %struct._NSZone** %zone.addr
	%2 = bitcast %struct._NSZone** %zone.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable368 to { }*))
	call void @llvm.dbg.stoppoint(i32 28, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = bitcast %3** %copy to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable370 to { }*))
	%self1 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp = load %struct._NSZone** %zone.addr		; <%struct._NSZone*> [#uses=1]
	%objc_super = alloca %struct._objc_super		; <%struct._objc_super*> [#uses=3]
	%4 = bitcast %3* %self1 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%5 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 0		; <%struct.objc_object**> [#uses=1]
	store %struct.objc_object* %4, %struct.objc_object** %5
	%tmp2 = load %struct._objc_class** @"\01L_OBJC_CLASS_REFERENCES_"		; <%struct._objc_class*> [#uses=1]
	%6 = bitcast %struct._objc_class* %tmp2 to %struct.objc_class*		; <%struct.objc_class*> [#uses=1]
	%7 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 1		; <%struct.objc_class**> [#uses=1]
	store %struct.objc_class* %6, %struct.objc_class** %7
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_372"		; <%struct.objc_selector*> [#uses=1]
	%call = call %struct.objc_object* bitcast (%struct.objc_object* (%struct._objc_super*, %struct.objc_selector*, ...)* @objc_msgSendSuper to %struct.objc_object* (%struct._objc_super*, %struct.objc_selector*, %struct._NSZone*)*)(%struct._objc_super* %objc_super, %struct.objc_selector* %tmp3, %struct._NSZone* %tmp)		; <%struct.objc_object*> [#uses=1]
	%conv = bitcast %struct.objc_object* %call to %3*		; <%3*> [#uses=1]
	store %3* %conv, %3** %copy
	call void @llvm.dbg.stoppoint(i32 29, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp4 = load %3** %copy		; <%3*> [#uses=1]
	%8 = bitcast %3* %tmp4 to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %8, i32 48		; <i8*> [#uses=1]
	%9 = bitcast i8* %add.ptr to %5**		; <%5**> [#uses=1]
	%tmp5 = load %3** %self.addr		; <%3*> [#uses=1]
	%10 = bitcast %3* %tmp5 to i8*		; <i8*> [#uses=1]
	%add.ptr6 = getelementptr i8* %10, i32 48		; <i8*> [#uses=1]
	%11 = bitcast i8* %add.ptr6 to %5**		; <%5**> [#uses=1]
	%tmp7 = load %5** %11		; <%5*> [#uses=1]
	%tmp8 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_374"		; <%struct.objc_selector*> [#uses=1]
	%tmp9 = bitcast %5* %tmp7 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call10 = call %struct.objc_object* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %struct.objc_object* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp9, %struct.objc_selector* %tmp8)		; <%struct.objc_object*> [#uses=1]
	%conv11 = bitcast %struct.objc_object* %call10 to %5*		; <%5*> [#uses=1]
	store %5* %conv11, %5** %9
	call void @llvm.dbg.stoppoint(i32 30, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp12 = load %3** %copy		; <%3*> [#uses=1]
	%12 = bitcast %3* %tmp12 to i8*		; <i8*> [#uses=1]
	%add.ptr13 = getelementptr i8* %12, i32 52		; <i8*> [#uses=1]
	%13 = bitcast i8* %add.ptr13 to %struct.CGPoint*		; <%struct.CGPoint*> [#uses=1]
	%tmp14 = load %3** %self.addr		; <%3*> [#uses=1]
	%14 = bitcast %3* %tmp14 to i8*		; <i8*> [#uses=1]
	%add.ptr15 = getelementptr i8* %14, i32 52		; <i8*> [#uses=1]
	%15 = bitcast i8* %add.ptr15 to %struct.CGPoint*		; <%struct.CGPoint*> [#uses=1]
	%tmp16 = bitcast %struct.CGPoint* %13 to i8*		; <i8*> [#uses=1]
	%tmp17 = bitcast %struct.CGPoint* %15 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp16, i8* %tmp17, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 31, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp18 = load %3** %copy		; <%3*> [#uses=1]
	%conv19 = bitcast %3* %tmp18 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	store %struct.objc_object* %conv19, %struct.objc_object** %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %16, %entry
	call void @llvm.dbg.stoppoint(i32 32, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram358 to { }*))
	%17 = load %struct.objc_object** %retval		; <%struct.objc_object*> [#uses=1]
	ret %struct.objc_object* %17
}

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind

define internal void @"\01-[DVIconAndTextCell dealloc]"(%3* %self, %struct.objc_selector* %_cmd) nounwind {
entry:
	%self.addr = alloca %3*		; <%3**> [#uses=4]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram376 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable377 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable378 to { }*))
	call void @llvm.dbg.stoppoint(i32 35, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%2 = bitcast %3* %tmp to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %2, i32 48		; <i8*> [#uses=1]
	%3 = bitcast i8* %add.ptr to %5**		; <%5**> [#uses=1]
	%tmp1 = load %5** %3		; <%5*> [#uses=1]
	%tmp2 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_380"		; <%struct.objc_selector*> [#uses=1]
	%tmp3 = bitcast %5* %tmp1 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp3, %struct.objc_selector* %tmp2)
	call void @llvm.dbg.stoppoint(i32 36, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%self4 = load %3** %self.addr		; <%3*> [#uses=1]
	%objc_super = alloca %struct._objc_super		; <%struct._objc_super*> [#uses=3]
	%4 = bitcast %3* %self4 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%5 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 0		; <%struct.objc_object**> [#uses=1]
	store %struct.objc_object* %4, %struct.objc_object** %5
	%tmp5 = load %struct._objc_class** @"\01L_OBJC_CLASS_REFERENCES_"		; <%struct._objc_class*> [#uses=1]
	%6 = bitcast %struct._objc_class* %tmp5 to %struct.objc_class*		; <%struct.objc_class*> [#uses=1]
	%7 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 1		; <%struct.objc_class**> [#uses=1]
	store %struct.objc_class* %6, %struct.objc_class** %7
	%tmp6 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_382"		; <%struct.objc_selector*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct._objc_super*, %struct.objc_selector*, ...)* @objc_msgSendSuper to void (%struct._objc_super*, %struct.objc_selector*)*)(%struct._objc_super* %objc_super, %struct.objc_selector* %tmp6)
	call void @llvm.dbg.stoppoint(i32 37, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram376 to { }*))
	ret void
}

define internal void @"\01-[DVIconAndTextCell setIcon:]"(%3* %self, %struct.objc_selector* %_cmd, %5* %newIcon) nounwind {
entry:
	%self.addr = alloca %3*		; <%3**> [#uses=5]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%newIcon.addr = alloca %5*		; <%5**> [#uses=4]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram384 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable385 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable386 to { }*))
	store %5* %newIcon, %5** %newIcon.addr
	%2 = bitcast %5** %newIcon.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable388 to { }*))
	call void @llvm.dbg.stoppoint(i32 40, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%3 = bitcast %3* %tmp to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %3, i32 48		; <i8*> [#uses=1]
	%4 = bitcast i8* %add.ptr to %5**		; <%5**> [#uses=1]
	%tmp1 = load %5** %4		; <%5*> [#uses=1]
	%tmp2 = load %5** %newIcon.addr		; <%5*> [#uses=1]
	%cmp = icmp ne %5* %tmp1, %tmp2		; <i1> [#uses=1]
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 41, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp3 = load %3** %self.addr		; <%3*> [#uses=1]
	%5 = bitcast %3* %tmp3 to i8*		; <i8*> [#uses=1]
	%add.ptr4 = getelementptr i8* %5, i32 48		; <i8*> [#uses=1]
	%6 = bitcast i8* %add.ptr4 to %5**		; <%5**> [#uses=1]
	%tmp5 = load %5** %6		; <%5*> [#uses=1]
	%tmp6 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_380"		; <%struct.objc_selector*> [#uses=1]
	%tmp7 = bitcast %5* %tmp5 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp7, %struct.objc_selector* %tmp6)
	call void @llvm.dbg.stoppoint(i32 42, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp8 = load %3** %self.addr		; <%3*> [#uses=1]
	%7 = bitcast %3* %tmp8 to i8*		; <i8*> [#uses=1]
	%add.ptr9 = getelementptr i8* %7, i32 48		; <i8*> [#uses=1]
	%8 = bitcast i8* %add.ptr9 to %5**		; <%5**> [#uses=1]
	%tmp10 = load %5** %newIcon.addr		; <%5*> [#uses=1]
	%tmp11 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_374"		; <%struct.objc_selector*> [#uses=1]
	%tmp12 = bitcast %5* %tmp10 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call = call %struct.objc_object* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %struct.objc_object* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp12, %struct.objc_selector* %tmp11)		; <%struct.objc_object*> [#uses=1]
	%conv = bitcast %struct.objc_object* %call to %5*		; <%5*> [#uses=1]
	store %5* %conv, %5** %8
	br label %if.end

if.end:		; preds = %if.then, %entry
	call void @llvm.dbg.stoppoint(i32 44, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram384 to { }*))
	ret void
}

define internal %5* @"\01-[DVIconAndTextCell icon]"(%3* %self, %struct.objc_selector* %_cmd) nounwind {
entry:
	%retval = alloca %5*		; <%5**> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram390 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable391 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable392 to { }*))
	call void @llvm.dbg.stoppoint(i32 47, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%2 = bitcast %3* %tmp to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %2, i32 48		; <i8*> [#uses=1]
	%3 = bitcast i8* %add.ptr to %5**		; <%5**> [#uses=1]
	%tmp1 = load %5** %3		; <%5*> [#uses=1]
	store %5* %tmp1, %5** %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %4, %entry
	call void @llvm.dbg.stoppoint(i32 48, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram390 to { }*))
	%5 = load %5** %retval		; <%5*> [#uses=1]
	ret %5* %5
}

define internal void @"\01-[DVIconAndTextCell preferredIconSize]"(%struct.CGPoint* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd) nounwind {
entry:
	%retval = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram394 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable395 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable396 to { }*))
	call void @llvm.dbg.stoppoint(i32 51, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%2 = bitcast %3* %tmp to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %2, i32 52		; <i8*> [#uses=1]
	%3 = bitcast i8* %add.ptr to %struct.CGPoint*		; <%struct.CGPoint*> [#uses=1]
	%tmp1 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGPoint* %3 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 16, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %4, %entry
	call void @llvm.dbg.stoppoint(i32 52, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram394 to { }*))
	%tmp3 = bitcast %struct.CGPoint* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp4 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp3, i8* %tmp4, i32 16, i32 4)
	ret void
}

define internal void @"\01-[DVIconAndTextCell setPreferredIconSize:]"(%3* %self, %struct.objc_selector* %_cmd, double %size.0, double %size.1) nounwind {
entry:
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%size = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=4]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram398 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable399 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable400 to { }*))
	%tmp = getelementptr %struct.CGPoint* %size, i32 0, i32 0		; <double*> [#uses=1]
	store double %size.0, double* %tmp
	%tmp1 = getelementptr %struct.CGPoint* %size, i32 0, i32 1		; <double*> [#uses=1]
	store double %size.1, double* %tmp1
	%2 = bitcast %struct.CGPoint* %size to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable402 to { }*))
	call void @llvm.dbg.stoppoint(i32 55, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp2 = load %3** %self.addr		; <%3*> [#uses=1]
	%3 = bitcast %3* %tmp2 to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %3, i32 52		; <i8*> [#uses=1]
	%4 = bitcast i8* %add.ptr to %struct.CGPoint*		; <%struct.CGPoint*> [#uses=1]
	%tmp3 = bitcast %struct.CGPoint* %4 to i8*		; <i8*> [#uses=1]
	%tmp4 = bitcast %struct.CGPoint* %size to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp3, i8* %tmp4, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 56, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram398 to { }*))
	ret void
}

define internal void @"\01-[DVIconAndTextCell iconSizeForBounds:]"(%struct.CGPoint* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=6]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%iconSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=5]
	%agg.tmp = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=3]
	%tmp22 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram404 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable405 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable406 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable427 to { }*))
	call void @llvm.dbg.stoppoint(i32 59, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = bitcast %struct.CGPoint* %iconSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable429 to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%4 = bitcast %3* %tmp to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %4, i32 52		; <i8*> [#uses=1]
	%5 = bitcast i8* %add.ptr to %struct.CGPoint*		; <%struct.CGPoint*> [#uses=1]
	%tmp1 = bitcast %struct.CGPoint* %iconSize to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGPoint* %5 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 60, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp3 = load %3** %self.addr		; <%3*> [#uses=1]
	%6 = bitcast %3* %tmp3 to i8*		; <i8*> [#uses=1]
	%add.ptr4 = getelementptr i8* %6, i32 52		; <i8*> [#uses=1]
	%7 = bitcast i8* %add.ptr4 to %struct.CGPoint*		; <%struct.CGPoint*> [#uses=1]
	%tmp5 = bitcast %struct.CGPoint* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp6 = bitcast %struct.CGPoint* %7 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp5, i8* %tmp6, i32 16, i32 4)
	%tmp7 = getelementptr %struct.CGPoint* %agg.tmp, i32 0, i32 0		; <double*> [#uses=1]
	%tmp8 = load double* %tmp7		; <double> [#uses=1]
	%tmp9 = getelementptr %struct.CGPoint* %agg.tmp, i32 0, i32 1		; <double*> [#uses=1]
	%tmp10 = load double* %tmp9		; <double> [#uses=1]
	%call = call signext i8 @DVIsEmptySize(double %tmp8, double %tmp10)		; <i8> [#uses=1]
	%tobool = icmp ne i8 %call, 0		; <i1> [#uses=1]
	br i1 %tobool, label %if.then, label %if.end

if.then:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 61, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp11 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp12 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_431"		; <%struct.objc_selector*> [#uses=1]
	%tmp13 = bitcast %3* %tmp11 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call14 = call %5* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %5* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp13, %struct.objc_selector* %tmp12)		; <%5*> [#uses=1]
	%tobool15 = icmp ne %5* %call14, null		; <i1> [#uses=1]
	br i1 %tobool15, label %cond.true, label %cond.false

cond.true:		; preds = %if.then
	%tmp16 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp17 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_431"		; <%struct.objc_selector*> [#uses=1]
	%tmp18 = bitcast %3* %tmp16 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call19 = call %5* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %5* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp18, %struct.objc_selector* %tmp17)		; <%5*> [#uses=1]
	%tmp20 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_433"		; <%struct.objc_selector*> [#uses=1]
	%tmp21 = bitcast %5* %call19 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*)*)(%struct.CGPoint* noalias sret %tmp22, %struct.objc_object* %tmp21, %struct.objc_selector* %tmp20)
	%tmp23 = bitcast %struct.CGPoint* %iconSize to i8*		; <i8*> [#uses=1]
	%tmp24 = bitcast %struct.CGPoint* %tmp22 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp23, i8* %tmp24, i32 16, i32 4)
	br label %cond.end

cond.false:		; preds = %if.then
	%tmp25 = bitcast %struct.CGPoint* %iconSize to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp25, i8* bitcast (%struct.CGPoint* @NSZeroSize to i8*), i32 16, i32 4)
	br label %cond.end

cond.end:		; preds = %cond.false, %cond.true
	br label %if.end

if.end:		; preds = %cond.end, %entry
	call void @llvm.dbg.stoppoint(i32 63, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp26 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	%tmp27 = bitcast %struct.CGPoint* %iconSize to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp26, i8* %tmp27, i32 16, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %8, %if.end
	call void @llvm.dbg.stoppoint(i32 64, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram404 to { }*))
	%tmp28 = bitcast %struct.CGPoint* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp29 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp28, i8* %tmp29, i32 16, i32 4)
	ret void
}

declare signext i8 @DVIsEmptySize(double, double)

declare void @objc_msgSend_stret(%struct.objc_object*, %struct.objc_selector*, ...)

define internal double @"\01-[DVIconAndTextCell iconInsetForBounds:]"(%3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval = alloca double		; <double*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp5 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram435 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable436 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable437 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable438 to { }*))
	call void @llvm.dbg.stoppoint(i32 67, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_440"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGPoint* noalias sret %tmp5, %struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)
	%tmp6 = getelementptr %struct.CGPoint* %tmp5, i32 0, i32 0		; <double*> [#uses=1]
	%tmp7 = load double* %tmp6		; <double> [#uses=1]
	%div = fdiv double %tmp7, 3.000000e+00		; <double> [#uses=1]
	%call = call double @floor(double %div)		; <double> [#uses=1]
	store double %call, double* %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %3, %entry
	call void @llvm.dbg.stoppoint(i32 68, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram435 to { }*))
	%4 = load double* %retval		; <double> [#uses=1]
	ret double %4
}

declare double @floor(double)

define internal double @"\01-[DVIconAndTextCell textInsetForBounds:]"(%3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval = alloca double		; <double*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp5 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram442 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable443 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable444 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable445 to { }*))
	call void @llvm.dbg.stoppoint(i32 71, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_440"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGPoint* noalias sret %tmp5, %struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)
	%tmp6 = getelementptr %struct.CGPoint* %tmp5, i32 0, i32 0		; <double*> [#uses=1]
	%tmp7 = load double* %tmp6		; <double> [#uses=1]
	%div = fdiv double %tmp7, 3.000000e+00		; <double> [#uses=1]
	%call = call double @floor(double %div)		; <double> [#uses=1]
	store double %call, double* %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %3, %entry
	call void @llvm.dbg.stoppoint(i32 72, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram442 to { }*))
	%4 = load double* %retval		; <double> [#uses=1]
	ret double %4
}

define internal %6* @"\01-[DVIconAndTextCell attributedTitle]"(%3* %self, %struct.objc_selector* %_cmd) nounwind {
entry:
	%retval = alloca %6*		; <%6**> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%title = alloca %7*, align 4		; <%7**> [#uses=3]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram455 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable456 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable457 to { }*))
	call void @llvm.dbg.stoppoint(i32 75, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = bitcast %7** %title to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable465 to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_467"		; <%struct.objc_selector*> [#uses=1]
	%tmp2 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call = call %6* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %6* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp2, %struct.objc_selector* %tmp1)		; <%6*> [#uses=1]
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_469"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %6* %call to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call5 = call %struct.objc_object* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %struct.objc_object* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp4, %struct.objc_selector* %tmp3)		; <%struct.objc_object*> [#uses=1]
	%tmp6 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_471"		; <%struct.objc_selector*> [#uses=1]
	%call7 = call %struct.objc_object* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %struct.objc_object* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %call5, %struct.objc_selector* %tmp6)		; <%struct.objc_object*> [#uses=1]
	%conv = bitcast %struct.objc_object* %call7 to %7*		; <%7*> [#uses=1]
	store %7* %conv, %7** %title
	call void @llvm.dbg.stoppoint(i32 76, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp8 = load %7** %title		; <%7*> [#uses=1]
	%conv9 = bitcast %7* %tmp8 to %6*		; <%6*> [#uses=1]
	store %6* %conv9, %6** %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %3, %entry
	call void @llvm.dbg.stoppoint(i32 77, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram455 to { }*))
	%4 = load %6** %retval		; <%6*> [#uses=1]
	ret %6* %4
}

define internal void @"\01-[DVIconAndTextCell titleSize]"(%struct.CGPoint* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd) nounwind {
entry:
	%retval = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%tmp5 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram473 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable474 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable475 to { }*))
	call void @llvm.dbg.stoppoint(i32 80, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_477"		; <%struct.objc_selector*> [#uses=1]
	%tmp2 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call = call %6* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %6* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp2, %struct.objc_selector* %tmp1)		; <%6*> [#uses=1]
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_433"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %6* %call to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*)*)(%struct.CGPoint* noalias sret %tmp5, %struct.objc_object* %tmp4, %struct.objc_selector* %tmp3)
	%tmp6 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	%tmp7 = bitcast %struct.CGPoint* %tmp5 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp6, i8* %tmp7, i32 16, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %2, %entry
	call void @llvm.dbg.stoppoint(i32 81, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram473 to { }*))
	%tmp8 = bitcast %struct.CGPoint* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp9 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp8, i8* %tmp9, i32 16, i32 4)
	ret void
}

define internal void @"\01-[DVIconAndTextCell titleAndIconRectForBounds:]"(%struct.CGRect* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval.i175 = alloca double		; <double*> [#uses=2]
	%aRect172 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i168 = alloca double		; <double*> [#uses=2]
	%aRect165 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i154 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%w.addr.i155 = alloca double		; <double*> [#uses=3]
	%h.addr.i156 = alloca double		; <double*> [#uses=3]
	%s.i = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=4]
	%retval.i150 = alloca double		; <double*> [#uses=2]
	%aRect147 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i143 = alloca double		; <double*> [#uses=2]
	%aRect140 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i136 = alloca double		; <double*> [#uses=2]
	%aRect133 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i129 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%x.addr.i = alloca double		; <double*> [#uses=3]
	%y.addr.i = alloca double		; <double*> [#uses=3]
	%w.addr.i = alloca double		; <double*> [#uses=3]
	%h.addr.i = alloca double		; <double*> [#uses=3]
	%r.i = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=6]
	%retval.i = alloca double		; <double*> [#uses=2]
	%aRect = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=6]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%iconInset = alloca double, align 8		; <double*> [#uses=3]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%textInset = alloca double, align 8		; <double*> [#uses=5]
	%agg.tmp7 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%iconSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=8]
	%agg.tmp15 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp20 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%textSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=8]
	%tmp27 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%maxLegalWidth = alloca double, align 8		; <double*> [#uses=4]
	%agg.tmp31 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%titleAndIconSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=5]
	%agg.tmp65 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp85 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp91 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%x = alloca double, align 8		; <double*> [#uses=4]
	%y = alloca double, align 8		; <double*> [#uses=3]
	%agg.tmp96 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp100 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp110 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp122 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable480 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable481 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable482 to { }*))
	call void @llvm.dbg.stoppoint(i32 84, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = bitcast double* %iconInset to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable484 to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_486"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call = call double bitcast (double (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_fpret to double (%struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)		; <double> [#uses=1]
	store double %call, double* %iconInset
	call void @llvm.dbg.stoppoint(i32 85, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = bitcast double* %textInset to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable488 to { }*))
	%tmp6 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp8 = bitcast %struct.CGRect* %agg.tmp7 to i8*		; <i8*> [#uses=1]
	%tmp9 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp8, i8* %tmp9, i32 32, i32 4)
	%tmp10 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_490"		; <%struct.objc_selector*> [#uses=1]
	%tmp11 = bitcast %3* %tmp6 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call12 = call double bitcast (double (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_fpret to double (%struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.objc_object* %tmp11, %struct.objc_selector* %tmp10, %struct.CGRect* byval %agg.tmp7)		; <double> [#uses=1]
	store double %call12, double* %textInset
	call void @llvm.dbg.stoppoint(i32 86, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = bitcast %struct.CGPoint* %iconSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable491 to { }*))
	%tmp14 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp16 = bitcast %struct.CGRect* %agg.tmp15 to i8*		; <i8*> [#uses=1]
	%tmp17 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp16, i8* %tmp17, i32 32, i32 4)
	%tmp18 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_440"		; <%struct.objc_selector*> [#uses=1]
	%tmp19 = bitcast %3* %tmp14 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGPoint* noalias sret %tmp20, %struct.objc_object* %tmp19, %struct.objc_selector* %tmp18, %struct.CGRect* byval %agg.tmp15)
	%tmp21 = bitcast %struct.CGPoint* %iconSize to i8*		; <i8*> [#uses=1]
	%tmp22 = bitcast %struct.CGPoint* %tmp20 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp21, i8* %tmp22, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 87, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = bitcast %struct.CGPoint* %textSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %6, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable493 to { }*))
	%tmp24 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp25 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_495"		; <%struct.objc_selector*> [#uses=1]
	%tmp26 = bitcast %3* %tmp24 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*)*)(%struct.CGPoint* noalias sret %tmp27, %struct.objc_object* %tmp26, %struct.objc_selector* %tmp25)
	%tmp28 = bitcast %struct.CGPoint* %textSize to i8*		; <i8*> [#uses=1]
	%tmp29 = bitcast %struct.CGPoint* %tmp27 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp28, i8* %tmp29, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 88, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%7 = bitcast double* %maxLegalWidth to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %7, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable497 to { }*))
	%tmp32 = bitcast %struct.CGRect* %agg.tmp31 to i8*		; <i8*> [#uses=1]
	%tmp33 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp32, i8* %tmp33, i32 32, i32 4)
	%tmp127 = bitcast %struct.CGRect* %aRect to i8*		; <i8*> [#uses=1]
	%tmp128 = bitcast %struct.CGRect* %agg.tmp31 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp127, i8* %tmp128, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1010 to { }*))
	%8 = bitcast %struct.CGRect* %aRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %8, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1011 to { }*))
	call void @llvm.dbg.stoppoint(i32 134, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i = getelementptr %struct.CGRect* %aRect, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i = getelementptr %struct.CGPoint* %tmp.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i = load double* %tmp1.i		; <double> [#uses=1]
	store double %tmp2.i, double* %retval.i
	call void @llvm.dbg.stoppoint(i32 135, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%9 = load double* %retval.i		; <double> [#uses=1]
	%tmp35 = load double* %textInset		; <double> [#uses=1]
	%sub = sub double %9, %tmp35		; <double> [#uses=1]
	store double %sub, double* %maxLegalWidth
	call void @llvm.dbg.stoppoint(i32 89, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1010 to { }*))
	%10 = bitcast %struct.CGPoint* %titleAndIconSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %10, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable499 to { }*))
	%tmp37 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp38 = load double* %tmp37		; <double> [#uses=1]
	%tmp39 = load double* %textInset		; <double> [#uses=1]
	%add = add double %tmp38, %tmp39		; <double> [#uses=1]
	%tmp40 = getelementptr %struct.CGPoint* %textSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp41 = load double* %tmp40		; <double> [#uses=1]
	%add42 = add double %add, %tmp41		; <double> [#uses=1]
	%tmp43 = load double* %maxLegalWidth		; <double> [#uses=1]
	%cmp = fcmp olt double %add42, %tmp43		; <i1> [#uses=1]
	br i1 %cmp, label %cond.true, label %cond.false

cond.true:		; preds = %entry
	%tmp44 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp45 = load double* %tmp44		; <double> [#uses=1]
	%tmp46 = load double* %textInset		; <double> [#uses=1]
	%add47 = add double %tmp45, %tmp46		; <double> [#uses=1]
	%tmp48 = getelementptr %struct.CGPoint* %textSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp49 = load double* %tmp48		; <double> [#uses=1]
	%add50 = add double %add47, %tmp49		; <double> [#uses=1]
	br label %cond.end

cond.false:		; preds = %entry
	%tmp51 = load double* %maxLegalWidth		; <double> [#uses=1]
	br label %cond.end

cond.end:		; preds = %cond.false, %cond.true
	%cond = phi double [ %add50, %cond.true ], [ %tmp51, %cond.false ]		; <double> [#uses=1]
	%tmp52 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp53 = load double* %tmp52		; <double> [#uses=1]
	%tmp54 = getelementptr %struct.CGPoint* %textSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp55 = load double* %tmp54		; <double> [#uses=1]
	%cmp56 = fcmp ogt double %tmp53, %tmp55		; <i1> [#uses=1]
	br i1 %cmp56, label %cond.true57, label %cond.false60

cond.true57:		; preds = %cond.end
	%tmp58 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp59 = load double* %tmp58		; <double> [#uses=1]
	br label %cond.end63

cond.false60:		; preds = %cond.end
	%tmp61 = getelementptr %struct.CGPoint* %textSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp62 = load double* %tmp61		; <double> [#uses=1]
	br label %cond.end63

cond.end63:		; preds = %cond.false60, %cond.true57
	%cond64 = phi double [ %tmp59, %cond.true57 ], [ %tmp62, %cond.false60 ]		; <double> [#uses=1]
	%tmp66 = bitcast %struct.CGRect* %agg.tmp65 to i8*		; <i8*> [#uses=1]
	%tmp67 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp66, i8* %tmp67, i32 32, i32 4)
	%tmp173 = bitcast %struct.CGRect* %aRect172 to i8*		; <i8*> [#uses=1]
	%tmp174 = bitcast %struct.CGRect* %agg.tmp65 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp173, i8* %tmp174, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%11 = bitcast %struct.CGRect* %aRect172 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %11, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i176 = getelementptr %struct.CGRect* %aRect172, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i177 = getelementptr %struct.CGPoint* %tmp.i176, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i178 = load double* %tmp1.i177		; <double> [#uses=1]
	store double %tmp2.i178, double* %retval.i175
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%12 = load double* %retval.i175		; <double> [#uses=1]
	%cmp69 = fcmp olt double %cond64, %12		; <i1> [#uses=1]
	br i1 %cmp69, label %cond.true70, label %cond.false84

cond.true70:		; preds = %cond.end63
	%tmp71 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp72 = load double* %tmp71		; <double> [#uses=1]
	%tmp73 = getelementptr %struct.CGPoint* %textSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp74 = load double* %tmp73		; <double> [#uses=1]
	%cmp75 = fcmp ogt double %tmp72, %tmp74		; <i1> [#uses=1]
	br i1 %cmp75, label %cond.true76, label %cond.false79

cond.true76:		; preds = %cond.true70
	%tmp77 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp78 = load double* %tmp77		; <double> [#uses=1]
	br label %cond.end82

cond.false79:		; preds = %cond.true70
	%tmp80 = getelementptr %struct.CGPoint* %textSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp81 = load double* %tmp80		; <double> [#uses=1]
	br label %cond.end82

cond.end82:		; preds = %cond.false79, %cond.true76
	%cond83 = phi double [ %tmp78, %cond.true76 ], [ %tmp81, %cond.false79 ]		; <double> [#uses=1]
	br label %cond.end89

cond.false84:		; preds = %cond.end63
	%tmp86 = bitcast %struct.CGRect* %agg.tmp85 to i8*		; <i8*> [#uses=1]
	%tmp87 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp86, i8* %tmp87, i32 32, i32 4)
	%tmp166 = bitcast %struct.CGRect* %aRect165 to i8*		; <i8*> [#uses=1]
	%tmp167 = bitcast %struct.CGRect* %agg.tmp85 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp166, i8* %tmp167, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%13 = bitcast %struct.CGRect* %aRect165 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %13, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i169 = getelementptr %struct.CGRect* %aRect165, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i170 = getelementptr %struct.CGPoint* %tmp.i169, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i171 = load double* %tmp1.i170		; <double> [#uses=1]
	store double %tmp2.i171, double* %retval.i168
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%14 = load double* %retval.i168		; <double> [#uses=1]
	br label %cond.end89

cond.end89:		; preds = %cond.false84, %cond.end82
	%cond90 = phi double [ %cond83, %cond.end82 ], [ %14, %cond.false84 ]		; <double> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1004 to { }*)) nounwind
	store double %cond, double* %w.addr.i155
	%15 = bitcast double* %w.addr.i155 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %15, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1005 to { }*)) nounwind
	store double %cond90, double* %h.addr.i156
	%16 = bitcast double* %h.addr.i156 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %16, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1006 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 94, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%17 = bitcast %struct.CGPoint* %s.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %17, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1008 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 95, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp.i157 = getelementptr %struct.CGPoint* %s.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp1.i158 = load double* %w.addr.i155		; <double> [#uses=1]
	store double %tmp1.i158, double* %tmp.i157
	call void @llvm.dbg.stoppoint(i32 96, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp2.i159 = getelementptr %struct.CGPoint* %s.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp3.i160 = load double* %h.addr.i156		; <double> [#uses=1]
	store double %tmp3.i160, double* %tmp2.i159
	call void @llvm.dbg.stoppoint(i32 97, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp4.i161 = bitcast %struct.CGPoint* %retval.i154 to i8*		; <i8*> [#uses=1]
	%tmp5.i162 = bitcast %struct.CGPoint* %s.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp4.i161, i8* %tmp5.i162, i32 16, i32 4) nounwind
	call void @llvm.dbg.stoppoint(i32 98, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp6.i163 = bitcast %struct.CGPoint* %tmp91 to i8*		; <i8*> [#uses=1]
	%tmp7.i164 = bitcast %struct.CGPoint* %retval.i154 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp6.i163, i8* %tmp7.i164, i32 16, i32 4) nounwind
	%tmp92 = bitcast %struct.CGPoint* %titleAndIconSize to i8*		; <i8*> [#uses=1]
	%tmp93 = bitcast %struct.CGPoint* %tmp91 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp92, i8* %tmp93, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 90, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1004 to { }*))
	%18 = bitcast double* %x to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %18, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable500 to { }*))
	store double 0.000000e+00, double* %x
	%19 = bitcast double* %y to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %19, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable501 to { }*))
	%tmp97 = bitcast %struct.CGRect* %agg.tmp96 to i8*		; <i8*> [#uses=1]
	%tmp98 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp97, i8* %tmp98, i32 32, i32 4)
	%tmp148 = bitcast %struct.CGRect* %aRect147 to i8*		; <i8*> [#uses=1]
	%tmp149 = bitcast %struct.CGRect* %agg.tmp96 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp148, i8* %tmp149, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram998 to { }*))
	%20 = bitcast %struct.CGRect* %aRect147 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %20, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable999 to { }*))
	call void @llvm.dbg.stoppoint(i32 130, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i151 = getelementptr %struct.CGRect* %aRect147, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i152 = getelementptr %struct.CGPoint* %tmp.i151, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i153 = load double* %tmp1.i152		; <double> [#uses=1]
	store double %tmp2.i153, double* %retval.i150
	call void @llvm.dbg.stoppoint(i32 131, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%21 = load double* %retval.i150		; <double> [#uses=1]
	%tmp101 = bitcast %struct.CGRect* %agg.tmp100 to i8*		; <i8*> [#uses=1]
	%tmp102 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp101, i8* %tmp102, i32 32, i32 4)
	%tmp141 = bitcast %struct.CGRect* %aRect140 to i8*		; <i8*> [#uses=1]
	%tmp142 = bitcast %struct.CGRect* %agg.tmp100 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp141, i8* %tmp142, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%22 = bitcast %struct.CGRect* %aRect140 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %22, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram998 to { }*))
	%tmp.i144 = getelementptr %struct.CGRect* %aRect140, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i145 = getelementptr %struct.CGPoint* %tmp.i144, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i146 = load double* %tmp1.i145		; <double> [#uses=1]
	store double %tmp2.i146, double* %retval.i143
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%23 = load double* %retval.i143		; <double> [#uses=1]
	%div = fdiv double %23, 2.000000e+00		; <double> [#uses=1]
	%tmp104 = getelementptr %struct.CGPoint* %titleAndIconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp105 = load double* %tmp104		; <double> [#uses=1]
	%div106 = fdiv double %tmp105, 2.000000e+00		; <double> [#uses=1]
	%sub107 = sub double %div, %div106		; <double> [#uses=1]
	%call108 = call double @floor(double %sub107)		; <double> [#uses=1]
	%add109 = add double %21, %call108		; <double> [#uses=1]
	store double %add109, double* %y
	call void @llvm.dbg.stoppoint(i32 91, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%tmp111 = bitcast %struct.CGRect* %agg.tmp110 to i8*		; <i8*> [#uses=1]
	%tmp112 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp111, i8* %tmp112, i32 32, i32 4)
	%tmp134 = bitcast %struct.CGRect* %aRect133 to i8*		; <i8*> [#uses=1]
	%tmp135 = bitcast %struct.CGRect* %agg.tmp110 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp134, i8* %tmp135, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram995 to { }*))
	%24 = bitcast %struct.CGRect* %aRect133 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %24, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable996 to { }*))
	call void @llvm.dbg.stoppoint(i32 126, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i137 = getelementptr %struct.CGRect* %aRect133, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i138 = getelementptr %struct.CGPoint* %tmp.i137, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i139 = load double* %tmp1.i138		; <double> [#uses=1]
	store double %tmp2.i139, double* %retval.i136
	call void @llvm.dbg.stoppoint(i32 127, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%25 = load double* %retval.i136		; <double> [#uses=1]
	%tmp114 = load double* %iconInset		; <double> [#uses=1]
	%add115 = add double %25, %tmp114		; <double> [#uses=1]
	store double %add115, double* %x
	call void @llvm.dbg.stoppoint(i32 92, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram995 to { }*))
	%tmp116 = load double* %x		; <double> [#uses=1]
	%tmp117 = load double* %y		; <double> [#uses=1]
	%tmp118 = getelementptr %struct.CGPoint* %titleAndIconSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp119 = load double* %tmp118		; <double> [#uses=1]
	%tmp120 = getelementptr %struct.CGPoint* %titleAndIconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp121 = load double* %tmp120		; <double> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*)) nounwind
	store double %tmp116, double* %x.addr.i
	%26 = bitcast double* %x.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %26, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable986 to { }*)) nounwind
	store double %tmp117, double* %y.addr.i
	%27 = bitcast double* %y.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %27, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable987 to { }*)) nounwind
	store double %tmp119, double* %w.addr.i
	%28 = bitcast double* %w.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %28, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable989 to { }*)) nounwind
	store double %tmp121, double* %h.addr.i
	%29 = bitcast double* %h.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %29, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable991 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 101, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%30 = bitcast %struct.CGRect* %r.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %30, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable993 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 102, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp.i130 = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i131 = getelementptr %struct.CGPoint* %tmp.i130, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i132 = load double* %x.addr.i		; <double> [#uses=1]
	store double %tmp2.i132, double* %tmp1.i131
	call void @llvm.dbg.stoppoint(i32 103, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp3.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp4.i = getelementptr %struct.CGPoint* %tmp3.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp5.i = load double* %y.addr.i		; <double> [#uses=1]
	store double %tmp5.i, double* %tmp4.i
	call void @llvm.dbg.stoppoint(i32 104, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp6.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp7.i = getelementptr %struct.CGPoint* %tmp6.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp8.i = load double* %w.addr.i		; <double> [#uses=1]
	store double %tmp8.i, double* %tmp7.i
	call void @llvm.dbg.stoppoint(i32 105, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp9.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp10.i = getelementptr %struct.CGPoint* %tmp9.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp11.i = load double* %h.addr.i		; <double> [#uses=1]
	store double %tmp11.i, double* %tmp10.i
	call void @llvm.dbg.stoppoint(i32 106, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp12.i = bitcast %struct.CGRect* %retval.i129 to i8*		; <i8*> [#uses=1]
	%tmp13.i = bitcast %struct.CGRect* %r.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp12.i, i8* %tmp13.i, i32 32, i32 4) nounwind
	call void @llvm.dbg.stoppoint(i32 107, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp14.i = bitcast %struct.CGRect* %tmp122 to i8*		; <i8*> [#uses=1]
	%tmp15.i = bitcast %struct.CGRect* %retval.i129 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp14.i, i8* %tmp15.i, i32 32, i32 4) nounwind
	%tmp123 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	%tmp124 = bitcast %struct.CGRect* %tmp122 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp123, i8* %tmp124, i32 32, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %31, %cond.end89
	call void @llvm.dbg.stoppoint(i32 93, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to { }*))
	%tmp125 = bitcast %struct.CGRect* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp126 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp125, i8* %tmp126, i32 32, i32 4)
	ret void
}

declare double @objc_msgSend_fpret(%struct.objc_object*, %struct.objc_selector*, ...)

define internal void @"\01-[DVIconAndTextCell iconRectForBounds:]"(%struct.CGRect* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval.i77 = alloca double		; <double*> [#uses=2]
	%aRect74 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i70 = alloca double		; <double*> [#uses=2]
	%aRect67 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i63 = alloca double		; <double*> [#uses=2]
	%aRect60 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i56 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%x.addr.i = alloca double		; <double*> [#uses=3]
	%y.addr.i = alloca double		; <double*> [#uses=3]
	%w.addr.i = alloca double		; <double*> [#uses=3]
	%h.addr.i = alloca double		; <double*> [#uses=3]
	%r.i = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=6]
	%retval.i = alloca double		; <double*> [#uses=2]
	%aRect = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=4]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%iconSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=6]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp5 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%unionRect = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=6]
	%agg.tmp10 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp15 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%shorter = alloca i8, align 1		; <i8*> [#uses=3]
	%agg.tmp21 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%y = alloca double, align 8		; <double*> [#uses=3]
	%agg.tmp26 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp32 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp40 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp49 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable504 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable505 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable506 to { }*))
	call void @llvm.dbg.stoppoint(i32 96, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = bitcast %struct.CGPoint* %iconSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable507 to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_440"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGPoint* noalias sret %tmp5, %struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)
	%tmp6 = bitcast %struct.CGPoint* %iconSize to i8*		; <i8*> [#uses=1]
	%tmp7 = bitcast %struct.CGPoint* %tmp5 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp6, i8* %tmp7, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 97, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = bitcast %struct.CGRect* %unionRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable509 to { }*))
	%tmp9 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp11 = bitcast %struct.CGRect* %agg.tmp10 to i8*		; <i8*> [#uses=1]
	%tmp12 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp11, i8* %tmp12, i32 32, i32 4)
	%tmp13 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_511"		; <%struct.objc_selector*> [#uses=1]
	%tmp14 = bitcast %3* %tmp9 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp15, %struct.objc_object* %tmp14, %struct.objc_selector* %tmp13, %struct.CGRect* byval %agg.tmp10)
	%tmp16 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	%tmp17 = bitcast %struct.CGRect* %tmp15 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp16, i8* %tmp17, i32 32, i32 4)
	call void @llvm.dbg.stoppoint(i32 98, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = bitcast i8* %shorter to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable517 to { }*))
	%tmp19 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp20 = load double* %tmp19		; <double> [#uses=1]
	%tmp22 = bitcast %struct.CGRect* %agg.tmp21 to i8*		; <i8*> [#uses=1]
	%tmp23 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp22, i8* %tmp23, i32 32, i32 4)
	%tmp54 = bitcast %struct.CGRect* %aRect to i8*		; <i8*> [#uses=1]
	%tmp55 = bitcast %struct.CGRect* %agg.tmp21 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp54, i8* %tmp55, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%6 = bitcast %struct.CGRect* %aRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %6, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i = getelementptr %struct.CGRect* %aRect, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i = getelementptr %struct.CGPoint* %tmp.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i = load double* %tmp1.i		; <double> [#uses=1]
	store double %tmp2.i, double* %retval.i
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%7 = load double* %retval.i		; <double> [#uses=1]
	%cmp = fcmp olt double %tmp20, %7		; <i1> [#uses=1]
	%conv = zext i1 %cmp to i32		; <i32> [#uses=1]
	%conv24 = trunc i32 %conv to i8		; <i8> [#uses=1]
	store i8 %conv24, i8* %shorter
	call void @llvm.dbg.stoppoint(i32 99, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%8 = bitcast double* %y to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %8, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable518 to { }*))
	%tmp27 = bitcast %struct.CGRect* %agg.tmp26 to i8*		; <i8*> [#uses=1]
	%tmp28 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp27, i8* %tmp28, i32 32, i32 4)
	%tmp75 = bitcast %struct.CGRect* %aRect74 to i8*		; <i8*> [#uses=1]
	%tmp76 = bitcast %struct.CGRect* %agg.tmp26 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp75, i8* %tmp76, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram998 to { }*))
	%9 = bitcast %struct.CGRect* %aRect74 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %9, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable999 to { }*))
	call void @llvm.dbg.stoppoint(i32 130, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i78 = getelementptr %struct.CGRect* %aRect74, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i79 = getelementptr %struct.CGPoint* %tmp.i78, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i80 = load double* %tmp1.i79		; <double> [#uses=1]
	store double %tmp2.i80, double* %retval.i77
	call void @llvm.dbg.stoppoint(i32 131, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%10 = load double* %retval.i77		; <double> [#uses=1]
	%tmp30 = load i8* %shorter		; <i8> [#uses=1]
	%conv31 = sext i8 %tmp30 to i32		; <i32> [#uses=1]
	%tobool = icmp ne i32 %conv31, 0		; <i1> [#uses=1]
	br i1 %tobool, label %cond.true, label %cond.false

cond.true:		; preds = %entry
	%tmp33 = bitcast %struct.CGRect* %agg.tmp32 to i8*		; <i8*> [#uses=1]
	%tmp34 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp33, i8* %tmp34, i32 32, i32 4)
	%tmp68 = bitcast %struct.CGRect* %aRect67 to i8*		; <i8*> [#uses=1]
	%tmp69 = bitcast %struct.CGRect* %agg.tmp32 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp68, i8* %tmp69, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%11 = bitcast %struct.CGRect* %aRect67 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %11, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i71 = getelementptr %struct.CGRect* %aRect67, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i72 = getelementptr %struct.CGPoint* %tmp.i71, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i73 = load double* %tmp1.i72		; <double> [#uses=1]
	store double %tmp2.i73, double* %retval.i70
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%12 = load double* %retval.i70		; <double> [#uses=1]
	%div = fdiv double %12, 2.000000e+00		; <double> [#uses=1]
	%tmp36 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp37 = load double* %tmp36		; <double> [#uses=1]
	%div38 = fdiv double %tmp37, 2.000000e+00		; <double> [#uses=1]
	%sub = sub double %div, %div38		; <double> [#uses=1]
	br label %cond.end

cond.false:		; preds = %entry
	br label %cond.end

cond.end:		; preds = %cond.false, %cond.true
	%cond = phi double [ %sub, %cond.true ], [ 0.000000e+00, %cond.false ]		; <double> [#uses=1]
	%call39 = call double @floor(double %cond)		; <double> [#uses=1]
	%add = add double %10, %call39		; <double> [#uses=1]
	store double %add, double* %y
	call void @llvm.dbg.stoppoint(i32 100, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp41 = bitcast %struct.CGRect* %agg.tmp40 to i8*		; <i8*> [#uses=1]
	%tmp42 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp41, i8* %tmp42, i32 32, i32 4)
	%tmp61 = bitcast %struct.CGRect* %aRect60 to i8*		; <i8*> [#uses=1]
	%tmp62 = bitcast %struct.CGRect* %agg.tmp40 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp61, i8* %tmp62, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram995 to { }*))
	%13 = bitcast %struct.CGRect* %aRect60 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %13, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable996 to { }*))
	call void @llvm.dbg.stoppoint(i32 126, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i64 = getelementptr %struct.CGRect* %aRect60, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i65 = getelementptr %struct.CGPoint* %tmp.i64, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i66 = load double* %tmp1.i65		; <double> [#uses=1]
	store double %tmp2.i66, double* %retval.i63
	call void @llvm.dbg.stoppoint(i32 127, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%14 = load double* %retval.i63		; <double> [#uses=1]
	%tmp44 = load double* %y		; <double> [#uses=1]
	%tmp45 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp46 = load double* %tmp45		; <double> [#uses=1]
	%tmp47 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp48 = load double* %tmp47		; <double> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*)) nounwind
	store double %14, double* %x.addr.i
	%15 = bitcast double* %x.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %15, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable986 to { }*)) nounwind
	store double %tmp44, double* %y.addr.i
	%16 = bitcast double* %y.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %16, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable987 to { }*)) nounwind
	store double %tmp46, double* %w.addr.i
	%17 = bitcast double* %w.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %17, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable989 to { }*)) nounwind
	store double %tmp48, double* %h.addr.i
	%18 = bitcast double* %h.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %18, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable991 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 101, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram995 to { }*))
	%19 = bitcast %struct.CGRect* %r.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %19, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable993 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 102, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp.i57 = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i58 = getelementptr %struct.CGPoint* %tmp.i57, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i59 = load double* %x.addr.i		; <double> [#uses=1]
	store double %tmp2.i59, double* %tmp1.i58
	call void @llvm.dbg.stoppoint(i32 103, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp3.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp4.i = getelementptr %struct.CGPoint* %tmp3.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp5.i = load double* %y.addr.i		; <double> [#uses=1]
	store double %tmp5.i, double* %tmp4.i
	call void @llvm.dbg.stoppoint(i32 104, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp6.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp7.i = getelementptr %struct.CGPoint* %tmp6.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp8.i = load double* %w.addr.i		; <double> [#uses=1]
	store double %tmp8.i, double* %tmp7.i
	call void @llvm.dbg.stoppoint(i32 105, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp9.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp10.i = getelementptr %struct.CGPoint* %tmp9.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp11.i = load double* %h.addr.i		; <double> [#uses=1]
	store double %tmp11.i, double* %tmp10.i
	call void @llvm.dbg.stoppoint(i32 106, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp12.i = bitcast %struct.CGRect* %retval.i56 to i8*		; <i8*> [#uses=1]
	%tmp13.i = bitcast %struct.CGRect* %r.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp12.i, i8* %tmp13.i, i32 32, i32 4) nounwind
	call void @llvm.dbg.stoppoint(i32 107, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp14.i = bitcast %struct.CGRect* %tmp49 to i8*		; <i8*> [#uses=1]
	%tmp15.i = bitcast %struct.CGRect* %retval.i56 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp14.i, i8* %tmp15.i, i32 32, i32 4) nounwind
	%tmp50 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	%tmp51 = bitcast %struct.CGRect* %tmp49 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp50, i8* %tmp51, i32 32, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %20, %cond.end
	call void @llvm.dbg.stoppoint(i32 101, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram503 to { }*))
	%tmp52 = bitcast %struct.CGRect* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp53 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp52, i8* %tmp53, i32 32, i32 4)
	ret void
}

define internal void @"\01-[DVIconAndTextCell titleRectForBounds:]"(%struct.CGRect* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval.i130 = alloca double		; <double*> [#uses=2]
	%aRect127 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i123 = alloca double		; <double*> [#uses=2]
	%aRect120 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i116 = alloca double		; <double*> [#uses=2]
	%aRect113 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i109 = alloca double		; <double*> [#uses=2]
	%aRect106 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i102 = alloca double		; <double*> [#uses=2]
	%aRect99 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval.i95 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%x.addr.i = alloca double		; <double*> [#uses=3]
	%y.addr.i = alloca double		; <double*> [#uses=3]
	%w.addr.i = alloca double		; <double*> [#uses=3]
	%h.addr.i = alloca double		; <double*> [#uses=3]
	%r.i = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=6]
	%retval.i = alloca double		; <double*> [#uses=2]
	%aRect = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=3]
	%retval = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=6]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%textInset = alloca double, align 8		; <double*> [#uses=3]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%iconSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=3]
	%agg.tmp7 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp12 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%titleSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=5]
	%tmp19 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%unionRect = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=8]
	%agg.tmp24 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp29 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%iconIndet = alloca double, align 8		; <double*> [#uses=4]
	%shorter = alloca i8, align 1		; <i8*> [#uses=4]
	%agg.tmp39 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%y = alloca double, align 8		; <double*> [#uses=3]
	%agg.tmp45 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp51 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%height = alloca double, align 8		; <double*> [#uses=3]
	%agg.tmp68 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp74 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp81 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp88 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable521 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable522 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable523 to { }*))
	call void @llvm.dbg.stoppoint(i32 104, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = bitcast double* %textInset to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable524 to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_490"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call = call double bitcast (double (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_fpret to double (%struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)		; <double> [#uses=1]
	store double %call, double* %textInset
	call void @llvm.dbg.stoppoint(i32 105, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = bitcast %struct.CGPoint* %iconSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable525 to { }*))
	%tmp6 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp8 = bitcast %struct.CGRect* %agg.tmp7 to i8*		; <i8*> [#uses=1]
	%tmp9 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp8, i8* %tmp9, i32 32, i32 4)
	%tmp10 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_440"		; <%struct.objc_selector*> [#uses=1]
	%tmp11 = bitcast %3* %tmp6 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGPoint* noalias sret %tmp12, %struct.objc_object* %tmp11, %struct.objc_selector* %tmp10, %struct.CGRect* byval %agg.tmp7)
	%tmp13 = bitcast %struct.CGPoint* %iconSize to i8*		; <i8*> [#uses=1]
	%tmp14 = bitcast %struct.CGPoint* %tmp12 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp13, i8* %tmp14, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 106, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = bitcast %struct.CGPoint* %titleSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable527 to { }*))
	%tmp16 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp17 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_495"		; <%struct.objc_selector*> [#uses=1]
	%tmp18 = bitcast %3* %tmp16 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*)*)(%struct.CGPoint* noalias sret %tmp19, %struct.objc_object* %tmp18, %struct.objc_selector* %tmp17)
	%tmp20 = bitcast %struct.CGPoint* %titleSize to i8*		; <i8*> [#uses=1]
	%tmp21 = bitcast %struct.CGPoint* %tmp19 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp20, i8* %tmp21, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 107, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = bitcast %struct.CGRect* %unionRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %6, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable528 to { }*))
	%tmp23 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp25 = bitcast %struct.CGRect* %agg.tmp24 to i8*		; <i8*> [#uses=1]
	%tmp26 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp25, i8* %tmp26, i32 32, i32 4)
	%tmp27 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_511"		; <%struct.objc_selector*> [#uses=1]
	%tmp28 = bitcast %3* %tmp23 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp29, %struct.objc_object* %tmp28, %struct.objc_selector* %tmp27, %struct.CGRect* byval %agg.tmp24)
	%tmp30 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	%tmp31 = bitcast %struct.CGRect* %tmp29 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp30, i8* %tmp31, i32 32, i32 4)
	call void @llvm.dbg.stoppoint(i32 108, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%7 = bitcast double* %iconIndet to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %7, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable530 to { }*))
	%tmp33 = getelementptr %struct.CGPoint* %iconSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp34 = load double* %tmp33		; <double> [#uses=1]
	%tmp35 = load double* %textInset		; <double> [#uses=1]
	%add = add double %tmp34, %tmp35		; <double> [#uses=1]
	store double %add, double* %iconIndet
	call void @llvm.dbg.stoppoint(i32 109, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%8 = bitcast i8* %shorter to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %8, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable531 to { }*))
	%tmp37 = getelementptr %struct.CGPoint* %titleSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp38 = load double* %tmp37		; <double> [#uses=1]
	%tmp40 = bitcast %struct.CGRect* %agg.tmp39 to i8*		; <i8*> [#uses=1]
	%tmp41 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp40, i8* %tmp41, i32 32, i32 4)
	%tmp93 = bitcast %struct.CGRect* %aRect to i8*		; <i8*> [#uses=1]
	%tmp94 = bitcast %struct.CGRect* %agg.tmp39 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp93, i8* %tmp94, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%9 = bitcast %struct.CGRect* %aRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %9, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i = getelementptr %struct.CGRect* %aRect, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i = getelementptr %struct.CGPoint* %tmp.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i = load double* %tmp1.i		; <double> [#uses=1]
	store double %tmp2.i, double* %retval.i
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%10 = load double* %retval.i		; <double> [#uses=1]
	%cmp = fcmp olt double %tmp38, %10		; <i1> [#uses=1]
	%conv = zext i1 %cmp to i32		; <i32> [#uses=1]
	%conv43 = trunc i32 %conv to i8		; <i8> [#uses=1]
	store i8 %conv43, i8* %shorter
	call void @llvm.dbg.stoppoint(i32 110, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%11 = bitcast double* %y to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %11, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable532 to { }*))
	%tmp46 = bitcast %struct.CGRect* %agg.tmp45 to i8*		; <i8*> [#uses=1]
	%tmp47 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp46, i8* %tmp47, i32 32, i32 4)
	%tmp128 = bitcast %struct.CGRect* %aRect127 to i8*		; <i8*> [#uses=1]
	%tmp129 = bitcast %struct.CGRect* %agg.tmp45 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp128, i8* %tmp129, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram998 to { }*))
	%12 = bitcast %struct.CGRect* %aRect127 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %12, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable999 to { }*))
	call void @llvm.dbg.stoppoint(i32 130, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i131 = getelementptr %struct.CGRect* %aRect127, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i132 = getelementptr %struct.CGPoint* %tmp.i131, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i133 = load double* %tmp1.i132		; <double> [#uses=1]
	store double %tmp2.i133, double* %retval.i130
	call void @llvm.dbg.stoppoint(i32 131, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%13 = load double* %retval.i130		; <double> [#uses=1]
	%tmp49 = load i8* %shorter		; <i8> [#uses=1]
	%conv50 = sext i8 %tmp49 to i32		; <i32> [#uses=1]
	%tobool = icmp ne i32 %conv50, 0		; <i1> [#uses=1]
	br i1 %tobool, label %cond.true, label %cond.false

cond.true:		; preds = %entry
	%tmp52 = bitcast %struct.CGRect* %agg.tmp51 to i8*		; <i8*> [#uses=1]
	%tmp53 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp52, i8* %tmp53, i32 32, i32 4)
	%tmp121 = bitcast %struct.CGRect* %aRect120 to i8*		; <i8*> [#uses=1]
	%tmp122 = bitcast %struct.CGRect* %agg.tmp51 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp121, i8* %tmp122, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%14 = bitcast %struct.CGRect* %aRect120 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %14, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i124 = getelementptr %struct.CGRect* %aRect120, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i125 = getelementptr %struct.CGPoint* %tmp.i124, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i126 = load double* %tmp1.i125		; <double> [#uses=1]
	store double %tmp2.i126, double* %retval.i123
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%15 = load double* %retval.i123		; <double> [#uses=1]
	%div = fdiv double %15, 2.000000e+00		; <double> [#uses=1]
	%tmp55 = getelementptr %struct.CGPoint* %titleSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp56 = load double* %tmp55		; <double> [#uses=1]
	%div57 = fdiv double %tmp56, 2.000000e+00		; <double> [#uses=1]
	%sub = sub double %div, %div57		; <double> [#uses=1]
	br label %cond.end

cond.false:		; preds = %entry
	br label %cond.end

cond.end:		; preds = %cond.false, %cond.true
	%cond = phi double [ %sub, %cond.true ], [ 0.000000e+00, %cond.false ]		; <double> [#uses=1]
	%call58 = call double @floor(double %cond)		; <double> [#uses=1]
	%add59 = add double %13, %call58		; <double> [#uses=1]
	store double %add59, double* %y
	call void @llvm.dbg.stoppoint(i32 111, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%16 = bitcast double* %height to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %16, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable533 to { }*))
	%tmp61 = load i8* %shorter		; <i8> [#uses=1]
	%conv62 = sext i8 %tmp61 to i32		; <i32> [#uses=1]
	%tobool63 = icmp ne i32 %conv62, 0		; <i1> [#uses=1]
	br i1 %tobool63, label %cond.true64, label %cond.false67

cond.true64:		; preds = %cond.end
	%tmp65 = getelementptr %struct.CGPoint* %titleSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp66 = load double* %tmp65		; <double> [#uses=1]
	br label %cond.end72

cond.false67:		; preds = %cond.end
	%tmp69 = bitcast %struct.CGRect* %agg.tmp68 to i8*		; <i8*> [#uses=1]
	%tmp70 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp69, i8* %tmp70, i32 32, i32 4)
	%tmp114 = bitcast %struct.CGRect* %aRect113 to i8*		; <i8*> [#uses=1]
	%tmp115 = bitcast %struct.CGRect* %agg.tmp68 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp114, i8* %tmp115, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1001 to { }*))
	%17 = bitcast %struct.CGRect* %aRect113 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %17, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1002 to { }*))
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i117 = getelementptr %struct.CGRect* %aRect113, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i118 = getelementptr %struct.CGPoint* %tmp.i117, i32 0, i32 1		; <double*> [#uses=1]
	%tmp2.i119 = load double* %tmp1.i118		; <double> [#uses=1]
	store double %tmp2.i119, double* %retval.i116
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%18 = load double* %retval.i116		; <double> [#uses=1]
	br label %cond.end72

cond.end72:		; preds = %cond.false67, %cond.true64
	%cond73 = phi double [ %tmp66, %cond.true64 ], [ %18, %cond.false67 ]		; <double> [#uses=1]
	store double %cond73, double* %height
	call void @llvm.dbg.stoppoint(i32 112, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp75 = bitcast %struct.CGRect* %agg.tmp74 to i8*		; <i8*> [#uses=1]
	%tmp76 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp75, i8* %tmp76, i32 32, i32 4)
	%tmp107 = bitcast %struct.CGRect* %aRect106 to i8*		; <i8*> [#uses=1]
	%tmp108 = bitcast %struct.CGRect* %agg.tmp74 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp107, i8* %tmp108, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram995 to { }*))
	%19 = bitcast %struct.CGRect* %aRect106 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %19, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable996 to { }*))
	call void @llvm.dbg.stoppoint(i32 126, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i110 = getelementptr %struct.CGRect* %aRect106, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i111 = getelementptr %struct.CGPoint* %tmp.i110, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i112 = load double* %tmp1.i111		; <double> [#uses=1]
	store double %tmp2.i112, double* %retval.i109
	call void @llvm.dbg.stoppoint(i32 127, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%20 = load double* %retval.i109		; <double> [#uses=1]
	%tmp78 = load double* %iconIndet		; <double> [#uses=1]
	%add79 = add double %20, %tmp78		; <double> [#uses=1]
	%tmp80 = load double* %y		; <double> [#uses=1]
	%tmp82 = bitcast %struct.CGRect* %agg.tmp81 to i8*		; <i8*> [#uses=1]
	%tmp83 = bitcast %struct.CGRect* %unionRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp82, i8* %tmp83, i32 32, i32 4)
	%tmp100 = bitcast %struct.CGRect* %aRect99 to i8*		; <i8*> [#uses=1]
	%tmp101 = bitcast %struct.CGRect* %agg.tmp81 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp100, i8* %tmp101, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1010 to { }*))
	%21 = bitcast %struct.CGRect* %aRect99 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %21, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1011 to { }*))
	call void @llvm.dbg.stoppoint(i32 134, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram995 to { }*))
	%tmp.i103 = getelementptr %struct.CGRect* %aRect99, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i104 = getelementptr %struct.CGPoint* %tmp.i103, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i105 = load double* %tmp1.i104		; <double> [#uses=1]
	store double %tmp2.i105, double* %retval.i102
	call void @llvm.dbg.stoppoint(i32 135, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%22 = load double* %retval.i102		; <double> [#uses=1]
	%tmp85 = load double* %iconIndet		; <double> [#uses=1]
	%sub86 = sub double %22, %tmp85		; <double> [#uses=1]
	%tmp87 = load double* %height		; <double> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*)) nounwind
	store double %add79, double* %x.addr.i
	%23 = bitcast double* %x.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %23, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable986 to { }*)) nounwind
	store double %tmp80, double* %y.addr.i
	%24 = bitcast double* %y.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %24, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable987 to { }*)) nounwind
	store double %sub86, double* %w.addr.i
	%25 = bitcast double* %w.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %25, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable989 to { }*)) nounwind
	store double %tmp87, double* %h.addr.i
	%26 = bitcast double* %h.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %26, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable991 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 101, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1010 to { }*))
	%27 = bitcast %struct.CGRect* %r.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %27, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable993 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 102, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp.i96 = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i97 = getelementptr %struct.CGPoint* %tmp.i96, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i98 = load double* %x.addr.i		; <double> [#uses=1]
	store double %tmp2.i98, double* %tmp1.i97
	call void @llvm.dbg.stoppoint(i32 103, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp3.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp4.i = getelementptr %struct.CGPoint* %tmp3.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp5.i = load double* %y.addr.i		; <double> [#uses=1]
	store double %tmp5.i, double* %tmp4.i
	call void @llvm.dbg.stoppoint(i32 104, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp6.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp7.i = getelementptr %struct.CGPoint* %tmp6.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp8.i = load double* %w.addr.i		; <double> [#uses=1]
	store double %tmp8.i, double* %tmp7.i
	call void @llvm.dbg.stoppoint(i32 105, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp9.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp10.i = getelementptr %struct.CGPoint* %tmp9.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp11.i = load double* %h.addr.i		; <double> [#uses=1]
	store double %tmp11.i, double* %tmp10.i
	call void @llvm.dbg.stoppoint(i32 106, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp12.i = bitcast %struct.CGRect* %retval.i95 to i8*		; <i8*> [#uses=1]
	%tmp13.i = bitcast %struct.CGRect* %r.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp12.i, i8* %tmp13.i, i32 32, i32 4) nounwind
	call void @llvm.dbg.stoppoint(i32 107, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp14.i = bitcast %struct.CGRect* %tmp88 to i8*		; <i8*> [#uses=1]
	%tmp15.i = bitcast %struct.CGRect* %retval.i95 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp14.i, i8* %tmp15.i, i32 32, i32 4) nounwind
	%tmp89 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	%tmp90 = bitcast %struct.CGRect* %tmp88 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp89, i8* %tmp90, i32 32, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %28, %cond.end72
	call void @llvm.dbg.stoppoint(i32 113, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram520 to { }*))
	%tmp91 = bitcast %struct.CGRect* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp92 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp91, i8* %tmp92, i32 32, i32 4)
	ret void
}

define internal void @"\01-[DVIconAndTextCell expansionFrameWithFrame:inView:]"(%struct.CGRect* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %cellFrame, %8* %view) nounwind {
entry:
	%retval = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%view.addr = alloca %8*		; <%8**> [#uses=2]
	%expansionFrame = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=7]
	%tmp8 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp13 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram535 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable536 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable537 to { }*))
	%2 = bitcast %struct.CGRect* %cellFrame to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable539 to { }*))
	store %8* %view, %8** %view.addr
	%3 = bitcast %8** %view.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable876 to { }*))
	call void @llvm.dbg.stoppoint(i32 116, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = bitcast %struct.CGRect* %expansionFrame to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable878 to { }*))
	%tmp = getelementptr %struct.CGRect* %expansionFrame, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1 = getelementptr %struct.CGRect* %cellFrame, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp2 = bitcast %struct.CGPoint* %tmp to i8*		; <i8*> [#uses=1]
	%tmp3 = bitcast %struct.CGPoint* %tmp1 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp2, i8* %tmp3, i32 16, i32 4)
	%tmp4 = getelementptr %struct.CGRect* %expansionFrame, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp5 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp6 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_880"		; <%struct.objc_selector*> [#uses=1]
	%tmp7 = bitcast %3* %tmp5 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, %struct.objc_object*, %struct.objc_selector*)*)(%struct.CGPoint* noalias sret %tmp8, %struct.objc_object* %tmp7, %struct.objc_selector* %tmp6)
	%tmp9 = bitcast %struct.CGPoint* %tmp4 to i8*		; <i8*> [#uses=1]
	%tmp10 = bitcast %struct.CGPoint* %tmp8 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp9, i8* %tmp10, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 117, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp11 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp12 = bitcast %struct.CGRect* %cellFrame to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp11, i8* %tmp12, i32 32, i32 4)
	%tmp14 = bitcast %struct.CGRect* %agg.tmp13 to i8*		; <i8*> [#uses=1]
	%tmp15 = bitcast %struct.CGRect* %expansionFrame to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp14, i8* %tmp15, i32 32, i32 4)
	%call = call signext i8 @NSContainsRect(%struct.CGRect* byval %agg.tmp, %struct.CGRect* byval %agg.tmp13)		; <i8> [#uses=1]
	%tobool = icmp ne i8 %call, 0		; <i1> [#uses=1]
	br i1 %tobool, label %if.else, label %if.then

if.then:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 118, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp16 = getelementptr %struct.CGRect* %expansionFrame, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp17 = getelementptr %struct.CGPoint* %tmp16, i32 0, i32 1		; <double*> [#uses=2]
	%tmp18 = load double* %tmp17		; <double> [#uses=1]
	%add = add double %tmp18, 1.000000e+00		; <double> [#uses=1]
	store double %add, double* %tmp17
	br label %if.end

if.else:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 120, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp19 = bitcast %struct.CGRect* %expansionFrame to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp19, i8* bitcast (%struct.CGRect* @NSZeroRect to i8*), i32 32, i32 4)
	br label %if.end

if.end:		; preds = %if.else, %if.then
	call void @llvm.dbg.stoppoint(i32 122, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp20 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	%tmp21 = bitcast %struct.CGRect* %expansionFrame to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp20, i8* %tmp21, i32 32, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %5, %if.end
	call void @llvm.dbg.stoppoint(i32 123, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram535 to { }*))
	%tmp22 = bitcast %struct.CGRect* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp23 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp22, i8* %tmp23, i32 32, i32 4)
	ret void
}

declare signext i8 @NSContainsRect(%struct.CGRect* byval, %struct.CGRect* byval)

define internal void @"\01-[DVIconAndTextCell cellSizeForBounds:]"(%struct.CGPoint* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval.i = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%w.addr.i = alloca double		; <double*> [#uses=3]
	%h.addr.i = alloca double		; <double*> [#uses=3]
	%s.i = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=4]
	%retval = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=4]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%iconInset = alloca double, align 8		; <double*> [#uses=3]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%contentSize = alloca %struct.CGPoint, align 4		; <%struct.CGPoint*> [#uses=4]
	%agg.tmp7 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp12 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp21 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram882 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable883 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable884 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable885 to { }*))
	call void @llvm.dbg.stoppoint(i32 126, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = bitcast double* %iconInset to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable886 to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_486"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call = call double bitcast (double (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_fpret to double (%struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)		; <double> [#uses=1]
	store double %call, double* %iconInset
	call void @llvm.dbg.stoppoint(i32 127, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = bitcast %struct.CGPoint* %contentSize to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable888 to { }*))
	%tmp6 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp8 = bitcast %struct.CGRect* %agg.tmp7 to i8*		; <i8*> [#uses=1]
	%tmp9 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp8, i8* %tmp9, i32 32, i32 4)
	%tmp10 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_511"		; <%struct.objc_selector*> [#uses=1]
	%tmp11 = bitcast %3* %tmp6 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp12, %struct.objc_object* %tmp11, %struct.objc_selector* %tmp10, %struct.CGRect* byval %agg.tmp7)
	%tmp13 = getelementptr %struct.CGRect* %tmp12, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp14 = bitcast %struct.CGPoint* %contentSize to i8*		; <i8*> [#uses=1]
	%tmp15 = bitcast %struct.CGPoint* %tmp13 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp14, i8* %tmp15, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 128, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp16 = getelementptr %struct.CGPoint* %contentSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp17 = load double* %tmp16		; <double> [#uses=1]
	%tmp18 = load double* %iconInset		; <double> [#uses=1]
	%add = add double %tmp17, %tmp18		; <double> [#uses=1]
	%tmp19 = getelementptr %struct.CGPoint* %contentSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp20 = load double* %tmp19		; <double> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1004 to { }*)) nounwind
	store double %add, double* %w.addr.i
	%5 = bitcast double* %w.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1005 to { }*)) nounwind
	store double %tmp20, double* %h.addr.i
	%6 = bitcast double* %h.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %6, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1006 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 94, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%7 = bitcast %struct.CGPoint* %s.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %7, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1008 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 95, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp.i = getelementptr %struct.CGPoint* %s.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp1.i = load double* %w.addr.i		; <double> [#uses=1]
	store double %tmp1.i, double* %tmp.i
	call void @llvm.dbg.stoppoint(i32 96, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp2.i = getelementptr %struct.CGPoint* %s.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp3.i = load double* %h.addr.i		; <double> [#uses=1]
	store double %tmp3.i, double* %tmp2.i
	call void @llvm.dbg.stoppoint(i32 97, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp4.i = bitcast %struct.CGPoint* %retval.i to i8*		; <i8*> [#uses=1]
	%tmp5.i = bitcast %struct.CGPoint* %s.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp4.i, i8* %tmp5.i, i32 16, i32 4) nounwind
	call void @llvm.dbg.stoppoint(i32 98, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp6.i = bitcast %struct.CGPoint* %tmp21 to i8*		; <i8*> [#uses=1]
	%tmp7.i = bitcast %struct.CGPoint* %retval.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp6.i, i8* %tmp7.i, i32 16, i32 4) nounwind
	%tmp22 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	%tmp23 = bitcast %struct.CGPoint* %tmp21 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp22, i8* %tmp23, i32 16, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %8, %entry
	call void @llvm.dbg.stoppoint(i32 129, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram882 to { }*))
	%tmp24 = bitcast %struct.CGPoint* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp25 = bitcast %struct.CGPoint* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp24, i8* %tmp25, i32 16, i32 4)
	ret void
}

define internal signext i8 @"\01-[DVIconAndTextCell isOpaque]"(%3* %self, %struct.objc_selector* %_cmd) nounwind {
entry:
	%retval = alloca i8		; <i8*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=2]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram890 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable891 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable892 to { }*))
	call void @llvm.dbg.stoppoint(i32 132, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i8 0, i8* %retval
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %2, %entry
	call void @llvm.dbg.stoppoint(i32 133, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram890 to { }*))
	%3 = load i8* %retval		; <i8> [#uses=1]
	ret i8 %3
}

define internal void @"\01-[DVIconAndTextCell drawWithExpansionFrame:inView:]"(%3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %cellFrame, %8* %view) nounwind {
entry:
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%view.addr = alloca %8*		; <%8**> [#uses=3]
	%agg.tmp = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=3]
	%tmp4 = alloca %struct.CGPoint		; <%struct.CGPoint*> [#uses=2]
	%agg.tmp12 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram894 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable895 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable896 to { }*))
	%2 = bitcast %struct.CGRect* %cellFrame to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable897 to { }*))
	store %8* %view, %8** %view.addr
	%3 = bitcast %8** %view.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable898 to { }*))
	call void @llvm.dbg.stoppoint(i32 137, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = getelementptr %struct.CGRect* %cellFrame, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1 = getelementptr %struct.CGRect* %cellFrame, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp2 = bitcast %struct.CGPoint* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp3 = bitcast %struct.CGPoint* %tmp1 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp2, i8* %tmp3, i32 16, i32 4)
	%tmp5 = getelementptr %struct.CGPoint* %agg.tmp, i32 0, i32 0		; <double*> [#uses=1]
	%tmp6 = load double* %tmp5		; <double> [#uses=1]
	%tmp7 = getelementptr %struct.CGPoint* %agg.tmp, i32 0, i32 1		; <double*> [#uses=1]
	%tmp8 = load double* %tmp7		; <double> [#uses=1]
	call void @DVPointByFlooringPoint(%struct.CGPoint* noalias sret %tmp4, double %tmp6, double %tmp8)
	%tmp9 = bitcast %struct.CGPoint* %tmp to i8*		; <i8*> [#uses=1]
	%tmp10 = bitcast %struct.CGPoint* %tmp4 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp9, i8* %tmp10, i32 16, i32 4)
	call void @llvm.dbg.stoppoint(i32 138, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%self11 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp13 = bitcast %struct.CGRect* %agg.tmp12 to i8*		; <i8*> [#uses=1]
	%tmp14 = bitcast %struct.CGRect* %cellFrame to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp13, i8* %tmp14, i32 32, i32 4)
	%tmp15 = load %8** %view.addr		; <%8*> [#uses=1]
	%objc_super = alloca %struct._objc_super		; <%struct._objc_super*> [#uses=3]
	%4 = bitcast %3* %self11 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%5 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 0		; <%struct.objc_object**> [#uses=1]
	store %struct.objc_object* %4, %struct.objc_object** %5
	%tmp16 = load %struct._objc_class** @"\01L_OBJC_CLASS_REFERENCES_"		; <%struct._objc_class*> [#uses=1]
	%6 = bitcast %struct._objc_class* %tmp16 to %struct.objc_class*		; <%struct.objc_class*> [#uses=1]
	%7 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 1		; <%struct.objc_class**> [#uses=1]
	store %struct.objc_class* %6, %struct.objc_class** %7
	%tmp17 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_900"		; <%struct.objc_selector*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct._objc_super*, %struct.objc_selector*, ...)* @objc_msgSendSuper to void (%struct._objc_super*, %struct.objc_selector*, %struct.CGRect*, %8*)*)(%struct._objc_super* %objc_super, %struct.objc_selector* %tmp17, %struct.CGRect* byval %agg.tmp12, %8* %tmp15)
	call void @llvm.dbg.stoppoint(i32 139, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram894 to { }*))
	ret void
}

declare void @DVPointByFlooringPoint(%struct.CGPoint* noalias sret, double, double)

define internal void @"\01-[DVIconAndTextCell drawInteriorWithFrame:inView:]"(%3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds, %8* %controlView) nounwind {
entry:
	%self.addr = alloca %3*		; <%3**> [#uses=7]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%controlView.addr = alloca %8*		; <%8**> [#uses=3]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp2 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp4 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp9 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp16 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp18 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp23 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%attributedTitle = alloca %6*, align 4		; <%6**> [#uses=3]
	%titleRect = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=3]
	%agg.tmp39 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp44 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp48 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable903 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable904 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable905 to { }*))
	store %8* %controlView, %8** %controlView.addr
	%3 = bitcast %8** %controlView.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable907 to { }*))
	call void @llvm.dbg.stoppoint(i32 142, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp, i8* %tmp1, i32 32, i32 4)
	%tmp3 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp5 = bitcast %struct.CGRect* %agg.tmp4 to i8*		; <i8*> [#uses=1]
	%tmp6 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp5, i8* %tmp6, i32 32, i32 4)
	%tmp7 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_909"		; <%struct.objc_selector*> [#uses=1]
	%tmp8 = bitcast %3* %tmp3 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp9, %struct.objc_object* %tmp8, %struct.objc_selector* %tmp7, %struct.CGRect* byval %agg.tmp4)
	%tmp10 = bitcast %struct.CGRect* %agg.tmp2 to i8*		; <i8*> [#uses=1]
	%tmp11 = bitcast %struct.CGRect* %tmp9 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp10, i8* %tmp11, i32 32, i32 4)
	%call = call signext i8 @NSContainsRect(%struct.CGRect* byval %agg.tmp, %struct.CGRect* byval %agg.tmp2)		; <i8> [#uses=1]
	%tobool = icmp ne i8 %call, 0		; <i1> [#uses=1]
	br i1 %tobool, label %if.then, label %if.end

if.then:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 143, i32 9, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp12 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp13 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_431"		; <%struct.objc_selector*> [#uses=1]
	%tmp14 = bitcast %3* %tmp12 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call15 = call %5* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %5* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp14, %struct.objc_selector* %tmp13)		; <%5*> [#uses=1]
	%tmp17 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp19 = bitcast %struct.CGRect* %agg.tmp18 to i8*		; <i8*> [#uses=1]
	%tmp20 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp19, i8* %tmp20, i32 32, i32 4)
	%tmp21 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_909"		; <%struct.objc_selector*> [#uses=1]
	%tmp22 = bitcast %3* %tmp17 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp23, %struct.objc_object* %tmp22, %struct.objc_selector* %tmp21, %struct.CGRect* byval %agg.tmp18)
	%tmp24 = bitcast %struct.CGRect* %agg.tmp16 to i8*		; <i8*> [#uses=1]
	%tmp25 = bitcast %struct.CGRect* %tmp23 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp24, i8* %tmp25, i32 32, i32 4)
	%tmp26 = load %8** %controlView.addr		; <%8*> [#uses=1]
	%tmp27 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_911"		; <%struct.objc_selector*> [#uses=1]
	%tmp28 = bitcast %8* %tmp26 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call29 = call signext i8 bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to i8 (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp28, %struct.objc_selector* %tmp27)		; <i8> [#uses=1]
	%tmp30 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_913"		; <%struct.objc_selector*> [#uses=1]
	%tmp31 = bitcast %5* %call15 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.objc_object*, %struct.objc_selector*, %struct.CGRect*, i32, double, i8)*)(%struct.objc_object* %tmp31, %struct.objc_selector* %tmp30, %struct.CGRect* byval %agg.tmp16, i32 2, double 1.000000e+00, i8 signext %call29)
	br label %if.end

if.end:		; preds = %if.then, %entry
	call void @llvm.dbg.stoppoint(i32 145, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = bitcast %6** %attributedTitle to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable915 to { }*))
	%tmp33 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp34 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_477"		; <%struct.objc_selector*> [#uses=1]
	%tmp35 = bitcast %3* %tmp33 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%call36 = call %6* bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to %6* (%struct.objc_object*, %struct.objc_selector*)*)(%struct.objc_object* %tmp35, %struct.objc_selector* %tmp34)		; <%6*> [#uses=1]
	store %6* %call36, %6** %attributedTitle
	call void @llvm.dbg.stoppoint(i32 146, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = bitcast %struct.CGRect* %titleRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable917 to { }*))
	%tmp38 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp40 = bitcast %struct.CGRect* %agg.tmp39 to i8*		; <i8*> [#uses=1]
	%tmp41 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp40, i8* %tmp41, i32 32, i32 4)
	%tmp42 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_919"		; <%struct.objc_selector*> [#uses=1]
	%tmp43 = bitcast %3* %tmp38 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp44, %struct.objc_object* %tmp43, %struct.objc_selector* %tmp42, %struct.CGRect* byval %agg.tmp39)
	%tmp45 = bitcast %struct.CGRect* %titleRect to i8*		; <i8*> [#uses=1]
	%tmp46 = bitcast %struct.CGRect* %tmp44 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp45, i8* %tmp46, i32 32, i32 4)
	call void @llvm.dbg.stoppoint(i32 147, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp47 = load %6** %attributedTitle		; <%6*> [#uses=1]
	%tmp49 = bitcast %struct.CGRect* %agg.tmp48 to i8*		; <i8*> [#uses=1]
	%tmp50 = bitcast %struct.CGRect* %titleRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp49, i8* %tmp50, i32 32, i32 4)
	%tmp51 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_921"		; <%struct.objc_selector*> [#uses=1]
	%tmp52 = bitcast %6* %tmp47 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.objc_object* %tmp52, %struct.objc_selector* %tmp51, %struct.CGRect* byval %agg.tmp48)
	call void @llvm.dbg.stoppoint(i32 148, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram902 to { }*))
	ret void
}

define internal void @"\01-[DVIconAndTextCell titleEditingRectForBounds:]"(%struct.CGRect* noalias sret %agg.result, %3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %bounds) nounwind {
entry:
	%retval.i46 = alloca double		; <double*> [#uses=2]
	%aRect43 = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=4]
	%retval.i36 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%x.addr.i = alloca double		; <double*> [#uses=3]
	%y.addr.i = alloca double		; <double*> [#uses=3]
	%w.addr.i = alloca double		; <double*> [#uses=3]
	%h.addr.i = alloca double		; <double*> [#uses=3]
	%r.i = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=6]
	%retval.i = alloca double		; <double*> [#uses=2]
	%aRect = alloca %struct.CGRect, align 8		; <%struct.CGRect*> [#uses=4]
	%retval = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%self.addr = alloca %3*		; <%3**> [#uses=3]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%titleRect = alloca %struct.CGRect, align 4		; <%struct.CGRect*> [#uses=7]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp5 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp17 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp20 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp29 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram923 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable924 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable925 to { }*))
	%2 = bitcast %struct.CGRect* %bounds to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable926 to { }*))
	call void @llvm.dbg.stoppoint(i32 151, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = bitcast %struct.CGRect* %titleRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable927 to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_919"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp5, %struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)
	%tmp6 = bitcast %struct.CGRect* %titleRect to i8*		; <i8*> [#uses=1]
	%tmp7 = bitcast %struct.CGRect* %tmp5 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp6, i8* %tmp7, i32 32, i32 4)
	call void @llvm.dbg.stoppoint(i32 152, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp8 = getelementptr %struct.CGRect* %titleRect, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp9 = getelementptr %struct.CGPoint* %tmp8, i32 0, i32 0		; <double*> [#uses=1]
	%tmp10 = load double* %tmp9		; <double> [#uses=1]
	%sub = sub double %tmp10, 2.000000e+00		; <double> [#uses=1]
	%tmp11 = getelementptr %struct.CGRect* %titleRect, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp12 = getelementptr %struct.CGPoint* %tmp11, i32 0, i32 1		; <double*> [#uses=1]
	%tmp13 = load double* %tmp12		; <double> [#uses=1]
	%tmp14 = getelementptr %struct.CGRect* %titleRect, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp15 = getelementptr %struct.CGPoint* %tmp14, i32 0, i32 0		; <double*> [#uses=1]
	%tmp16 = load double* %tmp15		; <double> [#uses=1]
	%tmp18 = bitcast %struct.CGRect* %agg.tmp17 to i8*		; <i8*> [#uses=1]
	%tmp19 = bitcast %struct.CGRect* %bounds to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp18, i8* %tmp19, i32 32, i32 4)
	%tmp34 = bitcast %struct.CGRect* %aRect to i8*		; <i8*> [#uses=1]
	%tmp35 = bitcast %struct.CGRect* %agg.tmp17 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp34, i8* %tmp35, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram981 to { }*))
	%4 = bitcast %struct.CGRect* %aRect to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable983 to { }*))
	call void @llvm.dbg.stoppoint(i32 110, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i = getelementptr %struct.CGRect* %aRect, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i = getelementptr %struct.CGPoint* %tmp.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i = load double* %tmp1.i		; <double> [#uses=1]
	%tmp3.i = getelementptr %struct.CGRect* %aRect, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp4.i = getelementptr %struct.CGPoint* %tmp3.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp5.i = load double* %tmp4.i		; <double> [#uses=1]
	%add.i = add double %tmp2.i, %tmp5.i		; <double> [#uses=1]
	store double %add.i, double* %retval.i
	call void @llvm.dbg.stoppoint(i32 111, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%5 = load double* %retval.i		; <double> [#uses=1]
	%add = add double %tmp16, %5		; <double> [#uses=1]
	%tmp21 = bitcast %struct.CGRect* %agg.tmp20 to i8*		; <i8*> [#uses=1]
	%tmp22 = bitcast %struct.CGRect* %titleRect to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp21, i8* %tmp22, i32 32, i32 4)
	%tmp44 = bitcast %struct.CGRect* %aRect43 to i8*		; <i8*> [#uses=1]
	%tmp45 = bitcast %struct.CGRect* %agg.tmp20 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64(i8* %tmp44, i8* %tmp45, i64 32, i32 1)
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram981 to { }*))
	%6 = bitcast %struct.CGRect* %aRect43 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %6, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable983 to { }*))
	call void @llvm.dbg.stoppoint(i32 110, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%tmp.i47 = getelementptr %struct.CGRect* %aRect43, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i48 = getelementptr %struct.CGPoint* %tmp.i47, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i49 = load double* %tmp1.i48		; <double> [#uses=1]
	%tmp3.i50 = getelementptr %struct.CGRect* %aRect43, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp4.i51 = getelementptr %struct.CGPoint* %tmp3.i50, i32 0, i32 0		; <double*> [#uses=1]
	%tmp5.i52 = load double* %tmp4.i51		; <double> [#uses=1]
	%add.i53 = add double %tmp2.i49, %tmp5.i52		; <double> [#uses=1]
	store double %add.i53, double* %retval.i46
	call void @llvm.dbg.stoppoint(i32 111, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*))
	%7 = load double* %retval.i46		; <double> [#uses=1]
	%sub24 = sub double %add, %7		; <double> [#uses=1]
	%sub25 = sub double %sub24, 4.000000e+00		; <double> [#uses=1]
	%tmp26 = getelementptr %struct.CGRect* %titleRect, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp27 = getelementptr %struct.CGPoint* %tmp26, i32 0, i32 1		; <double*> [#uses=1]
	%tmp28 = load double* %tmp27		; <double> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram985 to { }*)) nounwind
	store double %sub, double* %x.addr.i
	%8 = bitcast double* %x.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %8, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable986 to { }*)) nounwind
	store double %tmp13, double* %y.addr.i
	%9 = bitcast double* %y.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %9, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable987 to { }*)) nounwind
	store double %sub25, double* %w.addr.i
	%10 = bitcast double* %w.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %10, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable989 to { }*)) nounwind
	store double %tmp28, double* %h.addr.i
	%11 = bitcast double* %h.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %11, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable991 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 101, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram981 to { }*))
	%12 = bitcast %struct.CGRect* %r.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %12, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable993 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 102, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp.i37 = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp1.i38 = getelementptr %struct.CGPoint* %tmp.i37, i32 0, i32 0		; <double*> [#uses=1]
	%tmp2.i39 = load double* %x.addr.i		; <double> [#uses=1]
	store double %tmp2.i39, double* %tmp1.i38
	call void @llvm.dbg.stoppoint(i32 103, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp3.i40 = getelementptr %struct.CGRect* %r.i, i32 0, i32 0		; <%struct.CGPoint*> [#uses=1]
	%tmp4.i41 = getelementptr %struct.CGPoint* %tmp3.i40, i32 0, i32 1		; <double*> [#uses=1]
	%tmp5.i42 = load double* %y.addr.i		; <double> [#uses=1]
	store double %tmp5.i42, double* %tmp4.i41
	call void @llvm.dbg.stoppoint(i32 104, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp6.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp7.i = getelementptr %struct.CGPoint* %tmp6.i, i32 0, i32 0		; <double*> [#uses=1]
	%tmp8.i = load double* %w.addr.i		; <double> [#uses=1]
	store double %tmp8.i, double* %tmp7.i
	call void @llvm.dbg.stoppoint(i32 105, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp9.i = getelementptr %struct.CGRect* %r.i, i32 0, i32 1		; <%struct.CGPoint*> [#uses=1]
	%tmp10.i = getelementptr %struct.CGPoint* %tmp9.i, i32 0, i32 1		; <double*> [#uses=1]
	%tmp11.i = load double* %h.addr.i		; <double> [#uses=1]
	store double %tmp11.i, double* %tmp10.i
	call void @llvm.dbg.stoppoint(i32 106, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp12.i = bitcast %struct.CGRect* %retval.i36 to i8*		; <i8*> [#uses=1]
	%tmp13.i = bitcast %struct.CGRect* %r.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp12.i, i8* %tmp13.i, i32 32, i32 4) nounwind
	call void @llvm.dbg.stoppoint(i32 107, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit273 to { }*)) nounwind
	%tmp14.i = bitcast %struct.CGRect* %tmp29 to i8*		; <i8*> [#uses=1]
	%tmp15.i = bitcast %struct.CGRect* %retval.i36 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp14.i, i8* %tmp15.i, i32 32, i32 4) nounwind
	%tmp30 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	%tmp31 = bitcast %struct.CGRect* %tmp29 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp30, i8* %tmp31, i32 32, i32 4)
	br label %return
		; No predecessors!
	br label %return

return:		; preds = %13, %entry
	call void @llvm.dbg.stoppoint(i32 153, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram923 to { }*))
	%tmp32 = bitcast %struct.CGRect* %agg.result to i8*		; <i8*> [#uses=1]
	%tmp33 = bitcast %struct.CGRect* %retval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp32, i8* %tmp33, i32 32, i32 4)
	ret void
}

define internal void @"\01-[DVIconAndTextCell selectWithFrame:inView:editor:delegate:start:length:]"(%3* %self, %struct.objc_selector* %_cmd, %struct.CGRect* byval %frame, %8* %controlView, %9* %text, %struct.objc_object* %delegate, i32 %start, i32 %length) nounwind {
entry:
	%self.addr = alloca %3*		; <%3**> [#uses=4]
	%_cmd.addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=2]
	%controlView.addr = alloca %8*		; <%8**> [#uses=3]
	%text.addr = alloca %9*		; <%9**> [#uses=3]
	%delegate.addr = alloca %struct.objc_object*		; <%struct.objc_object**> [#uses=3]
	%start.addr = alloca i32		; <i32*> [#uses=3]
	%length.addr = alloca i32		; <i32*> [#uses=3]
	%agg.tmp = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%tmp5 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	%agg.tmp9 = alloca %struct.CGRect		; <%struct.CGRect*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*))
	store %3* %self, %3** %self.addr
	%0 = bitcast %3** %self.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable930 to { }*))
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd.addr
	%1 = bitcast %struct.objc_selector** %_cmd.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable931 to { }*))
	%2 = bitcast %struct.CGRect* %frame to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable933 to { }*))
	store %8* %controlView, %8** %controlView.addr
	%3 = bitcast %8** %controlView.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable934 to { }*))
	store %9* %text, %9** %text.addr
	%4 = bitcast %9** %text.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %4, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable944 to { }*))
	store %struct.objc_object* %delegate, %struct.objc_object** %delegate.addr
	%5 = bitcast %struct.objc_object** %delegate.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable946 to { }*))
	store i32 %start, i32* %start.addr
	%6 = bitcast i32* %start.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %6, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable948 to { }*))
	store i32 %length, i32* %length.addr
	%7 = bitcast i32* %length.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %7, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable950 to { }*))
	call void @llvm.dbg.stoppoint(i32 156, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp1 = bitcast %struct.CGRect* %agg.tmp to i8*		; <i8*> [#uses=1]
	%tmp2 = bitcast %struct.CGRect* %frame to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp1, i8* %tmp2, i32 32, i32 4)
	%tmp3 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_952"		; <%struct.objc_selector*> [#uses=1]
	%tmp4 = bitcast %3* %tmp to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	call void bitcast (void (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend_stret to void (%struct.CGRect*, %struct.objc_object*, %struct.objc_selector*, %struct.CGRect*)*)(%struct.CGRect* noalias sret %tmp5, %struct.objc_object* %tmp4, %struct.objc_selector* %tmp3, %struct.CGRect* byval %agg.tmp)
	%tmp6 = bitcast %struct.CGRect* %frame to i8*		; <i8*> [#uses=1]
	%tmp7 = bitcast %struct.CGRect* %tmp5 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp6, i8* %tmp7, i32 32, i32 4)
	call void @llvm.dbg.stoppoint(i32 157, i32 5, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%self8 = load %3** %self.addr		; <%3*> [#uses=1]
	%tmp10 = bitcast %struct.CGRect* %agg.tmp9 to i8*		; <i8*> [#uses=1]
	%tmp11 = bitcast %struct.CGRect* %frame to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* %tmp10, i8* %tmp11, i32 32, i32 4)
	%tmp12 = load %8** %controlView.addr		; <%8*> [#uses=1]
	%tmp13 = load %9** %text.addr		; <%9*> [#uses=1]
	%tmp14 = load %struct.objc_object** %delegate.addr		; <%struct.objc_object*> [#uses=1]
	%tmp15 = load i32* %start.addr		; <i32> [#uses=1]
	%tmp16 = load i32* %length.addr		; <i32> [#uses=1]
	%objc_super = alloca %struct._objc_super		; <%struct._objc_super*> [#uses=3]
	%8 = bitcast %3* %self8 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%9 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 0		; <%struct.objc_object**> [#uses=1]
	store %struct.objc_object* %8, %struct.objc_object** %9
	%tmp17 = load %struct._objc_class** @"\01L_OBJC_CLASS_REFERENCES_"		; <%struct._objc_class*> [#uses=1]
	%10 = bitcast %struct._objc_class* %tmp17 to %struct.objc_class*		; <%struct.objc_class*> [#uses=1]
	%11 = getelementptr %struct._objc_super* %objc_super, i32 0, i32 1		; <%struct.objc_class**> [#uses=1]
	store %struct.objc_class* %10, %struct.objc_class** %11
	%tmp18 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_954"		; <%struct.objc_selector*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct._objc_super*, %struct.objc_selector*, ...)* @objc_msgSendSuper to void (%struct._objc_super*, %struct.objc_selector*, %struct.CGRect*, %8*, %9*, %struct.objc_object*, i32, i32)*)(%struct._objc_super* %objc_super, %struct.objc_selector* %tmp18, %struct.CGRect* byval %agg.tmp9, %8* %tmp12, %9* %tmp13, %struct.objc_object* %tmp14, i32 %tmp15, i32 %tmp16)
	call void @llvm.dbg.stoppoint(i32 158, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram929 to { }*))
	ret void
}

declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind
