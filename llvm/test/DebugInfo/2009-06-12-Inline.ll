; RUN: llvm-as < %s | llc -f -o /dev/null 
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
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
	%struct.objc_object = type opaque
	%struct.objc_selector = type opaque
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__OBJC, __image_info,regular"		; <[2 x i32]*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [4 x i8] c"t.m\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [20 x i8] c"/Volumes/work/Radar\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@.str2 = internal constant [10 x i8] c"clang 1.0\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 16, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str3 = internal constant [3 x i8] c"f1\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([3 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([3 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([3 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* null, i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str4 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str5 = internal constant [3 x i8] c"l0\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([3 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 5, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str6 = internal constant [3 x i8] c"f0\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.subprogram7 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([3 x i8]* @.str6, i32 0, i32 0), i8* getelementptr ([3 x i8]* @.str6, i32 0, i32 0), i8* getelementptr ([3 x i8]* @.str6, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, { }* null, i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str8 = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable9 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to { }*), i8* getelementptr ([2 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 1		; <[1 x i8]*> [#uses=1]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module { i32 7, i32 16, i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct._objc_symtab* null }, section "__OBJC,__module_info,regular,no_dead_strip", align 4		; <%struct._objc_module*> [#uses=1]
@llvm.used = appending global [3 x i8*] [i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*)], section "llvm.metadata"		; <[3 x i8*]*> [#uses=0]

define void @f1() nounwind {
entry:
	%x.addr.i = alloca i32		; <i32*> [#uses=2]
	%l0 = alloca void (i32)*, align 4		; <void (i32)**> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	call void @llvm.dbg.stoppoint(i32 4, i32 3, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to { }*))
	store i32 1, i32* %x.addr.i
	%0 = bitcast i32* %x.addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable9 to { }*))
	call void @llvm.dbg.stoppoint(i32 2, i32 66, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.stoppoint(i32 5, i32 3, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to { }*))
	%1 = bitcast void (i32)** %l0 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	store void (i32)* @f0, void (i32)** %l0
	call void @llvm.dbg.stoppoint(i32 6, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret void
}

declare void @llvm.dbg.func.start({ }*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind readnone

define internal void @f0(i32 %x) nounwind alwaysinline {
entry:
	%x.addr = alloca i32		; <i32*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to { }*))
	store i32 %x, i32* %x.addr
	%0 = bitcast i32* %x.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable9 to { }*))
	call void @llvm.dbg.stoppoint(i32 2, i32 66, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram7 to { }*))
	ret void
}

declare void @llvm.dbg.declare({ }*, { }*) nounwind readnone

declare void @llvm.dbg.region.end({ }*) nounwind readnone
