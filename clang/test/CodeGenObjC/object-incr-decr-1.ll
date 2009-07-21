; ModuleID = '/Users/argiris/proj/llvm/tools/clang/test/CodeGenObjC/object-incr-decr-1.m'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
module asm ""
module asm ""
	type opaque		; type %0
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
	%struct._objc_super = type <{ i8*, i8* }>
	%struct._objc_symtab = type { i32, %struct.objc_selector*, i16, i16, [0 x i8*] }
	%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
	%struct._prop_t = type { i8*, i8* }
	%struct.objc_selector = type opaque
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__OBJC, __image_info,regular"		; <[2 x i32]*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 1		; <[1 x i8]*> [#uses=1]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module { i32 7, i32 16, i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct._objc_symtab* null }, section "__OBJC,__module_info,regular,no_dead_strip", align 4		; <%struct._objc_module*> [#uses=1]
@llvm.used = appending global [3 x i8*] [i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*)], section "llvm.metadata"		; <[3 x i8*]*> [#uses=0]

define %0* @foo() nounwind {
entry:
	%retval = alloca %0*		; <%0**> [#uses=2]
	%f = alloca %0*, align 4		; <%0**> [#uses=9]
	%tmp = load %0** %f		; <%0*> [#uses=1]
	%0 = bitcast %0* %tmp to i8*		; <i8*> [#uses=1]
	%add.ptr = getelementptr i8* %0, i32 24		; <i8*> [#uses=1]
	%1 = bitcast %0** %f to i8**		; <i8**> [#uses=1]
	store i8* %add.ptr, i8** %1
	%tmp1 = load %0** %f		; <%0*> [#uses=1]
	%2 = bitcast %0* %tmp1 to i8*		; <i8*> [#uses=1]
	%add.ptr2 = getelementptr i8* %2, i32 -24		; <i8*> [#uses=1]
	%3 = bitcast %0** %f to i8**		; <i8**> [#uses=1]
	store i8* %add.ptr2, i8** %3
	%tmp3 = load %0** %f		; <%0*> [#uses=1]
	%4 = bitcast %0* %tmp3 to i8*		; <i8*> [#uses=1]
	%add.ptr4 = getelementptr i8* %4, i32 -24		; <i8*> [#uses=1]
	%5 = bitcast %0** %f to i8**		; <i8**> [#uses=1]
	store i8* %add.ptr4, i8** %5
	%tmp5 = load %0** %f		; <%0*> [#uses=1]
	%6 = bitcast %0* %tmp5 to i8*		; <i8*> [#uses=1]
	%add.ptr6 = getelementptr i8* %6, i32 24		; <i8*> [#uses=1]
	%7 = bitcast %0** %f to i8**		; <i8**> [#uses=1]
	store i8* %add.ptr6, i8** %7
	%tmp7 = load %0** %f		; <%0*> [#uses=1]
	store %0* %tmp7, %0** %retval
	%8 = load %0** %retval		; <%0*> [#uses=1]
	ret %0* %8
}
