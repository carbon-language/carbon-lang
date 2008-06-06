; RUN: llvm-as < %s | llc -march=c | grep llvm_cbe_asmname | count 36
; PR2418

; ModuleID = 'main.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
module asm "\09.lazy_reference .objc_class_name_MyClass"
module asm "\09.lazy_reference .objc_class_name_Object"
	%struct.MyClass = type { %struct.Object }
	%struct.Object = type { %struct.objc_class* }
	%struct.Protocol = type opaque
	%struct._objc__method_prototype_list = type opaque
	%struct._objc_class = type { %struct._objc_class*, %struct._objc_class*, i8*, i32, i32, i32, %struct._objc_ivar_list*, %struct._objc_method_list*, %struct.objc_cache*, %struct._objc_protocol**, i8*, %struct._objc_class_ext* }
	%struct._objc_class_ext = type opaque
	%struct._objc_ivar_list = type opaque
	%struct._objc_method = type { %struct.objc_selector*, i8*, i8* }
	%struct._objc_method_list = type opaque
	%struct._objc_module = type { i32, i32, i8*, %struct._objc_symtab* }
	%struct._objc_protocol = type { %struct._objc_protocol_extension*, i8*, %struct._objc_protocol**, %struct._objc__method_prototype_list*, %struct._objc__method_prototype_list* }
	%struct._objc_protocol_extension = type opaque
	%struct._objc_symtab = type { i32, %struct.objc_selector**, i16, i16, [1 x i8*] }
	%struct.anon = type { %struct._objc__method_prototype_list*, i32, [1 x %struct._objc_method] }
	%struct.objc_cache = type { i32, i32, [1 x %struct.objc_method*] }
	%struct.objc_class = type { %struct.objc_class*, %struct.objc_class*, i8*, i32, i32, i32, %struct.objc_ivar_list*, %struct.objc_method_list**, %struct.objc_cache*, %struct.objc_protocol_list* }
	%struct.objc_ivar = type { i8*, i8*, i32 }
	%struct.objc_ivar_list = type { i32, [1 x %struct.objc_ivar] }
	%struct.objc_method = type { %struct.objc_selector*, i8*, %struct.Object* (%struct.Object*, %struct.objc_selector*, ...)* }
	%struct.objc_method_list = type { %struct.objc_method_list*, i32, [1 x %struct.objc_method] }
	%struct.objc_object = type { %struct.objc_class* }
	%struct.objc_protocol_list = type { %struct.objc_protocol_list*, i32, [1 x %struct.Protocol*] }
	%struct.objc_selector = type opaque
@.str = internal constant [13 x i8] c"Hello world!\00"		; <[13 x i8]*> [#uses=1]
@"\01L_OBJC_CLASS_REFERENCES_0" = internal global %struct.objc_class* bitcast ([8 x i8]* @"\01L_OBJC_CLASS_NAME_0" to %struct.objc_class*), section "__OBJC,__cls_refs,literal_pointers,no_dead_strip"		; <%struct.objc_class**> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_0" = internal global %struct.objc_selector* bitcast ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_1" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip"		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_SELECTOR_REFERENCES_1" = internal global %struct.objc_selector* bitcast ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_0" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip"		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_CLASS_MyClass" = internal global %struct._objc_class {
    %struct._objc_class* @"\01L_OBJC_METACLASS_MyClass", 
    %struct._objc_class* bitcast ([7 x i8]* @"\01L_OBJC_CLASS_NAME_1" to %struct._objc_class*), 
    i8* getelementptr ([8 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), 
    i32 0, 
    i32 1, 
    i32 4, 
    %struct._objc_ivar_list* null, 
    %struct._objc_method_list* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01L_OBJC_INSTANCE_METHODS_MyClass" to %struct._objc_method_list*), 
    %struct.objc_cache* null, 
    %struct._objc_protocol** null, 
    i8* null, 
    %struct._objc_class_ext* null }, section "__OBJC,__class,regular,no_dead_strip", align 32		; <%struct._objc_class*> [#uses=2]
@"\01L_OBJC_SYMBOLS" = internal global { i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] } {
    i32 0, 
    %struct.objc_selector** null, 
    i16 1, 
    i16 0, 
    [1 x %struct._objc_class*] [ %struct._objc_class* @"\01L_OBJC_CLASS_MyClass" ] }, section "__OBJC,__symbols,regular,no_dead_strip"		; <{ i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] }*> [#uses=2]
@L_OBJC_METH_VAR_NAME_0 = internal global [9 x i8] c"sayHello\00", section "__TEXT,__cstring,cstring_literals"		; <[9 x i8]*> [#uses=0]
@L_OBJC_METH_VAR_TYPE_0 = internal global [7 x i8] c"v8@0:4\00", section "__TEXT,__cstring,cstring_literals"		; <[7 x i8]*> [#uses=0]
@"\01L_OBJC_METH_VAR_NAME_0" = internal global [9 x i8] c"sayHello\00", section "__TEXT,__cstring,cstring_literals"		; <[9 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_0" = internal global [7 x i8] c"v8@0:4\00", section "__TEXT,__cstring,cstring_literals"		; <[7 x i8]*> [#uses=1]
@"\01L_OBJC_INSTANCE_METHODS_MyClass" = internal global { i32, i32, [1 x %struct._objc_method] } {
    i32 0, 
    i32 1, 
    [1 x %struct._objc_method] [ %struct._objc_method {
        %struct.objc_selector* bitcast ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_0" to %struct.objc_selector*), 
        i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_0", i32 0, i32 0), 
        i8* bitcast (void (%struct.MyClass*, %struct.objc_selector*)* @"-[MyClass sayHello]" to i8*) } ] }, section "__OBJC,__inst_meth,regular,no_dead_strip"		; <{ i32, i32, [1 x %struct._objc_method] }*> [#uses=2]
@L_OBJC_CLASS_NAME_0 = internal global [8 x i8] c"MyClass\00", section "__TEXT,__cstring,cstring_literals"		; <[8 x i8]*> [#uses=0]
@L_OBJC_CLASS_NAME_1 = internal global [7 x i8] c"Object\00", section "__TEXT,__cstring,cstring_literals"		; <[7 x i8]*> [#uses=0]
@"\01L_OBJC_METACLASS_MyClass" = internal global %struct._objc_class {
    %struct._objc_class* bitcast ([7 x i8]* @"\01L_OBJC_CLASS_NAME_1" to %struct._objc_class*), 
    %struct._objc_class* bitcast ([7 x i8]* @"\01L_OBJC_CLASS_NAME_1" to %struct._objc_class*), 
    i8* getelementptr ([8 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), 
    i32 0, 
    i32 2, 
    i32 48, 
    %struct._objc_ivar_list* null, 
    %struct._objc_method_list* null, 
    %struct.objc_cache* null, 
    %struct._objc_protocol** null, 
    i8* null, 
    %struct._objc_class_ext* null }, section "__OBJC,__meta_class,regular,no_dead_strip", align 32		; <%struct._objc_class*> [#uses=2]
@"\01L_OBJC_CLASS_NAME_1" = internal global [7 x i8] c"Object\00", section "__TEXT,__cstring,cstring_literals"		; <[7 x i8]*> [#uses=2]
@"\01L_OBJC_CLASS_NAME_0" = internal global [8 x i8] c"MyClass\00", section "__TEXT,__cstring,cstring_literals"		; <[8 x i8]*> [#uses=2]
@L_OBJC_METH_VAR_NAME_1 = internal global [6 x i8] c"alloc\00", section "__TEXT,__cstring,cstring_literals"		; <[6 x i8]*> [#uses=0]
@"\01L_OBJC_METH_VAR_NAME_1" = internal global [6 x i8] c"alloc\00", section "__TEXT,__cstring,cstring_literals"		; <[6 x i8]*> [#uses=2]
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] zeroinitializer, section "__OBJC,__image_info,regular"		; <[2 x i32]*> [#uses=1]
@L_OBJC_CLASS_NAME_2 = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals"		; <[1 x i8]*> [#uses=0]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module {
    i32 7, 
    i32 16, 
    i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_2", i32 0, i32 0), 
    %struct._objc_symtab* bitcast ({ i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] }* @"\01L_OBJC_SYMBOLS" to %struct._objc_symtab*) }, section "__OBJC,__module_info,regular,no_dead_strip"		; <%struct._objc_module*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_2" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals"		; <[1 x i8]*> [#uses=1]
@"\01.objc_class_name_MyClass" = constant i32 0		; <i32*> [#uses=1]
@llvm.used = appending global [16 x i8*] [ i8* bitcast (%struct.objc_class** @"\01L_OBJC_CLASS_REFERENCES_0" to i8*), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_0" to i8*), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_1" to i8*), i8* bitcast (%struct._objc_class* @"\01L_OBJC_CLASS_MyClass" to i8*), i8* bitcast ({ i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] }* @"\01L_OBJC_SYMBOLS" to i8*), i8* getelementptr ([9 x i8]* @"\01L_OBJC_METH_VAR_NAME_0", i32 0, i32 0), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_0", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01L_OBJC_INSTANCE_METHODS_MyClass" to i8*), i8* getelementptr ([7 x i8]* @"\01L_OBJC_CLASS_NAME_1", i32 0, i32 0), i8* getelementptr ([8 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), i8* bitcast (%struct._objc_class* @"\01L_OBJC_METACLASS_MyClass" to i8*), i8* getelementptr ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0), i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_2", i32 0, i32 0), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*), i8* bitcast (i32* @"\01.objc_class_name_MyClass" to i8*) ], section "llvm.metadata"		; <[16 x i8*]*> [#uses=0]

define internal void @"-[MyClass sayHello]"(%struct.MyClass* %self, %struct.objc_selector* %_cmd) {
entry:
	%self_addr = alloca %struct.MyClass*		; <%struct.MyClass**> [#uses=1]
	%_cmd_addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store %struct.MyClass* %self, %struct.MyClass** %self_addr
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd_addr
	%tmp = getelementptr [13 x i8]* @.str, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp1 = call i32 @puts( i8* %tmp ) nounwind 		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare i32 @puts(i8*)

define i32 @main() {
entry:
	%retval = alloca i32		; <i32*> [#uses=1]
	%anObject = alloca %struct.MyClass*		; <%struct.MyClass**> [#uses=2]
	%anObject.4 = alloca %struct.Object*		; <%struct.Object**> [#uses=2]
	%L_OBJC_CLASS_REFERENCES_0.2 = alloca %struct.Object*		; <%struct.Object**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load %struct.objc_class** @"\01L_OBJC_CLASS_REFERENCES_0", align 4		; <%struct.objc_class*> [#uses=1]
	%tmp1 = bitcast %struct.objc_class* %tmp to %struct.Object*		; <%struct.Object*> [#uses=1]
	store %struct.Object* %tmp1, %struct.Object** %L_OBJC_CLASS_REFERENCES_0.2, align 4
	%tmp2 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_0", align 4		; <%struct.objc_selector*> [#uses=1]
	%tmp3 = load %struct.Object** %L_OBJC_CLASS_REFERENCES_0.2, align 4		; <%struct.Object*> [#uses=1]
	%tmp4 = call %struct.Object* bitcast (%struct.Object* (%struct.Object*, %struct.objc_selector*, ...)* @objc_msgSend to %struct.Object* (%struct.Object*, %struct.objc_selector*)*)( %struct.Object* %tmp3, %struct.objc_selector* %tmp2 ) nounwind 		; <%struct.Object*> [#uses=1]
	%tmp45 = bitcast %struct.Object* %tmp4 to %struct.MyClass*		; <%struct.MyClass*> [#uses=1]
	store %struct.MyClass* %tmp45, %struct.MyClass** %anObject, align 4
	%tmp6 = load %struct.MyClass** %anObject, align 4		; <%struct.MyClass*> [#uses=1]
	%tmp67 = bitcast %struct.MyClass* %tmp6 to %struct.Object*		; <%struct.Object*> [#uses=1]
	store %struct.Object* %tmp67, %struct.Object** %anObject.4, align 4
	%tmp8 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_1", align 4		; <%struct.objc_selector*> [#uses=1]
	%tmp9 = load %struct.Object** %anObject.4, align 4		; <%struct.Object*> [#uses=1]
	call void bitcast (%struct.Object* (%struct.Object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.Object*, %struct.objc_selector*)*)( %struct.Object* %tmp9, %struct.objc_selector* %tmp8 ) nounwind 
	br label %return

return:		; preds = %entry
	%retval10 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval10
}

declare %struct.Object* @objc_msgSend(%struct.Object*, %struct.objc_selector*, ...)
