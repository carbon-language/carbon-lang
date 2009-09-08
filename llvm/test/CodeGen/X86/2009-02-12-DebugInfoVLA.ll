; RUN: llc < %s
; RUN: llc < %s -march=x86-64
; PR3538
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.block.type = type { i32, { }* }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [4 x i8] c"t.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [2 x i8] c".\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@.str2 = internal constant [6 x i8] c"clang\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([2 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str4 = internal constant [5 x i8] c"test\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str5 = internal constant [2 x i8] c"X\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([2 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.block = internal constant %llvm.dbg.block.type { i32 458763, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*) }, section "llvm.metadata"		; <%llvm.dbg.block.type*> [#uses=1]
@llvm.dbg.subrange = internal constant %llvm.dbg.subrange.type { i32 458785, i64 0, i64 0 }, section "llvm.metadata"		; <%llvm.dbg.subrange.type*> [#uses=1]
@llvm.dbg.array = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.subrange.type* @llvm.dbg.subrange to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458753, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 0, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast ([1 x { }*]* @llvm.dbg.array to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str6 = internal constant [2 x i8] c"Y\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable7 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.block.type* @llvm.dbg.block to { }*), i8* getelementptr ([2 x i8]* @.str6, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]

define i32 @test(i32 %X) nounwind {
entry:
	%retval = alloca i32		; <i32*> [#uses=1]
	%X.addr = alloca i32		; <i32*> [#uses=3]
	%saved_stack = alloca i8*		; <i8**> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	store i32 %X, i32* %X.addr
	%0 = bitcast i32* %X.addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	call void @llvm.dbg.region.start({ }* bitcast (%llvm.dbg.block.type* @llvm.dbg.block to { }*))
	call void @llvm.dbg.stoppoint(i32 4, i32 3, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%1 = call i8* @llvm.stacksave()		; <i8*> [#uses=1]
	store i8* %1, i8** %saved_stack
	%tmp = load i32* %X.addr		; <i32> [#uses=1]
	%2 = mul i32 4, %tmp		; <i32> [#uses=1]
	%vla = alloca i8, i32 %2		; <i8*> [#uses=1]
	%tmp1 = bitcast i8* %vla to i32*		; <i32*> [#uses=1]
	%3 = bitcast i32* %tmp1 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable7 to { }*))
	call void @llvm.dbg.stoppoint(i32 5, i32 1, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.block.type* @llvm.dbg.block to { }*))
	br label %cleanup

cleanup:		; preds = %entry
	%tmp2 = load i8** %saved_stack		; <i8*> [#uses=1]
	call void @llvm.stackrestore(i8* %tmp2)
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%4 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %4
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.region.start({ }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare i8* @llvm.stacksave() nounwind

declare void @llvm.stackrestore(i8*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind
