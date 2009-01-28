; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin | grep 0x49 | count 7
; Count number of DW_AT_Type attributes.
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32, i8*, i8* }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i8*, i8* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, i8*, i8* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, i8*, i8* }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }*, i8*, i8* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [4 x i8] c"a.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [26 x i8] c"/Volumes/Nanpura/dbg.test\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5, i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str4 = internal constant [5 x i8] c"char\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.basictype5 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 6, i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype5 to { }*), i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype6 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x { }*] [ { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype6 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array to { }*), i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str7 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str7, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 false, i1 true, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str8 = internal constant [5 x i8] c"argc\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([5 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str9 = internal constant [5 x i8] c"argv\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable10 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([5 x i8]* @.str9, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype6 to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subrange = internal constant %llvm.dbg.subrange.type { i32 458785, i64 0, i64 5 }, section "llvm.metadata"		; <%llvm.dbg.subrange.type*> [#uses=1]
@llvm.dbg.array11 = internal constant [1 x { }*] [ { }* bitcast (%llvm.dbg.subrange.type* @llvm.dbg.subrange to { }*) ], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite12 = internal constant %llvm.dbg.composite.type { i32 458753, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 48, i64 8, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype5 to { }*), { }* bitcast ([1 x { }*]* @llvm.dbg.array11 to { }*), i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str13 = internal constant [5 x i8] c"str1\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable14 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([5 x i8]* @.str13, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite12 to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@"\01LC" = internal constant [6 x i8] c"a.out\00"		; <[6 x i8]*> [#uses=6]

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	%argc_addr = alloca i32		; <i32*> [#uses=2]
	%argv_addr = alloca i8**		; <i8***> [#uses=2]
	%retval = alloca i32		; <i32*> [#uses=2]
	%str1 = alloca [6 x i8]		; <[6 x i8]*> [#uses=7]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%1 = bitcast i32* %argc_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	store i32 %argc, i32* %argc_addr
	%2 = bitcast i8*** %argv_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable10 to { }*))
	store i8** %argv, i8*** %argv_addr
	%3 = bitcast [6 x i8]* %str1 to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %3, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable14 to { }*))
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = getelementptr [6 x i8]* %str1, i32 0, i32 0		; <i8*> [#uses=1]
	%5 = load i8* getelementptr ([6 x i8]* @"\01LC", i32 0, i32 0), align 1		; <i8> [#uses=1]
	store i8 %5, i8* %4, align 1
	%6 = getelementptr [6 x i8]* %str1, i32 0, i32 1		; <i8*> [#uses=1]
	%7 = load i8* getelementptr ([6 x i8]* @"\01LC", i32 0, i32 1), align 1		; <i8> [#uses=1]
	store i8 %7, i8* %6, align 1
	%8 = getelementptr [6 x i8]* %str1, i32 0, i32 2		; <i8*> [#uses=1]
	%9 = load i8* getelementptr ([6 x i8]* @"\01LC", i32 0, i32 2), align 1		; <i8> [#uses=1]
	store i8 %9, i8* %8, align 1
	%10 = getelementptr [6 x i8]* %str1, i32 0, i32 3		; <i8*> [#uses=1]
	%11 = load i8* getelementptr ([6 x i8]* @"\01LC", i32 0, i32 3), align 1		; <i8> [#uses=1]
	store i8 %11, i8* %10, align 1
	%12 = getelementptr [6 x i8]* %str1, i32 0, i32 4		; <i8*> [#uses=1]
	%13 = load i8* getelementptr ([6 x i8]* @"\01LC", i32 0, i32 4), align 1		; <i8> [#uses=1]
	store i8 %13, i8* %12, align 1
	%14 = getelementptr [6 x i8]* %str1, i32 0, i32 5		; <i8*> [#uses=1]
	%15 = load i8* getelementptr ([6 x i8]* @"\01LC", i32 0, i32 5), align 1		; <i8> [#uses=1]
	store i8 %15, i8* %14, align 1
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 0, i32* %0, align 4
	%16 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %16, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret i32 %retval1
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind
