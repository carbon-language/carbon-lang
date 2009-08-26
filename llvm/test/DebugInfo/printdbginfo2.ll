; RUN: llvm-as < %s | opt -print-dbginfo -disable-output > %t1
; RUN: grep {%b is variable b of type x declared at x.c:7} %t1
; RUN: grep {%2 is variable b of type x declared at x.c:7} %t1
; RUN: grep {@c.1442 is variable c of type int declared at x.c:4} %t1
	type { }		; type %0
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, %0*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0*, %0*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0* }
	%llvm.dbg.global_variable.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1, %0* }
	%llvm.dbg.subprogram.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1 }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%llvm.dbg.variable.type = type { i32, %0*, i8*, %0*, i32, %0* }
	%struct..0x = type { i32 }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [4 x i8] c"x.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [27 x i8] c"/home/edwin/llvm-svn/llvm/\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5641) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str4 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str4, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 2, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str5 = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@.str7 = internal constant [2 x i8] c"a\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*), i8* getelementptr ([2 x i8]* @.str7, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 6, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array8 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite9 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*), i8* getelementptr ([2 x i8]* @.str5, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 5, i64 32, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array8 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str10 = internal constant [2 x i8] c"b\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*), i8* getelementptr ([2 x i8]* @.str10, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 7, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite9 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subrange = internal constant %llvm.dbg.subrange.type { i32 458785, i64 0, i64 3 }, section "llvm.metadata"		; <%llvm.dbg.subrange.type*> [#uses=1]
@llvm.dbg.array11 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.subrange.type* @llvm.dbg.subrange to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite12 = internal constant %llvm.dbg.composite.type { i32 458753, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 128, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*), %0* bitcast ([1 x %0*]* @llvm.dbg.array11 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.variable13 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*), i8* getelementptr ([2 x i8]* @.str7, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 3, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite12 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@c.1442 = internal global i32 5		; <i32*> [#uses=2]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str14 = internal constant [7 x i8] c"c.1442\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str15 = internal constant [2 x i8] c"c\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type { i32 458804, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str14, i32 0, i32 0), i8* getelementptr ([2 x i8]* @.str15, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 4, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*), i1 true, i1 true, %0* bitcast (i32* @c.1442 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]

define i32 @main() nounwind {
entry:
	%b = alloca %struct..0x		; <%struct..0x*> [#uses=2]
	%a = alloca [4 x i32]		; <[4 x i32]*> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*))
	%0 = bitcast %struct..0x* %b to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to %0*))
	%1 = bitcast [4 x i32]* %a to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %1, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable13 to %0*))
	call void @llvm.dbg.stoppoint(i32 8, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%2 = getelementptr %struct..0x* %b, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 5, i32* %2, align 4
	call void @llvm.dbg.stoppoint(i32 9, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = load i32* @c.1442, align 4		; <i32> [#uses=1]
	br label %return

return:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 9, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*))
	ret i32 %3
}

declare void @llvm.dbg.func.start(%0*) nounwind readnone

declare void @llvm.dbg.declare(%0*, %0*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, %0*) nounwind readnone

declare void @llvm.dbg.region.end(%0*) nounwind readnone
