; RUN: llc %s -o - -asm-verbose -O0 | not grep ".long	0x0	## DW_AT_abstract_origin"

	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.global_variable.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
	%struct.AAAAAImageParser = type { %struct.CObject* }
	%struct.CObject = type { i32 }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [9 x i8] c"tcase.cc\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str1 = internal constant [6 x i8] c"/tmp/\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str2 = internal constant [55 x i8] c"4.2.1 (Based on Apple Inc. build 5646) (LLVM build 00)\00", section "llvm.metadata"		; <[55 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([9 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([55 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [8 x i8] c"tcase.h\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.compile_unit4 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([55 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str5 = internal constant [8 x i8] c"CObject\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str6 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str6, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str7 = internal constant [2 x i8] c"d\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str7, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 6, i64 32, i64 32, i64 0, i32 1, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype8 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype8 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite9 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str10 = internal constant [4 x i8] c"set\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str11 = internal constant [18 x i8] c"_ZN7CObject3setEi\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str10, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str10, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str11, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 3, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite9 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array12 = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype8 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite13 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array12 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str14 = internal constant [8 x i8] c"release\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str15 = internal constant [22 x i8] c"_ZN7CObject7releaseEv\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram16 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str14, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str14, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str15, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 4, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite13 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array17 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite18 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 1, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array17 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype19 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array20 = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array20 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram21 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str14, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str14, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str15, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 4, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype22 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str23 = internal constant [5 x i8] c"this\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram21 to { }*), i8* getelementptr ([5 x i8]* @.str23, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 4, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype22 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.array24 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite25 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array24 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str26 = internal constant [14 x i8] c"ReleaseObject\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str27 = internal constant [27 x i8] c"_Z13ReleaseObjectP7CObject\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram28 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str26, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str26, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str27, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 10, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite25 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str29 = internal constant [7 x i8] c"object\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.variable30 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram28 to { }*), i8* getelementptr ([7 x i8]* @.str29, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 10, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype22 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str31 = internal constant [17 x i8] c"AAAAAImageParser\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@.str33 = internal constant [13 x i8] c"mCustomWhite\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype34 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str33, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 21, i64 32, i64 32, i64 0, i32 1, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype35 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite45 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array36 = internal constant [3 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype35 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite37 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array36 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str38 = internal constant [18 x i8] c"~AAAAAImageParser\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram39 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str38, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 24, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite37 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array40 = internal constant [3 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype35 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite41 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array40 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str42 = internal constant [36 x i8] c"_ZN16AAAAAImageParser3setEP7CObject\00", section "llvm.metadata"		; <[36 x i8]*> [#uses=1]
@llvm.dbg.subprogram43 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str10, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str10, i32 0, i32 0), i8* getelementptr ([36 x i8]* @.str42, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 19, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite41 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array44 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype34 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram39 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram43 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite45 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str31, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 16, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array44 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype46 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite45 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array47 = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype46 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite48 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array47 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str49 = internal constant [26 x i8] c"_ZN16AAAAAImageParserD2Ev\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram50 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str49, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 24, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite48 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype51 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype46 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.variable52 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram50 to { }*), i8* getelementptr ([5 x i8]* @.str23, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 24, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype51 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str53 = internal constant [26 x i8] c"_ZN16AAAAAImageParserD1Ev\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram54 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str38, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str53, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 24, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite48 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable55 = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram54 to { }*), i8* getelementptr ([5 x i8]* @.str23, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*), i32 24, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype51 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.array56 = internal constant [1 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*)], section "llvm.metadata"		; <[1 x { }*]*> [#uses=1]
@llvm.dbg.composite57 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([1 x { }*]* @llvm.dbg.array56 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str58 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram59 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str58, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str58, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str58, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite57 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str60 = internal constant [2 x i8] c"C\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable61 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram59 to { }*), i8* getelementptr ([2 x i8]* @.str60, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype46 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@_ZZ4mainE3C.0 = private constant %struct.AAAAAImageParser zeroinitializer		; <%struct.AAAAAImageParser*> [#uses=2]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str62 = internal constant [14 x i8] c"_ZZ4mainE3C.0\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str63 = internal constant [4 x i8] c"C.0\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type { i32 458804, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str62, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str63, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str62, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite45 to { }*), i1 false, i1 true, { }* bitcast (%struct.AAAAAImageParser* @_ZZ4mainE3C.0 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]

define void @_ZN16AAAAAImageParserD2Ev(%struct.AAAAAImageParser* %this) nounwind ssp {
entry:
	%object_addr.i = alloca %struct.CObject*		; <%struct.CObject**> [#uses=4]
	%retval.i = alloca %struct.CObject*		; <%struct.CObject**> [#uses=2]
	%0 = alloca %struct.CObject*		; <%struct.CObject**> [#uses=2]
	%this_addr = alloca %struct.AAAAAImageParser*		; <%struct.AAAAAImageParser**> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram50 to { }*))
	%1 = bitcast %struct.AAAAAImageParser** %this_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable52 to { }*))
	store %struct.AAAAAImageParser* %this, %struct.AAAAAImageParser** %this_addr
	call void @llvm.dbg.stoppoint(i32 26, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	%2 = load %struct.AAAAAImageParser** %this_addr, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	%3 = getelementptr %struct.AAAAAImageParser* %2, i32 0, i32 0		; <%struct.CObject**> [#uses=1]
	%4 = load %struct.CObject** %3, align 4		; <%struct.CObject*> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram28 to { }*)) nounwind
	%5 = bitcast %struct.CObject** %object_addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable30 to { }*)) nounwind
	store %struct.CObject* %4, %struct.CObject** %object_addr.i
	call void @llvm.dbg.stoppoint(i32 11, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	%6 = load %struct.CObject** %object_addr.i, align 4		; <%struct.CObject*> [#uses=1]
	%7 = icmp ne %struct.CObject* %6, null		; <i1> [#uses=1]
	br i1 %7, label %bb.i, label %_Z13ReleaseObjectP7CObject.exit

bb.i:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 12, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	%8 = load %struct.CObject** %object_addr.i, align 4		; <%struct.CObject*> [#uses=1]
	call void @_ZN7CObject7releaseEv(%struct.CObject* %8) nounwind
	br label %_Z13ReleaseObjectP7CObject.exit

_Z13ReleaseObjectP7CObject.exit:		; preds = %bb.i, %entry
	call void @llvm.dbg.stoppoint(i32 13, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	store %struct.CObject* null, %struct.CObject** %0, align 4
	%9 = load %struct.CObject** %0, align 4		; <%struct.CObject*> [#uses=1]
	store %struct.CObject* %9, %struct.CObject** %retval.i, align 4
	%retval2.i = load %struct.CObject** %retval.i		; <%struct.CObject*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 13, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram28 to { }*))
	br label %bb

bb:		; preds = %_Z13ReleaseObjectP7CObject.exit
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	br label %return

return:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram50 to { }*))
	ret void
}

define linkonce_odr void @_ZN7CObject7releaseEv(%struct.CObject* %this) nounwind ssp {
entry:
	%this_addr = alloca %struct.CObject*		; <%struct.CObject**> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram21 to { }*))
	%0 = bitcast %struct.CObject** %this_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable to { }*))
	store %struct.CObject* %this, %struct.CObject** %this_addr
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	%1 = load %struct.CObject** %this_addr, align 4		; <%struct.CObject*> [#uses=1]
	%2 = getelementptr %struct.CObject* %1, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %2, align 4
	br label %return

return:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram21 to { }*))
	ret void
}

declare void @llvm.dbg.func.start({ }*) nounwind readnone

declare void @llvm.dbg.declare({ }*, { }*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind readnone

declare void @llvm.dbg.region.end({ }*) nounwind readnone

define void @_ZN16AAAAAImageParserD1Ev(%struct.AAAAAImageParser* %this) nounwind ssp {
entry:
	%object_addr.i = alloca %struct.CObject*		; <%struct.CObject**> [#uses=4]
	%retval.i = alloca %struct.CObject*		; <%struct.CObject**> [#uses=2]
	%0 = alloca %struct.CObject*		; <%struct.CObject**> [#uses=2]
	%this_addr = alloca %struct.AAAAAImageParser*		; <%struct.AAAAAImageParser**> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram54 to { }*))
	%1 = bitcast %struct.AAAAAImageParser** %this_addr to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %1, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable55 to { }*))
	store %struct.AAAAAImageParser* %this, %struct.AAAAAImageParser** %this_addr
	call void @llvm.dbg.stoppoint(i32 26, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	%2 = load %struct.AAAAAImageParser** %this_addr, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	%3 = getelementptr %struct.AAAAAImageParser* %2, i32 0, i32 0		; <%struct.CObject**> [#uses=1]
	%4 = load %struct.CObject** %3, align 4		; <%struct.CObject*> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram28 to { }*)) nounwind
	%5 = bitcast %struct.CObject** %object_addr.i to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %5, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable30 to { }*)) nounwind
	store %struct.CObject* %4, %struct.CObject** %object_addr.i
	call void @llvm.dbg.stoppoint(i32 11, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	%6 = load %struct.CObject** %object_addr.i, align 4		; <%struct.CObject*> [#uses=1]
	%7 = icmp ne %struct.CObject* %6, null		; <i1> [#uses=1]
	br i1 %7, label %bb.i, label %_Z13ReleaseObjectP7CObject.exit

bb.i:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 12, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	%8 = load %struct.CObject** %object_addr.i, align 4		; <%struct.CObject*> [#uses=1]
	call void @_ZN7CObject7releaseEv(%struct.CObject* %8) nounwind
	br label %_Z13ReleaseObjectP7CObject.exit

_Z13ReleaseObjectP7CObject.exit:		; preds = %bb.i, %entry
	call void @llvm.dbg.stoppoint(i32 13, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	store %struct.CObject* null, %struct.CObject** %0, align 4
	%9 = load %struct.CObject** %0, align 4		; <%struct.CObject*> [#uses=1]
	store %struct.CObject* %9, %struct.CObject** %retval.i, align 4
	%retval2.i = load %struct.CObject** %retval.i		; <%struct.CObject*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 13, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram28 to { }*))
	br label %bb

bb:		; preds = %_Z13ReleaseObjectP7CObject.exit
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	br label %return

return:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 27, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit4 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram54 to { }*))
	ret void
}

define i32 @main() ssp {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%C = alloca %struct.AAAAAImageParser*		; <%struct.AAAAAImageParser**> [#uses=3]
	%0 = alloca i32		; <i32*> [#uses=2]
	%C.1 = alloca %struct.AAAAAImageParser*		; <%struct.AAAAAImageParser**> [#uses=4]
	%1 = alloca %struct.AAAAAImageParser*		; <%struct.AAAAAImageParser**> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram59 to { }*))
	%2 = bitcast %struct.AAAAAImageParser** %C to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %2, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable61 to { }*))
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = call i8* @_Znwm(i32 4)		; <i8*> [#uses=1]
	%4 = bitcast i8* %3 to %struct.AAAAAImageParser*		; <%struct.AAAAAImageParser*> [#uses=1]
	store %struct.AAAAAImageParser* %4, %struct.AAAAAImageParser** %1, align 4
	%5 = load %struct.AAAAAImageParser** %1, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	%6 = getelementptr %struct.AAAAAImageParser* %5, i32 0, i32 0		; <%struct.CObject**> [#uses=1]
	%7 = load %struct.CObject** getelementptr (%struct.AAAAAImageParser* @_ZZ4mainE3C.0, i32 0, i32 0), align 4		; <%struct.CObject*> [#uses=1]
	store %struct.CObject* %7, %struct.CObject** %6, align 4
	%8 = load %struct.AAAAAImageParser** %1, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	store %struct.AAAAAImageParser* %8, %struct.AAAAAImageParser** %C, align 4
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%9 = load %struct.AAAAAImageParser** %C, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	store %struct.AAAAAImageParser* %9, %struct.AAAAAImageParser** %C.1, align 4
	%10 = load %struct.AAAAAImageParser** %C.1, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	%11 = icmp ne %struct.AAAAAImageParser* %10, null		; <i1> [#uses=1]
	br i1 %11, label %bb, label %bb1
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb

bb:		; preds = %12, %entry
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%13 = load %struct.AAAAAImageParser** %C.1, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	call void @_ZN16AAAAAImageParserD1Ev(%struct.AAAAAImageParser* %13) nounwind
	%14 = load %struct.AAAAAImageParser** %C.1, align 4		; <%struct.AAAAAImageParser*> [#uses=1]
	%15 = bitcast %struct.AAAAAImageParser* %14 to i8*		; <i8*> [#uses=1]
	call void @_ZdlPv(i8* %15) nounwind
	br label %bb1

bb1:		; preds = %bb, %entry
	call void @llvm.dbg.stoppoint(i32 6, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 0, i32* %0, align 4
	%16 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %16, i32* %retval, align 4
	br label %return

return:		; preds = %bb1
	%retval2 = load i32* %retval		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 6, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram59 to { }*))
	ret i32 %retval2
}

declare i8* @_Znwm(i32)

declare void @_ZdlPv(i8*) nounwind
