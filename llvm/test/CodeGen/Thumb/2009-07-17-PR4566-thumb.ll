; RUN: llvm-as < %s | llc -march=thumb | grep {rsbs \\+r\[0-9\]\\+, \\+r\[0-9\]\\+, \\+#0}

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv6-elf"
	type { i32 }		; type %0
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%struct.anon = type { %struct.dwarf_fde* }
	%struct.dwarf_cie = type <{ i32, i32, i8, [0 x i8], [3 x i8] }>
	%struct.dwarf_eh_bases = type { i8*, i8*, i8* }
	%struct.dwarf_fde = type <{ i32, i32, [0 x i8] }>
	%struct.fde_accumulator = type { %struct.fde_vector*, %struct.fde_vector* }
	%struct.fde_vector = type { i8*, i32, [0 x %struct.dwarf_fde*] }
	%struct.object = type { i8*, i8*, i8*, %struct.anon, %0, %struct.object* }
@.str = internal constant [17 x i8] c"unwind-dw2-fde.c\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@.str1 = internal constant [61 x i8] c"/home/asl/proj/llvm/llvm-gcc-4.2/build_arm/gcc/../../src/gcc\00", section "llvm.metadata"		; <[61 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5646) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = linkonce constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([17 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [14 x i8] c"unsigned char\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 8 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 8, i64 8, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype4 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str5 = internal constant [13 x i8] c"unsigned int\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.basictype6 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str7 = internal constant [9 x i8] c"unwind.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str8 = internal constant [57 x i8] c"/home/asl/proj/llvm/llvm-gcc-4.2/build_arm/gcc/./include\00", section "llvm.metadata"		; <[57 x i8]*> [#uses=1]
@llvm.dbg.compile_unit9 = linkonce constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([9 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([57 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str10 = internal constant [13 x i8] c"_Unwind_Word\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype11 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str10, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit9 to { }*), i32 47, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype12 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype4 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype4 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str13 = internal constant [12 x i8] c"unwind-pe.h\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.compile_unit14 = linkonce constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([12 x i8]* @.str13, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str15 = internal constant [13 x i8] c"read_uleb128\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str15, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str15, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str15, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i32 134, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str16 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype17 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str16, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str18 = internal constant [12 x i8] c"coretypes.h\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.compile_unit19 = linkonce constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([12 x i8]* @.str18, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str20 = internal constant [8 x i8] c"wchar_t\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype21 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str20, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit19 to { }*), i32 72, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str22 = internal constant [14 x i8] c"_Unwind_Sword\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype23 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str22, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit9 to { }*), i32 51, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype24 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype23 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array25 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype4 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype4 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype24 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite26 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array25 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str27 = internal constant [13 x i8] c"read_sleb128\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram28 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str27, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str27, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str27, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i32 156, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite26 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str29 = internal constant [17 x i8] c"unwind-dw2-fde.h\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.compile_unit30 = linkonce constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([17 x i8]* @.str29, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str31 = internal constant [10 x i8] c"dwarf_cie\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str33 = internal constant [12 x i8] c"_Unwind_Ptr\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype34 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str33, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit9 to { }*), i32 53, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str35 = internal constant [21 x i8] c"_Unwind_Internal_Ptr\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype36 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([21 x i8]* @.str35, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit9 to { }*), i32 59, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype34 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype37 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype36 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str38 = internal constant [6 x i8] c"uword\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype39 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str38, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 114, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype37 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str40 = internal constant [7 x i8] c"length\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype41 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str40, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 142, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype39 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str42 = internal constant [15 x i8] c"_Unwind_Action\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype43 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str42, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit9 to { }*), i32 115, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype23 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype44 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype43 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str45 = internal constant [6 x i8] c"sword\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype46 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str45, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 113, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype44 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str47 = internal constant [7 x i8] c"CIE_id\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype48 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str47, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 143, i64 32, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype46 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str49 = internal constant [6 x i8] c"ubyte\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype50 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str49, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 141, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str51 = internal constant [8 x i8] c"version\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype52 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str51, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 144, i64 8, i64 8, i64 64, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype50 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array53 = internal constant [0 x { }*] zeroinitializer, section "llvm.metadata"		; <[0 x { }*]*> [#uses=1]
@llvm.dbg.composite54 = internal constant %llvm.dbg.composite.type { i32 458753, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 8, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast ([0 x { }*]* @llvm.dbg.array53 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str55 = internal constant [13 x i8] c"augmentation\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype56 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str55, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 145, i64 0, i64 8, i64 72, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite54 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array57 = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype41 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype48 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype52 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype56 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite58 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str31, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 141, i64 96, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array57 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype59 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 96, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite58 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype60 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype59 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str61 = internal constant [10 x i8] c"dwarf_fde\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype63 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str40, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 151, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype39 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str64 = internal constant [10 x i8] c"CIE_delta\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype65 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str64, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 152, i64 32, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype46 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str67 = internal constant [9 x i8] c"pc_begin\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype68 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str67, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 153, i64 0, i64 8, i64 64, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite54 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array69 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype63 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype65 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype68 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite70 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str61, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 43, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array69 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype71 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 64, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite70 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype72 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype71 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array73 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype60 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype72 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite74 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array73 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str75 = internal constant [8 x i8] c"get_cie\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.subprogram76 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str75, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str75, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str75, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 162, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite74 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str77 = internal constant [4 x i8] c"fde\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.derivedtype78 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str77, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 162, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype71 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype79 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str77, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 162, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype78 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype80 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype79 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array81 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite82 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array81 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str83 = internal constant [9 x i8] c"next_fde\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.subprogram84 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str83, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str83, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str83, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 168, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite82 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=0]
@.str85 = internal constant [7 x i8] c"object\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype87 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype88 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str67, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 48, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str89 = internal constant [6 x i8] c"tbase\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype90 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str89, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 49, i64 32, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str91 = internal constant [6 x i8] c"dbase\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype92 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str91, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 50, i64 32, i64 32, i64 64, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str94 = internal constant [7 x i8] c"single\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype95 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str94, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 52, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype72 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype96 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite70 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype97 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype96 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str98 = internal constant [6 x i8] c"array\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype99 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str98, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 53, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype97 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str100 = internal constant [11 x i8] c"fde_vector\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str103 = internal constant [10 x i8] c"orig_data\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype104 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str103, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 41, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str105 = internal constant [18 x i8] c"long unsigned int\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.basictype106 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str105, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str107 = internal constant [9 x i8] c"stddef.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit108 = linkonce constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([9 x i8]* @.str107, i32 0, i32 0), i8* getelementptr ([57 x i8]* @.str8, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str109 = internal constant [7 x i8] c"size_t\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype110 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str109, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit108 to { }*), i32 326, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype106 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str111 = internal constant [6 x i8] c"count\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype112 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str111, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 42, i64 32, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype110 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.composite113 = internal constant %llvm.dbg.composite.type { i32 458753, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype72 to { }*), { }* bitcast ([0 x { }*]* @llvm.dbg.array53 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype114 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str98, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 43, i64 0, i64 32, i64 64, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite113 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array115 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype104 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype112 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype114 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite116 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str100, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 40, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array115 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype117 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite116 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str118 = internal constant [5 x i8] c"sort\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype119 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str118, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 54, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype117 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array120 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype95 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype99 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype119 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite121 = internal constant %llvm.dbg.composite.type { i32 458775, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 51, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array120 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str122 = internal constant [2 x i8] c"u\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype123 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str122, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 55, i64 32, i64 32, i64 96, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite121 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str126 = internal constant [7 x i8] c"sorted\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype127 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str126, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 59, i64 1, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype106 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str128 = internal constant [11 x i8] c"from_array\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype129 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str128, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 60, i64 1, i64 32, i64 1, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype106 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str130 = internal constant [15 x i8] c"mixed_encoding\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype131 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str130, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 61, i64 1, i64 32, i64 2, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype106 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str132 = internal constant [9 x i8] c"encoding\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype133 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str132, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 62, i64 8, i64 32, i64 3, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype106 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype134 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str111, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 65, i64 21, i64 32, i64 11, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype106 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array135 = internal constant [5 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype127 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype129 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype131 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype133 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype134 to { }*)], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite136 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 58, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array135 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str137 = internal constant [2 x i8] c"b\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype138 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str137, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 66, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite136 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str139 = internal constant [2 x i8] c"i\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype140 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str139, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 67, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype110 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array141 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype138 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype140 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite142 = internal constant %llvm.dbg.composite.type { i32 458775, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 57, i64 32, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array141 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str143 = internal constant [2 x i8] c"s\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype144 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str143, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 68, i64 32, i64 32, i64 128, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite142 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype145 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite149 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str146 = internal constant [5 x i8] c"next\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype147 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str146, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 74, i64 32, i64 32, i64 160, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array148 = internal constant [6 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype88 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype90 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype92 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype123 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype144 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype147 to { }*)], section "llvm.metadata"		; <[6 x { }*]*> [#uses=1]
@llvm.dbg.composite149 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str85, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 47, i64 192, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([6 x { }*]* @llvm.dbg.array148 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array151 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite152 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array151 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str153 = internal constant [9 x i8] c"last_fde\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.subprogram154 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str153, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str153, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str153, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 176, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite152 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str155 = internal constant [6 x i8] c"saddr\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype156 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str155, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 116, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype46 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype157 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str45, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 113, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype156 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str158 = internal constant [14 x i8] c"gthr-single.h\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.compile_unit159 = linkonce constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([14 x i8]* @.str158, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str160 = internal constant [18 x i8] c"__gthread_mutex_t\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.derivedtype161 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str160, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit159 to { }*), i32 35, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype157 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype162 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype161 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array163 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype162 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite164 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array163 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str165 = internal constant [21 x i8] c"__gthread_mutex_lock\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram166 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([21 x i8]* @.str165, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str165, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str165, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit159 to { }*), i32 220, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite164 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str167 = internal constant [23 x i8] c"__gthread_mutex_unlock\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.subprogram168 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([23 x i8]* @.str167, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str167, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str167, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit159 to { }*), i32 232, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite164 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array169 = internal constant [5 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*)], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite170 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array169 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str171 = internal constant [28 x i8] c"__register_frame_info_bases\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram172 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([28 x i8]* @.str171, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str171, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str171, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 80, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite170 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@unseen_objects = internal global %struct.object* null		; <%struct.object**> [#uses=15]
@llvm.dbg.array173 = internal constant [3 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite174 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array173 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str175 = internal constant [22 x i8] c"__register_frame_info\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram176 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([22 x i8]* @.str175, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str175, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str175, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 106, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite174 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str179 = internal constant [34 x i8] c"__register_frame_info_table_bases\00", section "llvm.metadata"		; <[34 x i8]*> [#uses=1]
@llvm.dbg.subprogram180 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([34 x i8]* @.str179, i32 0, i32 0), i8* getelementptr ([34 x i8]* @.str179, i32 0, i32 0), i8* getelementptr ([34 x i8]* @.str179, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 130, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite170 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str183 = internal constant [28 x i8] c"__register_frame_info_table\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram184 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([28 x i8]* @.str183, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str183, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str183, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 150, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite174 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array185 = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite186 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array185 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str187 = internal constant [22 x i8] c"fde_unencoded_compare\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram188 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([22 x i8]* @.str187, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str187, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str187, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 325, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite186 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str189 = internal constant [16 x i8] c"fde_accumulator\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str191 = internal constant [7 x i8] c"linear\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype192 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str191, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 389, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype117 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str193 = internal constant [8 x i8] c"erratic\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype194 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([8 x i8]* @.str193, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 390, i64 32, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype117 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array195 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype192 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype194 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite196 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([16 x i8]* @.str189, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 388, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array195 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype197 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite196 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array198 = internal constant [3 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype197 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite199 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array198 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str200 = internal constant [11 x i8] c"fde_insert\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram201 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str200, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str200, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str200, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 414, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite199 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=0]
@llvm.dbg.derivedtype202 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite186 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str203 = internal constant [14 x i8] c"fde_compare_t\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype204 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str203, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 388, i64 0, i64 0, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype202 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array205 = internal constant [5 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype204 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype117 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype117 to { }*)], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite206 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array205 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str207 = internal constant [10 x i8] c"fde_split\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.subprogram208 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str207, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str207, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str207, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 434, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite206 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@marker.2702 = internal global %struct.dwarf_fde* null		; <%struct.dwarf_fde**> [#uses=2]
@llvm.dbg.derivedtype209 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array210 = internal constant [6 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype204 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype209 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*)], section "llvm.metadata"		; <[6 x { }*]*> [#uses=1]
@llvm.dbg.composite211 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([6 x { }*]* @llvm.dbg.array210 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str212 = internal constant [15 x i8] c"frame_downheap\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram213 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str212, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str212, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str212, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 480, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite211 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array214 = internal constant [4 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype204 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype117 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite215 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array214 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str216 = internal constant [15 x i8] c"frame_heapsort\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram217 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str216, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str216, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str216, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 506, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite215 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str218 = internal constant [10 x i8] c"fde_merge\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.subprogram219 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([10 x i8]* @.str218, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str218, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str218, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 538, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite206 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array220 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite221 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array220 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str222 = internal constant [29 x i8] c"binary_search_unencoded_fdes\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram223 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([29 x i8]* @.str222, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str222, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str222, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 840, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite221 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array224 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite225 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array224 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str226 = internal constant [22 x i8] c"size_of_encoded_value\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram227 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([22 x i8]* @.str226, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str226, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str226, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i32 75, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite225 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array228 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype34 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite229 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array228 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str230 = internal constant [17 x i8] c"base_from_object\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram231 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str230, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str230, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str230, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 241, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite229 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype232 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype34 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array233 = internal constant [5 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype4 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype34 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype4 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype232 to { }*)], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite234 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array233 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str235 = internal constant [29 x i8] c"read_encoded_value_with_base\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram236 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([29 x i8]* @.str235, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str235, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str235, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i32 185, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite234 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array237 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype60 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite238 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array237 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str239 = internal constant [17 x i8] c"get_cie_encoding\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram240 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str239, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str239, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str239, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 266, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite238 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array241 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype110 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite242 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array241 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str243 = internal constant [26 x i8] c"classify_object_over_fdes\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram244 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([26 x i8]* @.str243, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str243, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str243, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 599, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite242 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array245 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype72 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite246 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array245 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str247 = internal constant [17 x i8] c"get_fde_encoding\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram248 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str247, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str247, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str247, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 311, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite246 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array249 = internal constant [4 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype197 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite250 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array249 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str251 = internal constant [9 x i8] c"add_fdes\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.subprogram252 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([9 x i8]* @.str251, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str251, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str251, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 654, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite250 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str253 = internal constant [28 x i8] c"fde_single_encoding_compare\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram254 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([28 x i8]* @.str253, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str253, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str253, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 338, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite186 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str255 = internal constant [27 x i8] c"fde_mixed_encoding_compare\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram256 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([27 x i8]* @.str255, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str255, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str255, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 354, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite186 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str257 = internal constant [34 x i8] c"binary_search_mixed_encoding_fdes\00", section "llvm.metadata"		; <[34 x i8]*> [#uses=1]
@llvm.dbg.subprogram258 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([34 x i8]* @.str257, i32 0, i32 0), i8* getelementptr ([34 x i8]* @.str257, i32 0, i32 0), i8* getelementptr ([34 x i8]* @.str257, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 897, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite221 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str259 = internal constant [35 x i8] c"binary_search_single_encoding_fdes\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram260 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([35 x i8]* @.str259, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str259, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str259, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 867, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite221 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array261 = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite262 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array261 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str263 = internal constant [19 x i8] c"linear_search_fdes\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram264 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str263, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str263, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str263, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 771, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite262 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array265 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype17 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype197 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype110 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite266 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array265 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str267 = internal constant [15 x i8] c"start_fde_sort\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram268 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str267, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str267, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str267, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 395, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite266 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array269 = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite270 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array269 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str271 = internal constant [23 x i8] c"__register_frame_table\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.subprogram272 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([23 x i8]* @.str271, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str271, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str271, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 156, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite270 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str273 = internal constant [17 x i8] c"__register_frame\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram274 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str273, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str273, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str273, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 112, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite270 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array275 = internal constant [4 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype197 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype110 to { }*)], section "llvm.metadata"		; <[4 x { }*]*> [#uses=1]
@llvm.dbg.composite276 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([4 x { }*]* @llvm.dbg.array275 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str277 = internal constant [13 x i8] c"end_fde_sort\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram278 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str277, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str277, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str277, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 564, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite276 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array279 = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype145 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite280 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array279 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str281 = internal constant [12 x i8] c"init_object\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram282 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([12 x i8]* @.str281, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str281, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str281, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 717, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite280 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str283 = internal constant [14 x i8] c"search_object\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram284 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([14 x i8]* @.str283, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str283, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str283, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 928, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite221 to { }*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str285 = internal constant [15 x i8] c"dwarf_eh_bases\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype287 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str89, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 93, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype288 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([6 x i8]* @.str91, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 94, i64 32, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str289 = internal constant [5 x i8] c"func\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype290 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str289, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 95, i64 32, i64 32, i64 64, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array291 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype287 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype288 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype290 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite292 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([15 x i8]* @.str285, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*), i32 92, i64 96, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array291 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype293 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite292 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array294 = internal constant [3 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype80 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype293 to { }*)], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite295 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array294 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str296 = internal constant [17 x i8] c"_Unwind_Find_FDE\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram297 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([17 x i8]* @.str296, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str296, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str296, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 972, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite295 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@seen_objects = internal global %struct.object* null		; <%struct.object**> [#uses=3]
@llvm.dbg.array298 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype87 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite299 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array298 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str300 = internal constant [30 x i8] c"__deregister_frame_info_bases\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram301 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([30 x i8]* @.str300, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str300, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str300, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 175, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite299 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str302 = internal constant [24 x i8] c"__deregister_frame_info\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram303 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([24 x i8]* @.str302, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str302, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str302, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 223, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite299 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str304 = internal constant [19 x i8] c"__deregister_frame\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram305 = linkonce constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([19 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str304, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 229, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite270 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]

define arm_apcscc void @__register_frame_info_bases(i8* %begin, %struct.object* %ob, i8* %tbase, i8* %dbase) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram172 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 82, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = icmp eq i8* %begin, null		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

bb:		; preds = %entry
	%1 = bitcast i8* %begin to i32*		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %return, label %bb1

bb1:		; preds = %bb
	tail call void @llvm.dbg.stoppoint(i32 85, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = getelementptr %struct.object* %ob, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* inttoptr (i64 4294967295 to i8*), i8** %4, align 4
	tail call void @llvm.dbg.stoppoint(i32 86, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	store i8* %tbase, i8** %5, align 4
	tail call void @llvm.dbg.stoppoint(i32 87, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	store i8* %dbase, i8** %6, align 4
	tail call void @llvm.dbg.stoppoint(i32 88, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%7 = bitcast i8* %begin to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	%8 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	store %struct.dwarf_fde* %7, %struct.dwarf_fde** %8, align 4
	tail call void @llvm.dbg.stoppoint(i32 90, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%9 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	store i32 2040, i32* %9
	tail call void @llvm.dbg.stoppoint(i32 96, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 98, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	%10 = load %struct.object** @unseen_objects, align 4		; <%struct.object*> [#uses=1]
	%11 = getelementptr %struct.object* %ob, i32 0, i32 5		; <%struct.object**> [#uses=1]
	store %struct.object* %10, %struct.object** %11, align 4
	tail call void @llvm.dbg.stoppoint(i32 99, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store %struct.object* %ob, %struct.object** @unseen_objects, align 4
	tail call void @llvm.dbg.stoppoint(i32 101, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 233, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit159 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram172 to { }*))
	ret void

return:		; preds = %bb, %entry
	tail call void @llvm.dbg.stoppoint(i32 101, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
}

declare void @llvm.dbg.func.start({ }*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind readnone

declare void @llvm.dbg.region.end({ }*) nounwind readnone

define internal arm_apcscc i8* @read_sleb128(i8* %p, i32* nocapture %val) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram28 to { }*))
	br label %bb

bb:		; preds = %bb, %entry
	%indvar = phi i32 [ 0, %entry ], [ %tmp, %bb ]		; <i32> [#uses=3]
	%result.0 = phi i32 [ 0, %entry ], [ %4, %bb ]		; <i32> [#uses=1]
	%shift.0 = mul i32 %indvar, 7		; <i32> [#uses=2]
	%tmp12 = add i32 %shift.0, 7		; <i32> [#uses=2]
	%tmp = add i32 %indvar, 1		; <i32> [#uses=2]
	%scevgep = getelementptr i8* %p, i32 %tmp		; <i8*> [#uses=1]
	%p_addr.0 = getelementptr i8* %p, i32 %indvar		; <i8*> [#uses=1]
	%0 = load i8* %p_addr.0, align 1		; <i8> [#uses=2]
	%1 = zext i8 %0 to i32		; <i32> [#uses=2]
	%2 = and i32 %1, 127		; <i32> [#uses=1]
	%3 = shl i32 %2, %shift.0		; <i32> [#uses=1]
	%4 = or i32 %3, %result.0		; <i32> [#uses=4]
	%5 = icmp slt i8 %0, 0		; <i1> [#uses=1]
	br i1 %5, label %bb, label %bb1

bb1:		; preds = %bb
	tail call void @llvm.dbg.stoppoint(i32 171, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%6 = icmp ugt i32 %tmp12, 31		; <i1> [#uses=1]
	br i1 %6, label %bb4, label %bb2

bb2:		; preds = %bb1
	%7 = and i32 %1, 64		; <i32> [#uses=1]
	%8 = icmp eq i32 %7, 0		; <i1> [#uses=1]
	br i1 %8, label %bb4, label %bb3

bb3:		; preds = %bb2
	tail call void @llvm.dbg.stoppoint(i32 172, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%9 = shl i32 1, %tmp12		; <i32> [#uses=1]
	%10 = sub i32 0, %9		; <i32> [#uses=1]
	%11 = or i32 %4, %10		; <i32> [#uses=1]
	br label %bb4

bb4:		; preds = %bb3, %bb2, %bb1
	%result.1 = phi i32 [ %11, %bb3 ], [ %4, %bb1 ], [ %4, %bb2 ]		; <i32> [#uses=1]
	tail call void @llvm.dbg.stoppoint(i32 174, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	store i32 %result.1, i32* %val, align 4
	tail call void @llvm.dbg.stoppoint(i32 175, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram28 to { }*))
	ret i8* %scevgep
}

define arm_apcscc void @__register_frame_info(i8* %begin, %struct.object* %ob) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram176 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 107, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram172 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 82, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = icmp eq i8* %begin, null		; <i1> [#uses=1]
	br i1 %0, label %__register_frame_info_bases.exit, label %bb.i

bb.i:		; preds = %entry
	%1 = bitcast i8* %begin to i32*		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %__register_frame_info_bases.exit, label %bb1.i

bb1.i:		; preds = %bb.i
	tail call void @llvm.dbg.stoppoint(i32 85, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = getelementptr %struct.object* %ob, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* inttoptr (i64 4294967295 to i8*), i8** %4, align 4
	tail call void @llvm.dbg.stoppoint(i32 86, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	store i8* null, i8** %5, align 4
	tail call void @llvm.dbg.stoppoint(i32 87, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	store i8* null, i8** %6, align 4
	tail call void @llvm.dbg.stoppoint(i32 88, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%7 = bitcast i8* %begin to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	%8 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	store %struct.dwarf_fde* %7, %struct.dwarf_fde** %8, align 4
	tail call void @llvm.dbg.stoppoint(i32 90, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%9 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	store i32 2040, i32* %9
	tail call void @llvm.dbg.stoppoint(i32 96, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 98, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	%10 = load %struct.object** @unseen_objects, align 4		; <%struct.object*> [#uses=1]
	%11 = getelementptr %struct.object* %ob, i32 0, i32 5		; <%struct.object**> [#uses=1]
	store %struct.object* %10, %struct.object** %11, align 4
	tail call void @llvm.dbg.stoppoint(i32 99, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store %struct.object* %ob, %struct.object** @unseen_objects, align 4
	tail call void @llvm.dbg.stoppoint(i32 101, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 233, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit159 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram176 to { }*))
	ret void

__register_frame_info_bases.exit:		; preds = %bb.i, %entry
	tail call void @llvm.dbg.stoppoint(i32 108, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram172 to { }*))
	ret void
}

define arm_apcscc void @__register_frame_info_table_bases(i8* %begin, %struct.object* %ob, i8* %tbase, i8* %dbase) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram180 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 131, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.object* %ob, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* inttoptr (i64 4294967295 to i8*), i8** %0, align 4
	tail call void @llvm.dbg.stoppoint(i32 132, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%1 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	store i8* %tbase, i8** %1, align 4
	tail call void @llvm.dbg.stoppoint(i32 133, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	store i8* %dbase, i8** %2, align 4
	tail call void @llvm.dbg.stoppoint(i32 134, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	%.c = bitcast i8* %begin to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	store %struct.dwarf_fde* %.c, %struct.dwarf_fde** %3
	tail call void @llvm.dbg.stoppoint(i32 137, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	store i32 2042, i32* %4
	tail call void @llvm.dbg.stoppoint(i32 140, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 142, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	%5 = load %struct.object** @unseen_objects, align 4		; <%struct.object*> [#uses=1]
	%6 = getelementptr %struct.object* %ob, i32 0, i32 5		; <%struct.object**> [#uses=1]
	store %struct.object* %5, %struct.object** %6, align 4
	tail call void @llvm.dbg.stoppoint(i32 143, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store %struct.object* %ob, %struct.object** @unseen_objects, align 4
	tail call void @llvm.dbg.stoppoint(i32 145, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 146, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram180 to { }*))
	ret void
}

define arm_apcscc void @__register_frame_info_table(i8* %begin, %struct.object* %ob) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram184 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 151, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram180 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 131, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.object* %ob, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* inttoptr (i64 4294967295 to i8*), i8** %0, align 4
	tail call void @llvm.dbg.stoppoint(i32 132, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%1 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	store i8* null, i8** %1, align 4
	tail call void @llvm.dbg.stoppoint(i32 133, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	store i8* null, i8** %2, align 4
	tail call void @llvm.dbg.stoppoint(i32 134, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	%.c.i = bitcast i8* %begin to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	store %struct.dwarf_fde* %.c.i, %struct.dwarf_fde** %3
	tail call void @llvm.dbg.stoppoint(i32 137, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	store i32 2042, i32* %4
	tail call void @llvm.dbg.stoppoint(i32 140, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 142, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	%5 = load %struct.object** @unseen_objects, align 4		; <%struct.object*> [#uses=1]
	%6 = getelementptr %struct.object* %ob, i32 0, i32 5		; <%struct.object**> [#uses=1]
	store %struct.object* %5, %struct.object** %6, align 4
	tail call void @llvm.dbg.stoppoint(i32 143, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store %struct.object* %ob, %struct.object** @unseen_objects, align 4
	tail call void @llvm.dbg.stoppoint(i32 145, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 146, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 152, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram180 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram184 to { }*))
	ret void
}

define internal arm_apcscc i32 @fde_unencoded_compare(%struct.object* nocapture %ob, %struct.dwarf_fde* nocapture %x, %struct.dwarf_fde* nocapture %y) nounwind readonly {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram188 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 326, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.dwarf_fde* %x, i32 0, i32 2		; <[0 x i8]*> [#uses=1]
	%1 = bitcast [0 x i8]* %0 to i32*		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=2]
	tail call void @llvm.dbg.stoppoint(i32 327, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = getelementptr %struct.dwarf_fde* %y, i32 0, i32 2		; <[0 x i8]*> [#uses=1]
	%4 = bitcast [0 x i8]* %3 to i32*		; <i32*> [#uses=1]
	%5 = load i32* %4, align 4		; <i32> [#uses=2]
	tail call void @llvm.dbg.stoppoint(i32 329, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = icmp ugt i32 %2, %5		; <i1> [#uses=1]
	br i1 %6, label %bb4, label %bb1

bb1:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 331, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%7 = icmp ult i32 %2, %5		; <i1> [#uses=1]
	%retval = select i1 %7, i32 -1, i32 0		; <i32> [#uses=1]
	ret i32 %retval

bb4:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 333, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret i32 1
}

define internal arm_apcscc void @frame_downheap(%struct.object* %ob, i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)* nocapture %fde_compare, %struct.dwarf_fde** nocapture %a, i32 %lo, i32 %hi) {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram213 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 483, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb5

bb:		; preds = %bb5
	%0 = add i32 %j.1, 1		; <i32> [#uses=2]
	%1 = icmp slt i32 %0, %hi		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb3

bb1:		; preds = %bb
	%2 = getelementptr %struct.dwarf_fde** %a, i32 %j.1		; <%struct.dwarf_fde**> [#uses=1]
	%3 = load %struct.dwarf_fde** %2, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%4 = getelementptr %struct.dwarf_fde** %a, i32 %0		; <%struct.dwarf_fde**> [#uses=1]
	%5 = load %struct.dwarf_fde** %4, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%6 = tail call arm_apcscc  i32 %fde_compare(%struct.object* %ob, %struct.dwarf_fde* %3, %struct.dwarf_fde* %5)		; <i32> [#uses=1]
	%.lobit = lshr i32 %6, 31		; <i32> [#uses=1]
	%.j.1 = add i32 %.lobit, %j.1		; <i32> [#uses=1]
	br label %bb3

bb3:		; preds = %bb1, %bb
	%j.0 = phi i32 [ %.j.1, %bb1 ], [ %j.1, %bb ]		; <i32> [#uses=3]
	%7 = getelementptr %struct.dwarf_fde** %a, i32 %i.0		; <%struct.dwarf_fde**> [#uses=3]
	%8 = load %struct.dwarf_fde** %7, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%9 = getelementptr %struct.dwarf_fde** %a, i32 %j.0		; <%struct.dwarf_fde**> [#uses=3]
	%10 = load %struct.dwarf_fde** %9, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%11 = tail call arm_apcscc  i32 %fde_compare(%struct.object* %ob, %struct.dwarf_fde* %8, %struct.dwarf_fde* %10)		; <i32> [#uses=1]
	%12 = icmp slt i32 %11, 0		; <i1> [#uses=1]
	br i1 %12, label %bb4, label %return

bb4:		; preds = %bb3
	%13 = load %struct.dwarf_fde** %7, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%14 = load %struct.dwarf_fde** %9, align 4		; <%struct.dwarf_fde*> [#uses=1]
	store %struct.dwarf_fde* %14, %struct.dwarf_fde** %7, align 4
	store %struct.dwarf_fde* %13, %struct.dwarf_fde** %9, align 4
	br label %bb5

bb5:		; preds = %bb4, %entry
	%j.1.in.in = phi i32 [ %lo, %entry ], [ %j.0, %bb4 ]		; <i32> [#uses=1]
	%i.0 = phi i32 [ %lo, %entry ], [ %j.0, %bb4 ]		; <i32> [#uses=1]
	%j.1.in = shl i32 %j.1.in.in, 1		; <i32> [#uses=1]
	%j.1 = or i32 %j.1.in, 1		; <i32> [#uses=5]
	%15 = icmp slt i32 %j.1, %hi		; <i1> [#uses=1]
	br i1 %15, label %bb, label %return

return:		; preds = %bb5, %bb3
	tail call void @llvm.dbg.stoppoint(i32 498, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram213 to { }*))
	ret void
}

define internal arm_apcscc void @frame_heapsort(%struct.object* %ob, i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)* nocapture %fde_compare, %struct.fde_vector* nocapture %erratic) {
entry:
	%erratic15 = bitcast %struct.fde_vector* %erratic to i8*		; <i8*> [#uses=1]
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram217 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 510, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.fde_vector* %erratic, i32 0, i32 2, i32 0		; <%struct.dwarf_fde**> [#uses=4]
	tail call void @llvm.dbg.stoppoint(i32 514, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%1 = getelementptr %struct.fde_vector* %erratic, i32 0, i32 1		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=4]
	tail call void @llvm.dbg.stoppoint(i32 520, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = lshr i32 %2, 1		; <i32> [#uses=2]
	%m.010 = add i32 %3, -1		; <i32> [#uses=2]
	%4 = icmp slt i32 %m.010, 0		; <i1> [#uses=1]
	br i1 %4, label %bb4.loopexit, label %bb.nph12

bb.nph12:		; preds = %entry
	%tmp25 = add i32 %3, -2		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph12
	%indvar21 = phi i32 [ 0, %bb.nph12 ], [ %indvar.next22, %bb ]		; <i32> [#uses=3]
	%m.011 = sub i32 %m.010, %indvar21		; <i32> [#uses=1]
	tail call arm_apcscc  void @frame_downheap(%struct.object* %ob, i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)* %fde_compare, %struct.dwarf_fde** %0, i32 %m.011, i32 %2)
	%m.0 = sub i32 %tmp25, %indvar21		; <i32> [#uses=1]
	%5 = icmp slt i32 %m.0, 0		; <i1> [#uses=1]
	%indvar.next22 = add i32 %indvar21, 1		; <i32> [#uses=1]
	br i1 %5, label %bb4.loopexit, label %bb

bb.nph:		; preds = %bb4.loopexit
	%tmp17 = shl i32 %2, 2		; <i32> [#uses=1]
	%tmp18 = add i32 %tmp17, 4		; <i32> [#uses=1]
	br label %bb3

bb3:		; preds = %bb3, %bb.nph
	%indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb3 ]		; <i32> [#uses=3]
	%m.18 = sub i32 %m.17, %indvar		; <i32> [#uses=1]
	%tmp16 = mul i32 %indvar, -4		; <i32> [#uses=1]
	%tmp19 = add i32 %tmp16, %tmp18		; <i32> [#uses=1]
	%scevgep = getelementptr i8* %erratic15, i32 %tmp19		; <i8*> [#uses=1]
	%scevgep20 = bitcast i8* %scevgep to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=2]
	%6 = load %struct.dwarf_fde** %0, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%7 = load %struct.dwarf_fde** %scevgep20, align 4		; <%struct.dwarf_fde*> [#uses=1]
	store %struct.dwarf_fde* %7, %struct.dwarf_fde** %0, align 4
	store %struct.dwarf_fde* %6, %struct.dwarf_fde** %scevgep20, align 4
	tail call arm_apcscc  void @frame_downheap(%struct.object* %ob, i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)* %fde_compare, %struct.dwarf_fde** %0, i32 0, i32 %m.18)
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %m.17		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb3

bb4.loopexit:		; preds = %bb, %entry
	%m.17 = add i32 %2, -1		; <i32> [#uses=3]
	tail call void @llvm.dbg.stoppoint(i32 526, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%8 = icmp sgt i32 %m.17, 0		; <i1> [#uses=1]
	br i1 %8, label %bb.nph, label %return

return:		; preds = %bb4.loopexit, %bb3
	tail call void @llvm.dbg.stoppoint(i32 532, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram217 to { }*))
	ret void
}

define internal arm_apcscc i32 @size_of_encoded_value(i8 zeroext %encoding) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram227 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 76, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%0 = icmp eq i8 %encoding, -1		; <i1> [#uses=1]
	br i1 %0, label %bb7, label %bb1

bb1:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 79, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%1 = zext i8 %encoding to i32		; <i32> [#uses=1]
	%2 = and i32 %1, 7		; <i32> [#uses=1]
	switch i32 %2, label %bb6 [
		i32 0, label %bb7
		i32 2, label %bb3
		i32 3, label %bb7
		i32 4, label %bb5
	]

bb3:		; preds = %bb1
	tail call void @llvm.dbg.stoppoint(i32 84, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram227 to { }*))
	ret i32 2

bb5:		; preds = %bb1
	tail call void @llvm.dbg.stoppoint(i32 88, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	ret i32 8

bb6:		; preds = %bb1
	tail call void @llvm.dbg.stoppoint(i32 90, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	tail call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb7:		; preds = %bb1, %bb1, %entry
	%.0 = phi i32 [ 0, %entry ], [ 4, %bb1 ], [ 4, %bb1 ]		; <i32> [#uses=1]
	tail call void @llvm.dbg.stoppoint(i32 90, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	ret i32 %.0
}

declare arm_apcscc void @abort() noreturn nounwind

define internal arm_apcscc i8* @read_encoded_value_with_base(i8 zeroext %encoding, i32 %base, i8* %p, i32* nocapture %val) nounwind {
entry:
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram236 to { }*))
	call void @llvm.dbg.stoppoint(i32 200, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%0 = icmp eq i8 %encoding, 80		; <i1> [#uses=1]
	br i1 %0, label %bb, label %bb2

bb:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 203, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%ctg2 = getelementptr i8* %p, i32 3		; <i8*> [#uses=1]
	%1 = ptrtoint i8* %ctg2 to i32		; <i32> [#uses=1]
	%2 = and i32 %1, -4		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 204, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%3 = inttoptr i32 %2 to i32*		; <i32*> [#uses=1]
	%4 = load i32* %3, align 4		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 205, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%tmp21 = inttoptr i32 %2 to i8*		; <i8*> [#uses=1]
	%5 = getelementptr i8* %tmp21, i32 4		; <i8*> [#uses=1]
	br label %bb19

bb2:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 209, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%6 = zext i8 %encoding to i32		; <i32> [#uses=2]
	%7 = and i32 %6, 15		; <i32> [#uses=1]
	switch i32 %7, label %bb12 [
		i32 0, label %bb3
		i32 1, label %bb.i
		i32 2, label %bb6
		i32 3, label %bb7
		i32 4, label %bb8
		i32 9, label %bb5
		i32 10, label %bb9
		i32 11, label %bb10
		i32 12, label %bb11
	]

bb3:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 212, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%8 = bitcast i8* %p to i8**		; <i8**> [#uses=1]
	%9 = load i8** %8, align 1		; <i8*> [#uses=1]
	%10 = ptrtoint i8* %9 to i32		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 213, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%11 = getelementptr i8* %p, i32 4		; <i8*> [#uses=1]
	br label %bb13

bb.i:		; preds = %bb.i, %bb2
	%indvar.i = phi i32 [ 0, %bb2 ], [ %18, %bb.i ]		; <i32> [#uses=3]
	%result.0.i = phi i32 [ 0, %bb2 ], [ %16, %bb.i ]		; <i32> [#uses=1]
	%p_addr.0.i = getelementptr i8* %p, i32 %indvar.i		; <i8*> [#uses=1]
	%shift.0.i = mul i32 %indvar.i, 7		; <i32> [#uses=1]
	%12 = load i8* %p_addr.0.i, align 1		; <i8> [#uses=2]
	%13 = zext i8 %12 to i32		; <i32> [#uses=1]
	%14 = and i32 %13, 127		; <i32> [#uses=1]
	%15 = shl i32 %14, %shift.0.i		; <i32> [#uses=1]
	%16 = or i32 %15, %result.0.i		; <i32> [#uses=2]
	%17 = icmp slt i8 %12, 0		; <i1> [#uses=1]
	%18 = add i32 %indvar.i, 1		; <i32> [#uses=2]
	br i1 %17, label %bb.i, label %read_uleb128.exit

read_uleb128.exit:		; preds = %bb.i
	%scevgep.i = getelementptr i8* %p, i32 %18		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 220, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	br label %bb13

bb5:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 227, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%19 = call arm_apcscc  i8* @read_sleb128(i8* %p, i32* %tmp)		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 228, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%20 = load i32* %tmp, align 4		; <i32> [#uses=1]
	br label %bb13

bb6:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 233, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%21 = bitcast i8* %p to i16*		; <i16*> [#uses=1]
	%22 = load i16* %21, align 1		; <i16> [#uses=1]
	%23 = zext i16 %22 to i32		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 234, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%24 = getelementptr i8* %p, i32 2		; <i8*> [#uses=1]
	br label %bb13

bb7:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 237, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%25 = bitcast i8* %p to i32*		; <i32*> [#uses=1]
	%26 = load i32* %25, align 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 238, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%27 = getelementptr i8* %p, i32 4		; <i8*> [#uses=1]
	br label %bb13

bb8:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 241, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%28 = bitcast i8* %p to i64*		; <i64*> [#uses=1]
	%29 = load i64* %28, align 1		; <i64> [#uses=1]
	%30 = trunc i64 %29 to i32		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 242, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%31 = getelementptr i8* %p, i32 8		; <i8*> [#uses=1]
	br label %bb13

bb9:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 246, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%32 = bitcast i8* %p to i16*		; <i16*> [#uses=1]
	%33 = load i16* %32, align 1		; <i16> [#uses=1]
	%34 = sext i16 %33 to i32		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 247, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%35 = getelementptr i8* %p, i32 2		; <i8*> [#uses=1]
	br label %bb13

bb10:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 250, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%36 = bitcast i8* %p to i32*		; <i32*> [#uses=1]
	%37 = load i32* %36, align 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 251, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%38 = getelementptr i8* %p, i32 4		; <i8*> [#uses=1]
	br label %bb13

bb11:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 254, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%39 = bitcast i8* %p to i64*		; <i64*> [#uses=1]
	%40 = load i64* %39, align 1		; <i64> [#uses=1]
	%41 = trunc i64 %40 to i32		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%42 = getelementptr i8* %p, i32 8		; <i8*> [#uses=1]
	br label %bb13

bb12:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 259, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb13:		; preds = %bb11, %bb10, %bb9, %bb8, %bb7, %bb6, %bb5, %read_uleb128.exit, %bb3
	%p_addr.1 = phi i8* [ %42, %bb11 ], [ %38, %bb10 ], [ %35, %bb9 ], [ %19, %bb5 ], [ %31, %bb8 ], [ %27, %bb7 ], [ %24, %bb6 ], [ %scevgep.i, %read_uleb128.exit ], [ %11, %bb3 ]		; <i8*> [#uses=3]
	%result.1 = phi i32 [ %41, %bb11 ], [ %37, %bb10 ], [ %34, %bb9 ], [ %20, %bb5 ], [ %30, %bb8 ], [ %26, %bb7 ], [ %23, %bb6 ], [ %16, %read_uleb128.exit ], [ %10, %bb3 ]		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 262, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%43 = icmp eq i32 %result.1, 0		; <i1> [#uses=1]
	br i1 %43, label %bb19, label %bb14

bb14:		; preds = %bb13
	call void @llvm.dbg.stoppoint(i32 264, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%44 = and i32 %6, 112		; <i32> [#uses=1]
	%45 = icmp eq i32 %44, 16		; <i1> [#uses=1]
	br i1 %45, label %bb15, label %bb17

bb15:		; preds = %bb14
	%46 = ptrtoint i8* %p to i32		; <i32> [#uses=1]
	br label %bb17

bb17:		; preds = %bb15, %bb14
	%iftmp.9.0 = phi i32 [ %46, %bb15 ], [ %base, %bb14 ]		; <i32> [#uses=1]
	%47 = add i32 %iftmp.9.0, %result.1		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 266, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%48 = icmp slt i8 %encoding, 0		; <i1> [#uses=1]
	br i1 %48, label %bb18, label %bb19

bb18:		; preds = %bb17
	call void @llvm.dbg.stoppoint(i32 267, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	%49 = inttoptr i32 %47 to i32*		; <i32*> [#uses=1]
	%50 = load i32* %49, align 4		; <i32> [#uses=1]
	br label %bb19

bb19:		; preds = %bb18, %bb17, %bb13, %bb
	%p_addr.0 = phi i8* [ %5, %bb ], [ %p_addr.1, %bb18 ], [ %p_addr.1, %bb13 ], [ %p_addr.1, %bb17 ]		; <i8*> [#uses=1]
	%result.0 = phi i32 [ %4, %bb ], [ %50, %bb18 ], [ %result.1, %bb13 ], [ %47, %bb17 ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 271, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	store i32 %result.0, i32* %val, align 4
	call void @llvm.dbg.stoppoint(i32 272, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram236 to { }*))
	ret i8* %p_addr.0
}

define internal arm_apcscc i32 @get_cie_encoding(%struct.dwarf_cie* %cie) nounwind {
entry:
	%cie37 = bitcast %struct.dwarf_cie* %cie to i8*		; <i8*> [#uses=1]
	%stmp = alloca i32, align 4		; <i32*> [#uses=1]
	%dummy = alloca i32, align 4		; <i32*> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram240 to { }*))
	call void @llvm.dbg.stoppoint(i32 272, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.dwarf_cie* %cie, i32 0, i32 3, i32 0		; <i8*> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 273, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%1 = load i8* %0, align 1		; <i8> [#uses=1]
	%2 = icmp eq i8 %1, 122		; <i1> [#uses=1]
	br i1 %2, label %bb1, label %bb13

bb1:		; preds = %entry
	%3 = call arm_apcscc  i32 @strlen(i8* %0) nounwind readonly		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 277, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%tmp38 = add i32 %3, 10		; <i32> [#uses=1]
	br label %bb.i

bb.i:		; preds = %bb.i, %bb1
	%indvar.i = phi i32 [ 0, %bb1 ], [ %6, %bb.i ]		; <i32> [#uses=3]
	%tmp39 = add i32 %indvar.i, %tmp38		; <i32> [#uses=1]
	%p_addr.0.i = getelementptr i8* %cie37, i32 %tmp39		; <i8*> [#uses=1]
	%4 = load i8* %p_addr.0.i, align 1		; <i8> [#uses=1]
	%5 = icmp slt i8 %4, 0		; <i1> [#uses=1]
	%6 = add i32 %indvar.i, 1		; <i32> [#uses=1]
	br i1 %5, label %bb.i, label %read_uleb128.exit

read_uleb128.exit:		; preds = %bb.i
	call void @llvm.dbg.stoppoint(i32 276, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%.sum40 = add i32 %indvar.i, %3		; <i32> [#uses=1]
	%.sum31 = add i32 %.sum40, 2		; <i32> [#uses=1]
	%scevgep.i = getelementptr %struct.dwarf_cie* %cie, i32 0, i32 3, i32 %.sum31		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 278, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	%7 = call arm_apcscc  i8* @read_sleb128(i8* %scevgep.i, i32* %stmp)		; <i8*> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 279, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%8 = getelementptr %struct.dwarf_cie* %cie, i32 0, i32 2		; <i8*> [#uses=1]
	%9 = load i8* %8, align 1		; <i8> [#uses=1]
	%10 = icmp eq i8 %9, 1		; <i1> [#uses=1]
	br i1 %10, label %bb2, label %bb.i20

bb2:		; preds = %read_uleb128.exit
	call void @llvm.dbg.stoppoint(i32 280, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%11 = getelementptr i8* %7, i32 1		; <i8*> [#uses=1]
	br label %bb.i28

bb.i20:		; preds = %bb.i20, %read_uleb128.exit
	%indvar.i15 = phi i32 [ 0, %read_uleb128.exit ], [ %14, %bb.i20 ]		; <i32> [#uses=2]
	%p_addr.0.i18 = getelementptr i8* %7, i32 %indvar.i15		; <i8*> [#uses=1]
	%12 = load i8* %p_addr.0.i18, align 1		; <i8> [#uses=1]
	%13 = icmp slt i8 %12, 0		; <i1> [#uses=1]
	%14 = add i32 %indvar.i15, 1		; <i32> [#uses=2]
	br i1 %13, label %bb.i20, label %read_uleb128.exit22

read_uleb128.exit22:		; preds = %bb.i20
	%scevgep.i21 = getelementptr i8* %7, i32 %14		; <i8*> [#uses=1]
	tail call void @llvm.dbg.stoppoint(i32 149, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	br label %bb.i28

bb.i28:		; preds = %bb.i28, %read_uleb128.exit22, %bb2
	%p.0.ph = phi i8* [ %11, %bb2 ], [ %scevgep.i21, %read_uleb128.exit22 ], [ %p.0.ph, %bb.i28 ]		; <i8*> [#uses=3]
	%indvar.i23 = phi i32 [ 0, %read_uleb128.exit22 ], [ 0, %bb2 ], [ %17, %bb.i28 ]		; <i32> [#uses=2]
	%p_addr.0.i26 = getelementptr i8* %p.0.ph, i32 %indvar.i23		; <i8*> [#uses=1]
	%15 = load i8* %p_addr.0.i26, align 1		; <i8> [#uses=1]
	%16 = icmp slt i8 %15, 0		; <i1> [#uses=1]
	%17 = add i32 %indvar.i23, 1		; <i32> [#uses=2]
	br i1 %16, label %bb.i28, label %read_uleb128.exit30

read_uleb128.exit30:		; preds = %bb.i28
	%scevgep.i29 = getelementptr i8* %p.0.ph, i32 %17		; <i8*> [#uses=1]
	tail call void @llvm.dbg.stoppoint(i32 149, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
	br label %bb5

bb5:		; preds = %bb5.backedge, %read_uleb128.exit30
	%indvar = phi i32 [ 1, %read_uleb128.exit30 ], [ %phitmp, %bb5.backedge ]		; <i32> [#uses=2]
	%p.2 = phi i8* [ %scevgep.i29, %read_uleb128.exit30 ], [ %p.2.be, %bb5.backedge ]		; <i8*> [#uses=4]
	%aug.0 = getelementptr %struct.dwarf_cie* %cie, i32 0, i32 4, i32 %indvar		; <i8*> [#uses=1]
	%18 = load i8* %aug.0, align 1		; <i8> [#uses=1]
	switch i8 %18, label %bb13 [
		i8 82, label %bb6
		i8 80, label %bb8
		i8 76, label %bb10
	]

bb6:		; preds = %bb5
	call void @llvm.dbg.stoppoint(i32 290, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%19 = load i8* %p.2, align 1		; <i8> [#uses=1]
	%20 = zext i8 %19 to i32		; <i32> [#uses=1]
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram240 to { }*))
	ret i32 %20

bb8:		; preds = %bb5
	%21 = load i8* %p.2, align 1		; <i8> [#uses=1]
	%22 = and i8 %21, 127		; <i8> [#uses=1]
	%23 = getelementptr i8* %p.2, i32 1		; <i8*> [#uses=1]
	%24 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %22, i32 0, i8* %23, i32* %dummy)		; <i8*> [#uses=1]
	br label %bb5.backedge

bb10:		; preds = %bb5
	%25 = getelementptr i8* %p.2, i32 1		; <i8*> [#uses=1]
	br label %bb5.backedge

bb5.backedge:		; preds = %bb10, %bb8
	%p.2.be = phi i8* [ %24, %bb8 ], [ %25, %bb10 ]		; <i8*> [#uses=1]
	%phitmp = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb5

bb13:		; preds = %bb5, %entry
	call void @llvm.dbg.stoppoint(i32 305, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret i32 0
}

declare arm_apcscc i32 @strlen(i8* nocapture) nounwind readonly

define internal arm_apcscc i32 @classify_object_over_fdes(%struct.object* nocapture %ob, %struct.dwarf_fde* %this_fde) nounwind {
entry:
	%pc_begin = alloca i32, align 4		; <i32*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram244 to { }*))
	call void @llvm.dbg.stoppoint(i32 603, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.object* %ob, i32 0, i32 0		; <i8**> [#uses=2]
	%1 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=3]
	%2 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%3 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	br label %bb13

bb:		; preds = %bb13
	%4 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 1		; <i32*> [#uses=2]
	%5 = load i32* %4, align 1		; <i32> [#uses=2]
	%6 = icmp eq i32 %5, 0		; <i1> [#uses=1]
	br i1 %6, label %bb12, label %bb1

bb1:		; preds = %bb
	%7 = bitcast i32* %4 to i8*		; <i8*> [#uses=1]
	%8 = sub i32 0, %5		; <i32> [#uses=1]
	%9 = getelementptr i8* %7, i32 %8		; <i8*> [#uses=1]
	%10 = bitcast i8* %9 to %struct.dwarf_cie*		; <%struct.dwarf_cie*> [#uses=5]
	%11 = icmp eq %struct.dwarf_cie* %10, %last_cie.2		; <i1> [#uses=1]
	br i1 %11, label %bb6, label %bb2

bb2:		; preds = %bb1
	%12 = call arm_apcscc  i32 @get_cie_encoding(%struct.dwarf_cie* %10)		; <i32> [#uses=7]
	%13 = trunc i32 %12 to i8		; <i8> [#uses=1]
	%14 = icmp eq i8 %13, -1		; <i1> [#uses=1]
	br i1 %14, label %base_from_object.exit, label %bb1.i

bb1.i:		; preds = %bb2
	%15 = and i32 %12, 112		; <i32> [#uses=1]
	switch i32 %15, label %bb5.i [
		i32 0, label %base_from_object.exit
		i32 16, label %base_from_object.exit
		i32 32, label %bb3.i
		i32 48, label %bb4.i
		i32 80, label %base_from_object.exit
	]

bb3.i:		; preds = %bb1.i
	%16 = load i8** %2, align 4		; <i8*> [#uses=1]
	%17 = ptrtoint i8* %16 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb4.i:		; preds = %bb1.i
	%18 = load i8** %3, align 4		; <i8*> [#uses=1]
	%19 = ptrtoint i8* %18 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb5.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 605, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram154 to { }*))
	call void @llvm.dbg.stoppoint(i32 616, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*))
	call void @llvm.dbg.stoppoint(i32 617, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*))
	call void @llvm.dbg.stoppoint(i32 621, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

base_from_object.exit:		; preds = %bb4.i, %bb3.i, %bb1.i, %bb1.i, %bb1.i, %bb2
	%20 = phi i32 [ %19, %bb4.i ], [ %17, %bb3.i ], [ 0, %bb2 ], [ 0, %bb1.i ], [ 0, %bb1.i ], [ 0, %bb1.i ]		; <i32> [#uses=3]
	%21 = load i32* %1		; <i32> [#uses=4]
	%22 = and i32 %21, 2040		; <i32> [#uses=1]
	%23 = icmp eq i32 %22, 2040		; <i1> [#uses=1]
	br i1 %23, label %bb3, label %bb4

bb3:		; preds = %base_from_object.exit
	%24 = shl i32 %12, 3		; <i32> [#uses=1]
	%25 = and i32 %24, 2040		; <i32> [#uses=1]
	%26 = and i32 %21, -2041		; <i32> [#uses=1]
	%27 = or i32 %26, %25		; <i32> [#uses=1]
	store i32 %27, i32* %1
	br label %bb6

bb4:		; preds = %base_from_object.exit
	%28 = lshr i32 %21, 3		; <i32> [#uses=1]
	%29 = and i32 %28, 255		; <i32> [#uses=1]
	%30 = icmp eq i32 %29, %12		; <i1> [#uses=1]
	br i1 %30, label %bb6, label %bb5

bb5:		; preds = %bb4
	%31 = or i32 %21, 4		; <i32> [#uses=1]
	store i32 %31, i32* %1
	br label %bb6

bb6:		; preds = %bb5, %bb4, %bb3, %bb1
	%base.0 = phi i32 [ %20, %bb3 ], [ %20, %bb5 ], [ %base.2, %bb1 ], [ %20, %bb4 ]		; <i32> [#uses=4]
	%encoding.0 = phi i32 [ %12, %bb3 ], [ %12, %bb5 ], [ %encoding.2, %bb1 ], [ %12, %bb4 ]		; <i32> [#uses=4]
	%last_cie.0 = phi %struct.dwarf_cie* [ %10, %bb3 ], [ %10, %bb5 ], [ %last_cie.2, %bb1 ], [ %10, %bb4 ]		; <%struct.dwarf_cie*> [#uses=3]
	%32 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%33 = trunc i32 %encoding.0 to i8		; <i8> [#uses=2]
	%34 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %33, i32 %base.0, i8* %32, i32* %pc_begin)		; <i8*> [#uses=0]
	%35 = call arm_apcscc  i32 @size_of_encoded_value(i8 zeroext %33)		; <i32> [#uses=2]
	%36 = icmp ugt i32 %35, 3		; <i1> [#uses=1]
	br i1 %36, label %bb9, label %bb7

bb7:		; preds = %bb6
	%37 = shl i32 %35, 3		; <i32> [#uses=1]
	%38 = shl i32 1, %37		; <i32> [#uses=1]
	%39 = add i32 %38, -1		; <i32> [#uses=1]
	br label %bb9

bb9:		; preds = %bb7, %bb6
	%mask.0 = phi i32 [ %39, %bb7 ], [ -1, %bb6 ]		; <i32> [#uses=1]
	%40 = load i32* %pc_begin, align 4		; <i32> [#uses=2]
	%41 = and i32 %40, %mask.0		; <i32> [#uses=1]
	%42 = icmp eq i32 %41, 0		; <i1> [#uses=1]
	br i1 %42, label %bb12, label %bb10

bb10:		; preds = %bb9
	%43 = add i32 %count.1, 1		; <i32> [#uses=2]
	%44 = load i8** %0, align 4		; <i8*> [#uses=1]
	%45 = inttoptr i32 %40 to i8*		; <i8*> [#uses=2]
	%46 = icmp ugt i8* %44, %45		; <i1> [#uses=1]
	br i1 %46, label %bb11, label %bb12

bb11:		; preds = %bb10
	store i8* %45, i8** %0, align 4
	br label %bb12

bb12:		; preds = %bb11, %bb10, %bb9, %bb
	%base.1 = phi i32 [ %base.0, %bb11 ], [ %base.2, %bb ], [ %base.0, %bb9 ], [ %base.0, %bb10 ]		; <i32> [#uses=1]
	%encoding.1 = phi i32 [ %encoding.0, %bb11 ], [ %encoding.2, %bb ], [ %encoding.0, %bb9 ], [ %encoding.0, %bb10 ]		; <i32> [#uses=1]
	%count.0 = phi i32 [ %43, %bb11 ], [ %count.1, %bb ], [ %count.1, %bb9 ], [ %43, %bb10 ]		; <i32> [#uses=1]
	%last_cie.1 = phi %struct.dwarf_cie* [ %last_cie.0, %bb11 ], [ %last_cie.2, %bb ], [ %last_cie.0, %bb9 ], [ %last_cie.0, %bb10 ]		; <%struct.dwarf_cie*> [#uses=1]
	%47 = bitcast %struct.dwarf_fde* %this_fde_addr.0 to i8*		; <i8*> [#uses=1]
	%48 = load i32* %51, align 1		; <i32> [#uses=1]
	%.sum.i = add i32 %48, 4		; <i32> [#uses=1]
	%49 = getelementptr i8* %47, i32 %.sum.i		; <i8*> [#uses=1]
	%50 = bitcast i8* %49 to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	br label %bb13

bb13:		; preds = %bb12, %entry
	%base.2 = phi i32 [ 0, %entry ], [ %base.1, %bb12 ]		; <i32> [#uses=2]
	%encoding.2 = phi i32 [ 0, %entry ], [ %encoding.1, %bb12 ]		; <i32> [#uses=2]
	%count.1 = phi i32 [ 0, %entry ], [ %count.0, %bb12 ]		; <i32> [#uses=4]
	%this_fde_addr.0 = phi %struct.dwarf_fde* [ %this_fde, %entry ], [ %50, %bb12 ]		; <%struct.dwarf_fde*> [#uses=4]
	%last_cie.2 = phi %struct.dwarf_cie* [ null, %entry ], [ %last_cie.1, %bb12 ]		; <%struct.dwarf_cie*> [#uses=3]
	%51 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 0		; <i32*> [#uses=2]
	%52 = load i32* %51, align 1		; <i32> [#uses=1]
	%53 = icmp eq i32 %52, 0		; <i1> [#uses=1]
	br i1 %53, label %bb14, label %bb

bb14:		; preds = %bb13
	call void @llvm.dbg.stoppoint(i32 605, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram154 to { }*))
	call void @llvm.dbg.stoppoint(i32 649, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram244 to { }*))
	ret i32 %count.1
}

define internal arm_apcscc void @add_fdes(%struct.object* nocapture %ob, %struct.fde_accumulator* nocapture %accu, %struct.dwarf_fde* %this_fde) nounwind {
entry:
	%pc_begin = alloca i32, align 4		; <i32*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram252 to { }*))
	call void @llvm.dbg.stoppoint(i32 656, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=2]
	%1 = load i32* %0		; <i32> [#uses=1]
	%2 = lshr i32 %1, 3		; <i32> [#uses=3]
	%3 = and i32 %2, 255		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 657, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = trunc i32 %2 to i8		; <i8> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 242, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%5 = icmp eq i8 %4, -1		; <i1> [#uses=1]
	br i1 %5, label %bb12.preheader, label %bb1.i

bb1.i:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 245, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%6 = and i32 %2, 112		; <i32> [#uses=1]
	switch i32 %6, label %bb5.i [
		i32 0, label %bb12.preheader
		i32 16, label %bb12.preheader
		i32 32, label %bb3.i
		i32 48, label %bb4.i
		i32 80, label %bb12.preheader
	]

bb12.preheader:		; preds = %bb4.i, %bb3.i, %bb1.i, %bb1.i, %bb1.i, %entry
	%base.2.ph = phi i32 [ %15, %bb4.i ], [ %12, %bb3.i ], [ 0, %entry ], [ 0, %bb1.i ], [ 0, %bb1.i ], [ 0, %bb1.i ]		; <i32> [#uses=1]
	%7 = getelementptr %struct.fde_accumulator* %accu, i32 0, i32 0		; <%struct.fde_vector**> [#uses=1]
	%8 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%9 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	br label %bb12

bb3.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 253, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%10 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%11 = load i8** %10, align 4		; <i8*> [#uses=1]
	%12 = ptrtoint i8* %11 to i32		; <i32> [#uses=1]
	br label %bb12.preheader

bb4.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%13 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	%14 = load i8** %13, align 4		; <i8*> [#uses=1]
	%15 = ptrtoint i8* %14 to i32		; <i32> [#uses=1]
	br label %bb12.preheader

bb5.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb:		; preds = %bb12
	%16 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 1		; <i32*> [#uses=2]
	%17 = load i32* %16, align 1		; <i32> [#uses=2]
	%18 = icmp eq i32 %17, 0		; <i1> [#uses=1]
	br i1 %18, label %bb11, label %bb1

bb1:		; preds = %bb
	%19 = load i32* %0		; <i32> [#uses=1]
	%20 = and i32 %19, 4		; <i32> [#uses=1]
	%21 = icmp eq i32 %20, 0		; <i1> [#uses=1]
	br i1 %21, label %bb4, label %bb2

bb2:		; preds = %bb1
	%22 = bitcast i32* %16 to i8*		; <i8*> [#uses=1]
	%23 = sub i32 0, %17		; <i32> [#uses=1]
	%24 = getelementptr i8* %22, i32 %23		; <i8*> [#uses=1]
	%25 = bitcast i8* %24 to %struct.dwarf_cie*		; <%struct.dwarf_cie*> [#uses=8]
	%26 = icmp eq %struct.dwarf_cie* %25, %last_cie.2		; <i1> [#uses=1]
	br i1 %26, label %bb4, label %bb3

bb3:		; preds = %bb2
	%27 = call arm_apcscc  i32 @get_cie_encoding(%struct.dwarf_cie* %25)		; <i32> [#uses=8]
	%28 = trunc i32 %27 to i8		; <i8> [#uses=1]
	%29 = icmp eq i8 %28, -1		; <i1> [#uses=1]
	br i1 %29, label %bb4, label %bb1.i14

bb1.i14:		; preds = %bb3
	%30 = and i32 %27, 112		; <i32> [#uses=1]
	switch i32 %30, label %bb5.i17 [
		i32 0, label %bb4
		i32 16, label %bb4
		i32 32, label %bb3.i15
		i32 48, label %bb4.i16
		i32 80, label %bb4
	]

bb3.i15:		; preds = %bb1.i14
	%31 = load i8** %8, align 4		; <i8*> [#uses=1]
	%32 = ptrtoint i8* %31 to i32		; <i32> [#uses=1]
	br label %bb4

bb4.i16:		; preds = %bb1.i14
	%33 = load i8** %9, align 4		; <i8*> [#uses=1]
	%34 = ptrtoint i8* %33 to i32		; <i32> [#uses=1]
	br label %bb4

bb5.i17:		; preds = %bb1.i14
	call void @llvm.dbg.stoppoint(i32 659, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram154 to { }*))
	call void @llvm.dbg.stoppoint(i32 671, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*))
	call void @llvm.dbg.stoppoint(i32 672, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*))
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb4:		; preds = %bb4.i16, %bb3.i15, %bb1.i14, %bb1.i14, %bb1.i14, %bb3, %bb2, %bb1
	%base.0 = phi i32 [ %34, %bb4.i16 ], [ %32, %bb3.i15 ], [ %base.2, %bb1 ], [ %base.2, %bb2 ], [ 0, %bb3 ], [ 0, %bb1.i14 ], [ 0, %bb1.i14 ], [ 0, %bb1.i14 ]		; <i32> [#uses=5]
	%encoding.0 = phi i32 [ %27, %bb3.i15 ], [ %27, %bb4.i16 ], [ %encoding.2, %bb1 ], [ %encoding.2, %bb2 ], [ %27, %bb3 ], [ %27, %bb1.i14 ], [ %27, %bb1.i14 ], [ %27, %bb1.i14 ]		; <i32> [#uses=6]
	%last_cie.0 = phi %struct.dwarf_cie* [ %25, %bb3.i15 ], [ %25, %bb4.i16 ], [ %last_cie.2, %bb1 ], [ %last_cie.2, %bb2 ], [ %25, %bb3 ], [ %25, %bb1.i14 ], [ %25, %bb1.i14 ], [ %25, %bb1.i14 ]		; <%struct.dwarf_cie*> [#uses=4]
	%35 = icmp eq i32 %encoding.0, 0		; <i1> [#uses=1]
	br i1 %35, label %bb5, label %bb6

bb5:		; preds = %bb4
	%36 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 2		; <[0 x i8]*> [#uses=1]
	%37 = bitcast [0 x i8]* %36 to i32*		; <i32*> [#uses=1]
	%38 = load i32* %37, align 4		; <i32> [#uses=1]
	%39 = icmp eq i32 %38, 0		; <i1> [#uses=1]
	br i1 %39, label %bb11, label %bb10

bb6:		; preds = %bb4
	%40 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%41 = trunc i32 %encoding.0 to i8		; <i8> [#uses=2]
	%42 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %41, i32 %base.0, i8* %40, i32* %pc_begin)		; <i8*> [#uses=0]
	%43 = call arm_apcscc  i32 @size_of_encoded_value(i8 zeroext %41)		; <i32> [#uses=2]
	%44 = icmp ugt i32 %43, 3		; <i1> [#uses=1]
	br i1 %44, label %bb9, label %bb7

bb7:		; preds = %bb6
	%45 = shl i32 %43, 3		; <i32> [#uses=1]
	%46 = shl i32 1, %45		; <i32> [#uses=1]
	%47 = add i32 %46, -1		; <i32> [#uses=1]
	br label %bb9

bb9:		; preds = %bb7, %bb6
	%mask.0 = phi i32 [ %47, %bb7 ], [ -1, %bb6 ]		; <i32> [#uses=1]
	%48 = load i32* %pc_begin, align 4		; <i32> [#uses=1]
	%49 = and i32 %48, %mask.0		; <i32> [#uses=1]
	%50 = icmp eq i32 %49, 0		; <i1> [#uses=1]
	br i1 %50, label %bb11, label %bb10

bb10:		; preds = %bb9, %bb5
	%51 = load %struct.fde_vector** %7, align 4		; <%struct.fde_vector*> [#uses=3]
	%52 = icmp eq %struct.fde_vector* %51, null		; <i1> [#uses=1]
	br i1 %52, label %bb11, label %bb.i

bb.i:		; preds = %bb10
	%53 = getelementptr %struct.fde_vector* %51, i32 0, i32 1		; <i32*> [#uses=2]
	%54 = load i32* %53, align 4		; <i32> [#uses=2]
	%55 = getelementptr %struct.fde_vector* %51, i32 0, i32 2, i32 %54		; <%struct.dwarf_fde**> [#uses=1]
	store %struct.dwarf_fde* %this_fde_addr.0, %struct.dwarf_fde** %55, align 4
	%56 = add i32 %54, 1		; <i32> [#uses=1]
	store i32 %56, i32* %53, align 4
	br label %bb11

bb11:		; preds = %bb.i, %bb10, %bb9, %bb5, %bb
	%base.1 = phi i32 [ %base.0, %bb.i ], [ %base.2, %bb ], [ %base.0, %bb5 ], [ %base.0, %bb9 ], [ %base.0, %bb10 ]		; <i32> [#uses=1]
	%encoding.1 = phi i32 [ %encoding.0, %bb.i ], [ %encoding.2, %bb ], [ %encoding.0, %bb5 ], [ %encoding.0, %bb9 ], [ %encoding.0, %bb10 ]		; <i32> [#uses=1]
	%last_cie.1 = phi %struct.dwarf_cie* [ %last_cie.0, %bb.i ], [ %last_cie.2, %bb ], [ %last_cie.0, %bb5 ], [ %last_cie.0, %bb9 ], [ %last_cie.0, %bb10 ]		; <%struct.dwarf_cie*> [#uses=1]
	%57 = bitcast %struct.dwarf_fde* %this_fde_addr.0 to i8*		; <i8*> [#uses=1]
	%58 = load i32* %61, align 1		; <i32> [#uses=1]
	%.sum.i = add i32 %58, 4		; <i32> [#uses=1]
	%59 = getelementptr i8* %57, i32 %.sum.i		; <i8*> [#uses=1]
	%60 = bitcast i8* %59 to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	br label %bb12

bb12:		; preds = %bb11, %bb12.preheader
	%base.2 = phi i32 [ %base.1, %bb11 ], [ %base.2.ph, %bb12.preheader ]		; <i32> [#uses=3]
	%this_fde_addr.0 = phi %struct.dwarf_fde* [ %60, %bb11 ], [ %this_fde, %bb12.preheader ]		; <%struct.dwarf_fde*> [#uses=6]
	%encoding.2 = phi i32 [ %encoding.1, %bb11 ], [ %3, %bb12.preheader ]		; <i32> [#uses=3]
	%last_cie.2 = phi %struct.dwarf_cie* [ %last_cie.1, %bb11 ], [ null, %bb12.preheader ]		; <%struct.dwarf_cie*> [#uses=4]
	%61 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 0		; <i32*> [#uses=2]
	%62 = load i32* %61, align 1		; <i32> [#uses=1]
	%63 = icmp eq i32 %62, 0		; <i1> [#uses=1]
	br i1 %63, label %return, label %bb

return:		; preds = %bb12
	call void @llvm.dbg.stoppoint(i32 659, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram154 to { }*))
	call void @llvm.dbg.stoppoint(i32 708, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram252 to { }*))
	ret void
}

define internal arm_apcscc i32 @fde_single_encoding_compare(%struct.object* nocapture %ob, %struct.dwarf_fde* %x, %struct.dwarf_fde* %y) nounwind {
entry:
	%y_ptr = alloca i32, align 4		; <i32*> [#uses=2]
	%x_ptr = alloca i32, align 4		; <i32*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram254 to { }*))
	call void @llvm.dbg.stoppoint(i32 341, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=2]
	%1 = load i32* %0		; <i32> [#uses=1]
	%2 = lshr i32 %1, 3		; <i32> [#uses=2]
	%3 = trunc i32 %2 to i8		; <i8> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 242, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%4 = icmp eq i8 %3, -1		; <i1> [#uses=1]
	br i1 %4, label %base_from_object.exit, label %bb1.i

bb1.i:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 245, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%5 = and i32 %2, 112		; <i32> [#uses=1]
	switch i32 %5, label %bb5.i [
		i32 0, label %base_from_object.exit
		i32 16, label %base_from_object.exit
		i32 32, label %bb3.i
		i32 48, label %bb4.i
		i32 80, label %base_from_object.exit
	]

bb3.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 253, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%6 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%7 = load i8** %6, align 4		; <i8*> [#uses=1]
	%8 = ptrtoint i8* %7 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb4.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%9 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	%10 = load i8** %9, align 4		; <i8*> [#uses=1]
	%11 = ptrtoint i8* %10 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb5.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

base_from_object.exit:		; preds = %bb4.i, %bb3.i, %bb1.i, %bb1.i, %bb1.i, %entry
	%12 = phi i32 [ %11, %bb4.i ], [ %8, %bb3.i ], [ 0, %entry ], [ 0, %bb1.i ], [ 0, %bb1.i ], [ 0, %bb1.i ]		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 342, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*))
	%13 = getelementptr %struct.dwarf_fde* %x, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%14 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %3, i32 %12, i8* %13, i32* %x_ptr)		; <i8*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 343, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%15 = load i32* %0		; <i32> [#uses=1]
	%16 = lshr i32 %15, 3		; <i32> [#uses=1]
	%17 = getelementptr %struct.dwarf_fde* %y, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%18 = trunc i32 %16 to i8		; <i8> [#uses=1]
	%19 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %18, i32 %12, i8* %17, i32* %y_ptr)		; <i8*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 345, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%20 = load i32* %x_ptr, align 4		; <i32> [#uses=2]
	%21 = load i32* %y_ptr, align 4		; <i32> [#uses=2]
	%22 = icmp ugt i32 %20, %21		; <i1> [#uses=1]
	br i1 %22, label %bb4, label %bb1

bb1:		; preds = %base_from_object.exit
	call void @llvm.dbg.stoppoint(i32 347, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%23 = icmp ult i32 %20, %21		; <i1> [#uses=1]
	%retval = select i1 %23, i32 -1, i32 0		; <i32> [#uses=1]
	ret i32 %retval

bb4:		; preds = %base_from_object.exit
	call void @llvm.dbg.stoppoint(i32 349, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret i32 1
}

define internal arm_apcscc i32 @fde_mixed_encoding_compare(%struct.object* nocapture %ob, %struct.dwarf_fde* %x, %struct.dwarf_fde* %y) nounwind {
entry:
	%y_ptr = alloca i32, align 4		; <i32*> [#uses=2]
	%x_ptr = alloca i32, align 4		; <i32*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram256 to { }*))
	call void @llvm.dbg.stoppoint(i32 358, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram248 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 312, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 163, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*)) nounwind
	%0 = getelementptr %struct.dwarf_fde* %x, i32 0, i32 1		; <i32*> [#uses=2]
	%1 = bitcast i32* %0 to i8*		; <i8*> [#uses=1]
	%2 = load i32* %0, align 1		; <i32> [#uses=1]
	%3 = sub i32 0, %2		; <i32> [#uses=1]
	%4 = getelementptr i8* %1, i32 %3		; <i8*> [#uses=1]
	%5 = bitcast i8* %4 to %struct.dwarf_cie*		; <%struct.dwarf_cie*> [#uses=1]
	%6 = call arm_apcscc  i32 @get_cie_encoding(%struct.dwarf_cie* %5) nounwind		; <i32> [#uses=2]
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 359, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram248 to { }*))
	%7 = trunc i32 %6 to i8		; <i8> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 242, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%8 = icmp eq i8 %7, -1		; <i1> [#uses=1]
	br i1 %8, label %base_from_object.exit, label %bb1.i

bb1.i:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 245, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%9 = and i32 %6, 112		; <i32> [#uses=1]
	switch i32 %9, label %bb5.i [
		i32 0, label %base_from_object.exit
		i32 16, label %base_from_object.exit
		i32 32, label %bb3.i
		i32 48, label %bb4.i
		i32 80, label %base_from_object.exit
	]

bb3.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 253, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%10 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%11 = load i8** %10, align 4		; <i8*> [#uses=1]
	%12 = ptrtoint i8* %11 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb4.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%13 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	%14 = load i8** %13, align 4		; <i8*> [#uses=1]
	%15 = ptrtoint i8* %14 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb5.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

base_from_object.exit:		; preds = %bb4.i, %bb3.i, %bb1.i, %bb1.i, %bb1.i, %entry
	%16 = phi i32 [ %15, %bb4.i ], [ %12, %bb3.i ], [ 0, %entry ], [ 0, %bb1.i ], [ 0, %bb1.i ], [ 0, %bb1.i ]		; <i32> [#uses=1]
	%17 = getelementptr %struct.dwarf_fde* %x, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%18 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %7, i32 %16, i8* %17, i32* %x_ptr)		; <i8*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 362, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*))
	%19 = getelementptr %struct.dwarf_fde* %y, i32 0, i32 1		; <i32*> [#uses=2]
	%20 = bitcast i32* %19 to i8*		; <i8*> [#uses=1]
	%21 = load i32* %19, align 1		; <i32> [#uses=1]
	%22 = sub i32 0, %21		; <i32> [#uses=1]
	%23 = getelementptr i8* %20, i32 %22		; <i8*> [#uses=1]
	%24 = bitcast i8* %23 to %struct.dwarf_cie*		; <%struct.dwarf_cie*> [#uses=1]
	%25 = call arm_apcscc  i32 @get_cie_encoding(%struct.dwarf_cie* %24) nounwind		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 363, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%26 = trunc i32 %25 to i8		; <i8> [#uses=2]
	%27 = icmp eq i8 %26, -1		; <i1> [#uses=1]
	br i1 %27, label %base_from_object.exit11, label %bb1.i6

bb1.i6:		; preds = %base_from_object.exit
	call void @llvm.dbg.stoppoint(i32 245, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%28 = and i32 %25, 112		; <i32> [#uses=1]
	switch i32 %28, label %bb5.i9 [
		i32 0, label %base_from_object.exit11
		i32 16, label %base_from_object.exit11
		i32 32, label %bb3.i7
		i32 48, label %bb4.i8
		i32 80, label %base_from_object.exit11
	]

bb3.i7:		; preds = %bb1.i6
	call void @llvm.dbg.stoppoint(i32 253, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%29 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%30 = load i8** %29, align 4		; <i8*> [#uses=1]
	%31 = ptrtoint i8* %30 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit11

bb4.i8:		; preds = %bb1.i6
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%32 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	%33 = load i8** %32, align 4		; <i8*> [#uses=1]
	%34 = ptrtoint i8* %33 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit11

bb5.i9:		; preds = %bb1.i6
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

base_from_object.exit11:		; preds = %bb4.i8, %bb3.i7, %bb1.i6, %bb1.i6, %bb1.i6, %base_from_object.exit
	%35 = phi i32 [ %34, %bb4.i8 ], [ %31, %bb3.i7 ], [ 0, %base_from_object.exit ], [ 0, %bb1.i6 ], [ 0, %bb1.i6 ], [ 0, %bb1.i6 ]		; <i32> [#uses=1]
	%36 = getelementptr %struct.dwarf_fde* %y, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%37 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %26, i32 %35, i8* %36, i32* %y_ptr)		; <i8*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 366, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%38 = load i32* %x_ptr, align 4		; <i32> [#uses=2]
	%39 = load i32* %y_ptr, align 4		; <i32> [#uses=2]
	%40 = icmp ugt i32 %38, %39		; <i1> [#uses=1]
	br i1 %40, label %bb4, label %bb1

bb1:		; preds = %base_from_object.exit11
	call void @llvm.dbg.stoppoint(i32 368, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%41 = icmp ult i32 %38, %39		; <i1> [#uses=1]
	%retval = select i1 %41, i32 -1, i32 0		; <i32> [#uses=1]
	ret i32 %retval

bb4:		; preds = %base_from_object.exit11
	call void @llvm.dbg.stoppoint(i32 370, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret i32 1
}

define internal arm_apcscc %struct.dwarf_fde* @linear_search_fdes(%struct.object* nocapture %ob, %struct.dwarf_fde* %this_fde, i8* %pc) nounwind {
entry:
	%pc_range = alloca i32, align 4		; <i32*> [#uses=3]
	%pc_begin = alloca i32, align 4		; <i32*> [#uses=3]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram264 to { }*))
	call void @llvm.dbg.stoppoint(i32 773, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=2]
	%1 = load i32* %0		; <i32> [#uses=1]
	%2 = lshr i32 %1, 3		; <i32> [#uses=3]
	%3 = and i32 %2, 255		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 774, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = trunc i32 %2 to i8		; <i8> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 242, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%5 = icmp eq i8 %4, -1		; <i1> [#uses=1]
	br i1 %5, label %bb13.preheader, label %bb1.i

bb1.i:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 245, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%6 = and i32 %2, 112		; <i32> [#uses=1]
	switch i32 %6, label %bb5.i [
		i32 0, label %bb13.preheader
		i32 16, label %bb13.preheader
		i32 32, label %bb3.i
		i32 48, label %bb4.i
		i32 80, label %bb13.preheader
	]

bb13.preheader:		; preds = %bb4.i, %bb3.i, %bb1.i, %bb1.i, %bb1.i, %entry
	%base.2.ph = phi i32 [ %15, %bb4.i ], [ %12, %bb3.i ], [ 0, %entry ], [ 0, %bb1.i ], [ 0, %bb1.i ], [ 0, %bb1.i ]		; <i32> [#uses=1]
	%7 = ptrtoint i8* %pc to i32		; <i32> [#uses=1]
	%8 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%9 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	br label %bb13

bb3.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 253, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%10 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%11 = load i8** %10, align 4		; <i8*> [#uses=1]
	%12 = ptrtoint i8* %11 to i32		; <i32> [#uses=1]
	br label %bb13.preheader

bb4.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%13 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	%14 = load i8** %13, align 4		; <i8*> [#uses=1]
	%15 = ptrtoint i8* %14 to i32		; <i32> [#uses=1]
	br label %bb13.preheader

bb5.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb:		; preds = %bb13
	%16 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 1		; <i32*> [#uses=2]
	%17 = load i32* %16, align 1		; <i32> [#uses=2]
	%18 = icmp eq i32 %17, 0		; <i1> [#uses=1]
	br i1 %18, label %bb12, label %bb1

bb1:		; preds = %bb
	%19 = load i32* %0		; <i32> [#uses=1]
	%20 = and i32 %19, 4		; <i32> [#uses=1]
	%21 = icmp eq i32 %20, 0		; <i1> [#uses=1]
	br i1 %21, label %bb4, label %bb2

bb2:		; preds = %bb1
	%22 = bitcast i32* %16 to i8*		; <i8*> [#uses=1]
	%23 = sub i32 0, %17		; <i32> [#uses=1]
	%24 = getelementptr i8* %22, i32 %23		; <i8*> [#uses=1]
	%25 = bitcast i8* %24 to %struct.dwarf_cie*		; <%struct.dwarf_cie*> [#uses=8]
	%26 = icmp eq %struct.dwarf_cie* %25, %last_cie.2		; <i1> [#uses=1]
	br i1 %26, label %bb4, label %bb3

bb3:		; preds = %bb2
	%27 = call arm_apcscc  i32 @get_cie_encoding(%struct.dwarf_cie* %25)		; <i32> [#uses=8]
	%28 = trunc i32 %27 to i8		; <i8> [#uses=1]
	%29 = icmp eq i8 %28, -1		; <i1> [#uses=1]
	br i1 %29, label %bb4, label %bb1.i17

bb1.i17:		; preds = %bb3
	%30 = and i32 %27, 112		; <i32> [#uses=1]
	switch i32 %30, label %bb5.i20 [
		i32 0, label %bb4
		i32 16, label %bb4
		i32 32, label %bb3.i18
		i32 48, label %bb4.i19
		i32 80, label %bb4
	]

bb3.i18:		; preds = %bb1.i17
	%31 = load i8** %8, align 4		; <i8*> [#uses=1]
	%32 = ptrtoint i8* %31 to i32		; <i32> [#uses=1]
	br label %bb4

bb4.i19:		; preds = %bb1.i17
	%33 = load i8** %9, align 4		; <i8*> [#uses=1]
	%34 = ptrtoint i8* %33 to i32		; <i32> [#uses=1]
	br label %bb4

bb5.i20:		; preds = %bb1.i17
	call void @llvm.dbg.stoppoint(i32 776, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram154 to { }*))
	call void @llvm.dbg.stoppoint(i32 789, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*))
	call void @llvm.dbg.stoppoint(i32 790, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*))
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb4:		; preds = %bb4.i19, %bb3.i18, %bb1.i17, %bb1.i17, %bb1.i17, %bb3, %bb2, %bb1
	%base.0 = phi i32 [ %34, %bb4.i19 ], [ %32, %bb3.i18 ], [ %base.2, %bb1 ], [ %base.2, %bb2 ], [ 0, %bb3 ], [ 0, %bb1.i17 ], [ 0, %bb1.i17 ], [ 0, %bb1.i17 ]		; <i32> [#uses=4]
	%encoding.0 = phi i32 [ %27, %bb3.i18 ], [ %27, %bb4.i19 ], [ %encoding.2, %bb1 ], [ %encoding.2, %bb2 ], [ %27, %bb3 ], [ %27, %bb1.i17 ], [ %27, %bb1.i17 ], [ %27, %bb1.i17 ]		; <i32> [#uses=5]
	%last_cie.0 = phi %struct.dwarf_cie* [ %25, %bb3.i18 ], [ %25, %bb4.i19 ], [ %last_cie.2, %bb1 ], [ %last_cie.2, %bb2 ], [ %25, %bb3 ], [ %25, %bb1.i17 ], [ %25, %bb1.i17 ], [ %25, %bb1.i17 ]		; <%struct.dwarf_cie*> [#uses=3]
	%35 = icmp eq i32 %encoding.0, 0		; <i1> [#uses=1]
	br i1 %35, label %bb5, label %bb6

bb5:		; preds = %bb4
	%36 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 2		; <[0 x i8]*> [#uses=1]
	%37 = bitcast [0 x i8]* %36 to i32*		; <i32*> [#uses=2]
	%38 = load i32* %37, align 4		; <i32> [#uses=3]
	store i32 %38, i32* %pc_begin, align 4
	%39 = getelementptr i32* %37, i32 1		; <i32*> [#uses=1]
	%40 = load i32* %39, align 4		; <i32> [#uses=1]
	store i32 %40, i32* %pc_range, align 4
	%41 = icmp eq i32 %38, 0		; <i1> [#uses=1]
	br i1 %41, label %bb12, label %bb10

bb6:		; preds = %bb4
	%42 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%43 = trunc i32 %encoding.0 to i8		; <i8> [#uses=3]
	%44 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %43, i32 %base.0, i8* %42, i32* %pc_begin)		; <i8*> [#uses=1]
	%45 = and i8 %43, 15		; <i8> [#uses=1]
	%46 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %45, i32 0, i8* %44, i32* %pc_range)		; <i8*> [#uses=0]
	%47 = call arm_apcscc  i32 @size_of_encoded_value(i8 zeroext %43)		; <i32> [#uses=2]
	%48 = icmp ugt i32 %47, 3		; <i1> [#uses=1]
	br i1 %48, label %bb9, label %bb7

bb7:		; preds = %bb6
	%49 = shl i32 %47, 3		; <i32> [#uses=1]
	%50 = shl i32 1, %49		; <i32> [#uses=1]
	%51 = add i32 %50, -1		; <i32> [#uses=1]
	br label %bb9

bb9:		; preds = %bb7, %bb6
	%mask.0 = phi i32 [ %51, %bb7 ], [ -1, %bb6 ]		; <i32> [#uses=1]
	%52 = load i32* %pc_begin, align 4		; <i32> [#uses=2]
	%53 = and i32 %52, %mask.0		; <i32> [#uses=1]
	%54 = icmp eq i32 %53, 0		; <i1> [#uses=1]
	br i1 %54, label %bb12, label %bb10

bb10:		; preds = %bb9, %bb5
	%55 = phi i32 [ %38, %bb5 ], [ %52, %bb9 ]		; <i32> [#uses=1]
	%56 = sub i32 %7, %55		; <i32> [#uses=1]
	%57 = load i32* %pc_range, align 4		; <i32> [#uses=1]
	%58 = icmp ult i32 %56, %57		; <i1> [#uses=1]
	br i1 %58, label %bb15, label %bb12

bb12:		; preds = %bb10, %bb9, %bb5, %bb
	%base.1 = phi i32 [ %base.2, %bb ], [ %base.0, %bb5 ], [ %base.0, %bb9 ], [ %base.0, %bb10 ]		; <i32> [#uses=1]
	%encoding.1 = phi i32 [ %encoding.2, %bb ], [ %encoding.0, %bb5 ], [ %encoding.0, %bb9 ], [ %encoding.0, %bb10 ]		; <i32> [#uses=1]
	%last_cie.1 = phi %struct.dwarf_cie* [ %last_cie.2, %bb ], [ %last_cie.0, %bb5 ], [ %last_cie.0, %bb9 ], [ %last_cie.0, %bb10 ]		; <%struct.dwarf_cie*> [#uses=1]
	%59 = bitcast %struct.dwarf_fde* %this_fde_addr.0 to i8*		; <i8*> [#uses=1]
	%60 = load i32* %63, align 1		; <i32> [#uses=1]
	%.sum.i = add i32 %60, 4		; <i32> [#uses=1]
	%61 = getelementptr i8* %59, i32 %.sum.i		; <i8*> [#uses=1]
	%62 = bitcast i8* %61 to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	br label %bb13

bb13:		; preds = %bb12, %bb13.preheader
	%base.2 = phi i32 [ %base.1, %bb12 ], [ %base.2.ph, %bb13.preheader ]		; <i32> [#uses=3]
	%encoding.2 = phi i32 [ %encoding.1, %bb12 ], [ %3, %bb13.preheader ]		; <i32> [#uses=3]
	%last_cie.2 = phi %struct.dwarf_cie* [ %last_cie.1, %bb12 ], [ null, %bb13.preheader ]		; <%struct.dwarf_cie*> [#uses=4]
	%this_fde_addr.0 = phi %struct.dwarf_fde* [ %62, %bb12 ], [ %this_fde, %bb13.preheader ]		; <%struct.dwarf_fde*> [#uses=6]
	%63 = getelementptr %struct.dwarf_fde* %this_fde_addr.0, i32 0, i32 0		; <i32*> [#uses=2]
	%64 = load i32* %63, align 1		; <i32> [#uses=1]
	%65 = icmp eq i32 %64, 0		; <i1> [#uses=1]
	br i1 %65, label %bb15, label %bb

bb15:		; preds = %bb13, %bb10
	%.0 = phi %struct.dwarf_fde* [ %this_fde_addr.0, %bb10 ], [ null, %bb13 ]		; <%struct.dwarf_fde*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 776, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram154 to { }*))
	call void @llvm.dbg.stoppoint(i32 832, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram264 to { }*))
	ret %struct.dwarf_fde* %.0
}

define arm_apcscc void @__register_frame_table(i8* %begin) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram272 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 157, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = malloc %struct.object		; <%struct.object*> [#uses=7]
	tail call void @llvm.dbg.stoppoint(i32 158, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram184 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 151, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram180 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 131, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%1 = getelementptr %struct.object* %0, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* inttoptr (i64 4294967295 to i8*), i8** %1, align 4
	tail call void @llvm.dbg.stoppoint(i32 132, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = getelementptr %struct.object* %0, i32 0, i32 1		; <i8**> [#uses=1]
	store i8* null, i8** %2, align 4
	tail call void @llvm.dbg.stoppoint(i32 133, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = getelementptr %struct.object* %0, i32 0, i32 2		; <i8**> [#uses=1]
	store i8* null, i8** %3, align 4
	tail call void @llvm.dbg.stoppoint(i32 134, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = getelementptr %struct.object* %0, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	%.c.i.i = bitcast i8* %begin to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	store %struct.dwarf_fde* %.c.i.i, %struct.dwarf_fde** %4
	tail call void @llvm.dbg.stoppoint(i32 137, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = getelementptr %struct.object* %0, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	store i32 2042, i32* %5
	tail call void @llvm.dbg.stoppoint(i32 140, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 142, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	%6 = load %struct.object** @unseen_objects, align 4		; <%struct.object*> [#uses=1]
	%7 = getelementptr %struct.object* %0, i32 0, i32 5		; <%struct.object**> [#uses=1]
	store %struct.object* %6, %struct.object** %7, align 4
	tail call void @llvm.dbg.stoppoint(i32 143, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store %struct.object* %0, %struct.object** @unseen_objects, align 4
	tail call void @llvm.dbg.stoppoint(i32 145, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 146, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 152, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram180 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 159, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram184 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram272 to { }*))
	ret void
}

define arm_apcscc void @__register_frame(i8* %begin) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram274 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 116, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = bitcast i8* %begin to i32*		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb

bb:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 119, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = malloc %struct.object		; <%struct.object*> [#uses=7]
	tail call void @llvm.dbg.stoppoint(i32 120, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram176 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 107, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram172 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 82, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = icmp eq i8* %begin, null		; <i1> [#uses=1]
	br i1 %4, label %__register_frame_info.exit, label %bb.i.i

bb.i.i:		; preds = %bb
	tail call void @llvm.dbg.stoppoint(i32 85, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = getelementptr %struct.object* %3, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* inttoptr (i64 4294967295 to i8*), i8** %5, align 4
	tail call void @llvm.dbg.stoppoint(i32 86, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = getelementptr %struct.object* %3, i32 0, i32 1		; <i8**> [#uses=1]
	store i8* null, i8** %6, align 4
	tail call void @llvm.dbg.stoppoint(i32 87, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%7 = getelementptr %struct.object* %3, i32 0, i32 2		; <i8**> [#uses=1]
	store i8* null, i8** %7, align 4
	tail call void @llvm.dbg.stoppoint(i32 88, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%8 = bitcast i8* %begin to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	%9 = getelementptr %struct.object* %3, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	store %struct.dwarf_fde* %8, %struct.dwarf_fde** %9, align 4
	tail call void @llvm.dbg.stoppoint(i32 90, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%10 = getelementptr %struct.object* %3, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	store i32 2040, i32* %10
	tail call void @llvm.dbg.stoppoint(i32 96, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 98, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	%11 = load %struct.object** @unseen_objects, align 4		; <%struct.object*> [#uses=1]
	%12 = getelementptr %struct.object* %3, i32 0, i32 5		; <%struct.object**> [#uses=1]
	store %struct.object* %11, %struct.object** %12, align 4
	tail call void @llvm.dbg.stoppoint(i32 99, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store %struct.object* %3, %struct.object** @unseen_objects, align 4
	tail call void @llvm.dbg.stoppoint(i32 101, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 233, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit159 to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram274 to { }*))
	ret void

__register_frame_info.exit:		; preds = %bb
	ret void

return:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 120, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
}

define internal arm_apcscc %struct.dwarf_fde* @search_object(%struct.object* %ob, i8* %pc) {
entry:
	%pc_range.i33 = alloca i32, align 4		; <i32*> [#uses=2]
	%pc_begin.i34 = alloca i32, align 4		; <i32*> [#uses=2]
	%pc_range.i = alloca i32, align 4		; <i32*> [#uses=2]
	%pc_begin.i = alloca i32, align 4		; <i32*> [#uses=2]
	%accu.i = alloca %struct.fde_accumulator, align 4		; <%struct.fde_accumulator*> [#uses=4]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram284 to { }*))
	call void @llvm.dbg.stoppoint(i32 931, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = getelementptr %struct.object* %ob, i32 0, i32 4, i32 0		; <i32*> [#uses=7]
	%1 = load i32* %0		; <i32> [#uses=6]
	%2 = and i32 %1, 1		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %bb, label %bb2

bb:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 933, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram282 to { }*))
	call void @llvm.dbg.stoppoint(i32 721, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = lshr i32 %1, 11		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 722, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%5 = icmp eq i32 %4, 0		; <i1> [#uses=1]
	br i1 %5, label %bb.i, label %bb8.i

bb.i:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 724, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%6 = and i32 %1, 2		; <i32> [#uses=1]
	%7 = icmp eq i32 %6, 0		; <i1> [#uses=1]
	%8 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=2]
	br i1 %7, label %bb5.i, label %bb2.i

bb2.i:		; preds = %bb.i
	call void @llvm.dbg.stoppoint(i32 726, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%9 = load %struct.dwarf_fde** %8		; <%struct.dwarf_fde*> [#uses=3]
	%10 = bitcast %struct.dwarf_fde* %9 to i8*		; <i8*> [#uses=1]
	%11 = bitcast %struct.dwarf_fde* %9 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 727, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%12 = load %struct.dwarf_fde** %11, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%13 = icmp eq %struct.dwarf_fde* %12, null		; <i1> [#uses=1]
	br i1 %13, label %bb6.i, label %bb3.i

bb3.i:		; preds = %bb3.i, %bb2.i
	%indvar.i = phi i32 [ 0, %bb2.i ], [ %indvar.next.i, %bb3.i ]		; <i32> [#uses=3]
	%count.221.i = phi i32 [ 0, %bb2.i ], [ %16, %bb3.i ]		; <i32> [#uses=1]
	%scevgep = getelementptr %struct.dwarf_fde* %9, i32 0, i32 1		; <i32*> [#uses=1]
	%scevgep60 = getelementptr i32* %scevgep, i32 %indvar.i		; <i32*> [#uses=1]
	%scevgep2728.i = bitcast i32* %scevgep60 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%tmp = shl i32 %indvar.i, 2		; <i32> [#uses=1]
	%scevgep62 = getelementptr i8* %10, i32 %tmp		; <i8*> [#uses=1]
	%p1.020.i = bitcast i8* %scevgep62 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%14 = load %struct.dwarf_fde** %p1.020.i, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%15 = call arm_apcscc  i32 @classify_object_over_fdes(%struct.object* %ob, %struct.dwarf_fde* %14)		; <i32> [#uses=1]
	%16 = add i32 %15, %count.221.i		; <i32> [#uses=2]
	%17 = load %struct.dwarf_fde** %scevgep2728.i, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%18 = icmp eq %struct.dwarf_fde* %17, null		; <i1> [#uses=1]
	%indvar.next.i = add i32 %indvar.i, 1		; <i32> [#uses=1]
	br i1 %18, label %bb6.i, label %bb3.i

bb5.i:		; preds = %bb.i
	call void @llvm.dbg.stoppoint(i32 731, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%19 = load %struct.dwarf_fde** %8, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%20 = call arm_apcscc  i32 @classify_object_over_fdes(%struct.object* %ob, %struct.dwarf_fde* %19)		; <i32> [#uses=1]
	br label %bb6.i

bb6.i:		; preds = %bb5.i, %bb3.i, %bb2.i
	%count.0.i = phi i32 [ %20, %bb5.i ], [ 0, %bb2.i ], [ %16, %bb3.i ]		; <i32> [#uses=5]
	call void @llvm.dbg.stoppoint(i32 738, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%21 = load i32* %0		; <i32> [#uses=1]
	%22 = shl i32 %count.0.i, 11		; <i32> [#uses=1]
	%23 = and i32 %21, 2047		; <i32> [#uses=4]
	%24 = or i32 %23, %22		; <i32> [#uses=3]
	store i32 %24, i32* %0
	call void @llvm.dbg.stoppoint(i32 739, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%25 = and i32 %count.0.i, 2097151		; <i32> [#uses=1]
	%26 = icmp eq i32 %25, %count.0.i		; <i1> [#uses=1]
	br i1 %26, label %bb8.i, label %bb7.i

bb7.i:		; preds = %bb6.i
	call void @llvm.dbg.stoppoint(i32 740, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 %23, i32* %0
	br label %bb8.i

bb8.i:		; preds = %bb7.i, %bb6.i, %bb
	%.rle121 = phi i32 [ %23, %bb7.i ], [ %1, %bb ], [ %24, %bb6.i ]		; <i32> [#uses=2]
	%27 = phi i32 [ %23, %bb7.i ], [ %1, %bb ], [ %24, %bb6.i ]		; <i32> [#uses=1]
	%count.1.i = phi i32 [ %count.0.i, %bb7.i ], [ %4, %bb ], [ %count.0.i, %bb6.i ]		; <i32> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 743, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram268 to { }*))
	call void @llvm.dbg.stoppoint(i32 397, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%28 = icmp eq i32 %count.1.i, 0		; <i1> [#uses=1]
	br i1 %28, label %init_object.exit, label %bb1.i.i

bb1.i.i:		; preds = %bb8.i
	call void @llvm.dbg.stoppoint(i32 400, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%29 = shl i32 %count.1.i, 2		; <i32> [#uses=1]
	%30 = add i32 %29, 8		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 401, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%31 = malloc i8, i32 %30		; <i8*> [#uses=3]
	%32 = bitcast i8* %31 to %struct.fde_vector*		; <%struct.fde_vector*> [#uses=1]
	%33 = getelementptr %struct.fde_accumulator* %accu.i, i32 0, i32 0		; <%struct.fde_vector**> [#uses=6]
	store %struct.fde_vector* %32, %struct.fde_vector** %33, align 4
	%34 = icmp eq i8* %31, null		; <i1> [#uses=1]
	br i1 %34, label %init_object.exit, label %bb2.i.i

bb2.i.i:		; preds = %bb1.i.i
	call void @llvm.dbg.stoppoint(i32 403, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%35 = getelementptr i8* %31, i32 4		; <i8*> [#uses=1]
	%36 = bitcast i8* %35 to i32*		; <i32*> [#uses=1]
	store i32 0, i32* %36, align 4
	call void @llvm.dbg.stoppoint(i32 404, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%37 = malloc i8, i32 %30		; <i8*> [#uses=3]
	%38 = bitcast i8* %37 to %struct.fde_vector*		; <%struct.fde_vector*> [#uses=1]
	%39 = getelementptr %struct.fde_accumulator* %accu.i, i32 0, i32 1		; <%struct.fde_vector**> [#uses=5]
	store %struct.fde_vector* %38, %struct.fde_vector** %39, align 4
	%40 = icmp eq i8* %37, null		; <i1> [#uses=1]
	br i1 %40, label %bb9.i, label %start_fde_sort.exit.thread.i

start_fde_sort.exit.thread.i:		; preds = %bb2.i.i
	call void @llvm.dbg.stoppoint(i32 405, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%41 = getelementptr i8* %37, i32 4		; <i8*> [#uses=1]
	%42 = bitcast i8* %41 to i32*		; <i32*> [#uses=1]
	store i32 0, i32* %42, align 4
	br label %bb9.i

bb9.i:		; preds = %start_fde_sort.exit.thread.i, %bb2.i.i
	call void @llvm.dbg.stoppoint(i32 746, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%43 = and i32 %27, 2		; <i32> [#uses=1]
	%44 = icmp eq i32 %43, 0		; <i1> [#uses=1]
	%45 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=4]
	br i1 %44, label %bb13.i, label %bb10.i

bb10.i:		; preds = %bb9.i
	call void @llvm.dbg.stoppoint(i32 749, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%46 = load %struct.dwarf_fde** %45		; <%struct.dwarf_fde*> [#uses=3]
	%47 = bitcast %struct.dwarf_fde* %46 to i8*		; <i8*> [#uses=1]
	%48 = bitcast %struct.dwarf_fde* %46 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%49 = load %struct.dwarf_fde** %48, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%50 = icmp eq %struct.dwarf_fde* %49, null		; <i1> [#uses=1]
	br i1 %50, label %bb14.i, label %bb11.i

bb11.i:		; preds = %bb11.i, %bb10.i
	%indvar29.i = phi i32 [ 0, %bb10.i ], [ %indvar.next30.i, %bb11.i ]		; <i32> [#uses=3]
	%scevgep64 = getelementptr %struct.dwarf_fde* %46, i32 0, i32 1		; <i32*> [#uses=1]
	%scevgep65 = getelementptr i32* %scevgep64, i32 %indvar29.i		; <i32*> [#uses=1]
	%scevgep3536.i = bitcast i32* %scevgep65 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%tmp67 = shl i32 %indvar29.i, 2		; <i32> [#uses=1]
	%scevgep68 = getelementptr i8* %47, i32 %tmp67		; <i8*> [#uses=1]
	%p.023.i = bitcast i8* %scevgep68 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%51 = load %struct.dwarf_fde** %p.023.i, align 4		; <%struct.dwarf_fde*> [#uses=1]
	call arm_apcscc  void @add_fdes(%struct.object* %ob, %struct.fde_accumulator* %accu.i, %struct.dwarf_fde* %51)
	%52 = load %struct.dwarf_fde** %scevgep3536.i, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%53 = icmp eq %struct.dwarf_fde* %52, null		; <i1> [#uses=1]
	%indvar.next30.i = add i32 %indvar29.i, 1		; <i32> [#uses=1]
	br i1 %53, label %bb14.i, label %bb11.i

bb13.i:		; preds = %bb9.i
	call void @llvm.dbg.stoppoint(i32 753, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%54 = load %struct.dwarf_fde** %45, align 4		; <%struct.dwarf_fde*> [#uses=1]
	call arm_apcscc  void @add_fdes(%struct.object* %ob, %struct.fde_accumulator* %accu.i, %struct.dwarf_fde* %54)
	br label %bb14.i

bb14.i:		; preds = %bb13.i, %bb11.i, %bb10.i
	call void @llvm.dbg.stoppoint(i32 755, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram278 to { }*))
	call void @llvm.dbg.stoppoint(i32 567, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%55 = load %struct.fde_vector** %33, align 4		; <%struct.fde_vector*> [#uses=8]
	%56 = icmp eq %struct.fde_vector* %55, null		; <i1> [#uses=1]
	br i1 %56, label %bb2.i17.i, label %bb.i.i

bb.i.i:		; preds = %bb14.i
	%57 = getelementptr %struct.fde_vector* %55, i32 0, i32 1		; <i32*> [#uses=1]
	%58 = load i32* %57, align 4		; <i32> [#uses=1]
	%59 = icmp eq i32 %58, %count.1.i		; <i1> [#uses=1]
	br i1 %59, label %bb2.i17.i, label %bb1.i16.i

bb1.i16.i:		; preds = %bb.i.i
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb2.i17.i:		; preds = %bb.i.i, %bb14.i
	call void @llvm.dbg.stoppoint(i32 569, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%60 = load i32* %0		; <i32> [#uses=2]
	%61 = and i32 %60, 4		; <i32> [#uses=1]
	%62 = icmp eq i32 %61, 0		; <i1> [#uses=1]
	br i1 %62, label %bb4.i.i, label %bb7.i.i

bb4.i.i:		; preds = %bb2.i17.i
	call void @llvm.dbg.stoppoint(i32 571, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%63 = and i32 %60, 2040		; <i32> [#uses=1]
	%64 = icmp eq i32 %63, 0		; <i1> [#uses=1]
	br i1 %64, label %bb7.i.i, label %bb6.i18.i

bb6.i18.i:		; preds = %bb4.i.i
	call void @llvm.dbg.stoppoint(i32 574, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb7.i.i

bb7.i.i:		; preds = %bb6.i18.i, %bb4.i.i, %bb2.i17.i
	%fde_compare.0.i.i = phi i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)* [ @fde_single_encoding_compare, %bb6.i18.i ], [ @fde_mixed_encoding_compare, %bb2.i17.i ], [ @fde_unencoded_compare, %bb4.i.i ]		; <i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)*> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 576, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%65 = load %struct.fde_vector** %39, align 4		; <%struct.fde_vector*> [#uses=5]
	%66 = icmp eq %struct.fde_vector* %65, null		; <i1> [#uses=1]
	br i1 %66, label %bb11.i.i, label %bb8.i.i

bb8.i.i:		; preds = %bb7.i.i
	call void @llvm.dbg.stoppoint(i32 578, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram208 to { }*))
	call void @llvm.dbg.stoppoint(i32 436, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%67 = getelementptr %struct.fde_vector* %55, i32 0, i32 1		; <i32*> [#uses=2]
	%68 = load i32* %67, align 4		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 445, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%69 = icmp eq i32 %68, 0		; <i1> [#uses=2]
	br i1 %69, label %fde_split.exit.i.i, label %bb.nph20.i.i.i

bb1.i.i.i:		; preds = %bb3.i.i.i
	%70 = ptrtoint %struct.dwarf_fde** %probe.0.i.i.i to i32		; <i32> [#uses=1]
	%71 = sub i32 %70, %84		; <i32> [#uses=1]
	%72 = ashr i32 %71, 2		; <i32> [#uses=1]
	%73 = getelementptr %struct.fde_vector* %65, i32 0, i32 2, i32 %72		; <%struct.dwarf_fde**> [#uses=2]
	%74 = load %struct.dwarf_fde** %73, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%75 = bitcast %struct.dwarf_fde* %74 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=2]
	store %struct.dwarf_fde* null, %struct.dwarf_fde** %73, align 4
	br label %bb2.i.i.i

bb2.i.i.i:		; preds = %bb2.preheader.i.i.i, %bb1.i.i.i
	%probe.0.i.i.i = phi %struct.dwarf_fde** [ %75, %bb1.i.i.i ], [ %chain_end.119.i.i.i, %bb2.preheader.i.i.i ]		; <%struct.dwarf_fde**> [#uses=3]
	%chain_end.0.i.i.i = phi %struct.dwarf_fde** [ %75, %bb1.i.i.i ], [ %chain_end.119.i.i.i, %bb2.preheader.i.i.i ]		; <%struct.dwarf_fde**> [#uses=1]
	%76 = icmp eq %struct.dwarf_fde** %probe.0.i.i.i, @marker.2702		; <i1> [#uses=1]
	br i1 %76, label %bb4.i.i.i, label %bb3.i.i.i

bb3.i.i.i:		; preds = %bb2.i.i.i
	%scevgep333437.i.i.i = load i8** %scevgep33.i.i.i		; <i8*> [#uses=1]
	%77 = bitcast i8* %scevgep333437.i.i.i to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	%78 = load %struct.dwarf_fde** %probe.0.i.i.i, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%79 = call arm_apcscc  i32 %fde_compare.0.i.i(%struct.object* %ob, %struct.dwarf_fde* %77, %struct.dwarf_fde* %78)		; <i32> [#uses=1]
	%80 = icmp slt i32 %79, 0		; <i1> [#uses=1]
	br i1 %80, label %bb1.i.i.i, label %bb4.i.i.i

bb4.i.i.i:		; preds = %bb3.i.i.i, %bb2.i.i.i
	%.c.i.i.i = bitcast %struct.dwarf_fde** %chain_end.0.i.i.i to i8*		; <i8*> [#uses=1]
	store i8* %.c.i.i.i, i8** %scevgep30.i.i.i
	%81 = getelementptr %struct.fde_vector* %55, i32 0, i32 2, i32 %i.018.i.i.i		; <%struct.dwarf_fde**> [#uses=1]
	%82 = add i32 %i.018.i.i.i, 1		; <i32> [#uses=2]
	%exitcond77 = icmp eq i32 %82, %umax76		; <i1> [#uses=1]
	br i1 %exitcond77, label %bb11.loopexit.i.i.i, label %bb2.preheader.i.i.i

bb.nph20.i.i.i:		; preds = %bb8.i.i
	%83 = getelementptr %struct.fde_vector* %55, i32 0, i32 2		; <[0 x %struct.dwarf_fde*]*> [#uses=1]
	%84 = ptrtoint [0 x %struct.dwarf_fde*]* %83 to i32		; <i32> [#uses=1]
	%tmp75 = icmp ugt i32 %68, 1		; <i1> [#uses=1]
	%umax76 = select i1 %tmp75, i32 %68, i32 1		; <i32> [#uses=2]
	br label %bb2.preheader.i.i.i

bb2.preheader.i.i.i:		; preds = %bb.nph20.i.i.i, %bb4.i.i.i
	%chain_end.119.i.i.i = phi %struct.dwarf_fde** [ @marker.2702, %bb.nph20.i.i.i ], [ %81, %bb4.i.i.i ]		; <%struct.dwarf_fde**> [#uses=2]
	%i.018.i.i.i = phi i32 [ 0, %bb.nph20.i.i.i ], [ %82, %bb4.i.i.i ]		; <i32> [#uses=4]
	%scevgep78 = getelementptr %struct.fde_vector* %55, i32 1, i32 0		; <i8**> [#uses=2]
	%scevgep33.i.i.i = getelementptr i8** %scevgep78, i32 %i.018.i.i.i		; <i8**> [#uses=1]
	%scevgep80 = getelementptr %struct.fde_vector* %65, i32 1, i32 0		; <i8**> [#uses=2]
	%scevgep30.i.i.i = getelementptr i8** %scevgep80, i32 %i.018.i.i.i		; <i8**> [#uses=1]
	br label %bb2.i.i.i

bb7.i.i.i:		; preds = %bb11.loopexit.i.i.i, %bb10.i.i.i
	%i.115.i.i.i = phi i32 [ %91, %bb10.i.i.i ], [ 0, %bb11.loopexit.i.i.i ]		; <i32> [#uses=3]
	%j.114.i.i.i = phi i32 [ %j.0.i.i.i, %bb10.i.i.i ], [ 0, %bb11.loopexit.i.i.i ]		; <i32> [#uses=3]
	%k.113.i.i.i = phi i32 [ %k.0.i.i.i, %bb10.i.i.i ], [ 0, %bb11.loopexit.i.i.i ]		; <i32> [#uses=3]
	%scevgep24.i.i.i = getelementptr i8** %scevgep80, i32 %i.115.i.i.i		; <i8**> [#uses=1]
	%scevgep21.i.i.i = getelementptr i8** %scevgep78, i32 %i.115.i.i.i		; <i8**> [#uses=1]
	%scevgep242536.i.i.i = load i8** %scevgep24.i.i.i		; <i8*> [#uses=1]
	%85 = icmp eq i8* %scevgep242536.i.i.i, null		; <i1> [#uses=1]
	%scevgep212235.i.i.i = load i8** %scevgep21.i.i.i		; <i8*> [#uses=1]
	%86 = bitcast i8* %scevgep212235.i.i.i to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=2]
	br i1 %85, label %bb9.i.i.i, label %bb8.i.i.i

bb8.i.i.i:		; preds = %bb7.i.i.i
	%87 = getelementptr %struct.fde_vector* %55, i32 0, i32 2, i32 %j.114.i.i.i		; <%struct.dwarf_fde**> [#uses=1]
	store %struct.dwarf_fde* %86, %struct.dwarf_fde** %87, align 4
	%88 = add i32 %j.114.i.i.i, 1		; <i32> [#uses=1]
	br label %bb10.i.i.i

bb9.i.i.i:		; preds = %bb7.i.i.i
	%89 = getelementptr %struct.fde_vector* %65, i32 0, i32 2, i32 %k.113.i.i.i		; <%struct.dwarf_fde**> [#uses=1]
	store %struct.dwarf_fde* %86, %struct.dwarf_fde** %89, align 4
	%90 = add i32 %k.113.i.i.i, 1		; <i32> [#uses=1]
	br label %bb10.i.i.i

bb10.i.i.i:		; preds = %bb9.i.i.i, %bb8.i.i.i
	%k.0.i.i.i = phi i32 [ %k.113.i.i.i, %bb8.i.i.i ], [ %90, %bb9.i.i.i ]		; <i32> [#uses=2]
	%j.0.i.i.i = phi i32 [ %88, %bb8.i.i.i ], [ %j.114.i.i.i, %bb9.i.i.i ]		; <i32> [#uses=2]
	%91 = add i32 %i.115.i.i.i, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %91, %umax76		; <i1> [#uses=1]
	br i1 %exitcond, label %fde_split.exit.i.i, label %bb7.i.i.i

bb11.loopexit.i.i.i:		; preds = %bb4.i.i.i
	call void @llvm.dbg.stoppoint(i32 463, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br i1 %69, label %fde_split.exit.i.i, label %bb7.i.i.i

fde_split.exit.i.i:		; preds = %bb11.loopexit.i.i.i, %bb10.i.i.i, %bb8.i.i
	%j.1.lcssa.i.i.i = phi i32 [ 0, %bb8.i.i ], [ 0, %bb11.loopexit.i.i.i ], [ %j.0.i.i.i, %bb10.i.i.i ]		; <i32> [#uses=1]
	%k.1.lcssa.i.i.i = phi i32 [ 0, %bb8.i.i ], [ 0, %bb11.loopexit.i.i.i ], [ %k.0.i.i.i, %bb10.i.i.i ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 468, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 %j.1.lcssa.i.i.i, i32* %67, align 4
	call void @llvm.dbg.stoppoint(i32 469, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%92 = getelementptr %struct.fde_vector* %65, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %k.1.lcssa.i.i.i, i32* %92, align 4
	call void @llvm.dbg.stoppoint(i32 579, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram208 to { }*))
	%93 = load %struct.fde_vector** %33, align 4		; <%struct.fde_vector*> [#uses=1]
	%94 = getelementptr %struct.fde_vector* %93, i32 0, i32 1		; <i32*> [#uses=1]
	%95 = load i32* %94, align 4		; <i32> [#uses=1]
	%96 = load %struct.fde_vector** %39, align 4		; <%struct.fde_vector*> [#uses=2]
	%97 = getelementptr %struct.fde_vector* %96, i32 0, i32 1		; <i32*> [#uses=1]
	%98 = load i32* %97, align 4		; <i32> [#uses=1]
	%99 = add i32 %98, %95		; <i32> [#uses=1]
	%100 = icmp eq i32 %99, %count.1.i		; <i1> [#uses=1]
	br i1 %100, label %bb10.i.i, label %bb9.i.i

bb9.i.i:		; preds = %fde_split.exit.i.i
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb10.i.i:		; preds = %fde_split.exit.i.i
	call void @llvm.dbg.stoppoint(i32 580, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call arm_apcscc  void @frame_heapsort(%struct.object* %ob, i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)* %fde_compare.0.i.i, %struct.fde_vector* %96)
	call void @llvm.dbg.stoppoint(i32 581, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%101 = load %struct.fde_vector** %33, align 4		; <%struct.fde_vector*> [#uses=3]
	%v19.i.i.i = bitcast %struct.fde_vector* %101 to i8*		; <i8*> [#uses=2]
	%102 = load %struct.fde_vector** %39, align 4		; <%struct.fde_vector*> [#uses=3]
	%v230.i.i.i = bitcast %struct.fde_vector* %102 to i8*		; <i8*> [#uses=1]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram219 to { }*))
	call void @llvm.dbg.stoppoint(i32 542, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%103 = getelementptr %struct.fde_vector* %102, i32 0, i32 1		; <i32*> [#uses=2]
	%104 = load i32* %103, align 4		; <i32> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 543, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%105 = icmp eq i32 %104, 0		; <i1> [#uses=1]
	br i1 %105, label %fde_merge.exit.i.i, label %bb.i.i.i

bb.i.i.i:		; preds = %bb10.i.i
	%106 = getelementptr %struct.fde_vector* %101, i32 0, i32 1		; <i32*> [#uses=3]
	%107 = load i32* %106, align 4		; <i32> [#uses=1]
	%tmp18.i.i.i = add i32 %104, -1		; <i32> [#uses=1]
	%tmp104 = shl i32 %104, 2		; <i32> [#uses=1]
	%tmp105 = add i32 %tmp104, 4		; <i32> [#uses=1]
	br label %bb1.i13.i.i

bb1.i13.i.i:		; preds = %bb5.i.i.i, %bb.i.i.i
	%indvar15.i.i.i = phi i32 [ 0, %bb.i.i.i ], [ %indvar.next16.i.i.i, %bb5.i.i.i ]		; <i32> [#uses=3]
	%i1.1.i.i.i = phi i32 [ %107, %bb.i.i.i ], [ %i1.0.i.i.i, %bb5.i.i.i ]		; <i32> [#uses=4]
	%tmp100 = sub i32 %tmp18.i.i.i, %indvar15.i.i.i		; <i32> [#uses=2]
	%tmp103 = mul i32 %indvar15.i.i.i, -4		; <i32> [#uses=1]
	%tmp106 = add i32 %tmp103, %tmp105		; <i32> [#uses=1]
	%scevgep107 = getelementptr i8* %v230.i.i.i, i32 %tmp106		; <i8*> [#uses=1]
	%scevgep3536.i.i.i = bitcast i8* %scevgep107 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%108 = load %struct.dwarf_fde** %scevgep3536.i.i.i, align 4		; <%struct.dwarf_fde*> [#uses=2]
	%tmp85 = add i32 %i1.1.i.i.i, %tmp100		; <i32> [#uses=1]
	%tmp86 = shl i32 %tmp85, 2		; <i32> [#uses=1]
	%tmp87 = add i32 %tmp86, 8		; <i32> [#uses=1]
	%tmp91 = shl i32 %i1.1.i.i.i, 2		; <i32> [#uses=1]
	%tmp92 = add i32 %tmp91, 4		; <i32> [#uses=1]
	br label %bb3.i17.i.i

bb2.i14.i.i:		; preds = %bb4.i18.i.i
	%109 = load %struct.dwarf_fde** %scevgep14.i.i.i, align 4		; <%struct.dwarf_fde*> [#uses=1]
	store %struct.dwarf_fde* %109, %struct.dwarf_fde** %scevgep2425.i.i.i, align 4
	%indvar.next.i.i.i = add i32 %110, 1		; <i32> [#uses=1]
	br label %bb3.i17.i.i

bb3.i17.i.i:		; preds = %bb2.i14.i.i, %bb1.i13.i.i
	%110 = phi i32 [ 0, %bb1.i13.i.i ], [ %indvar.next.i.i.i, %bb2.i14.i.i ]		; <i32> [#uses=4]
	%tmp82 = mul i32 %110, -4		; <i32> [#uses=2]
	%tmp88 = add i32 %tmp82, %tmp87		; <i32> [#uses=1]
	%scevgep89 = getelementptr i8* %v19.i.i.i, i32 %tmp88		; <i8*> [#uses=1]
	%scevgep2425.i.i.i = bitcast i8* %scevgep89 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%tmp93 = add i32 %tmp82, %tmp92		; <i32> [#uses=1]
	%scevgep94 = getelementptr i8* %v19.i.i.i, i32 %tmp93		; <i8*> [#uses=1]
	%scevgep14.i.i.i = bitcast i8* %scevgep94 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=2]
	%i1.0.i.i.i = sub i32 %i1.1.i.i.i, %110		; <i32> [#uses=2]
	%111 = icmp eq i32 %i1.1.i.i.i, %110		; <i1> [#uses=1]
	br i1 %111, label %bb5.i.i.i, label %bb4.i18.i.i

bb4.i18.i.i:		; preds = %bb3.i17.i.i
	%112 = load %struct.dwarf_fde** %scevgep14.i.i.i, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%113 = call arm_apcscc  i32 %fde_compare.0.i.i(%struct.object* %ob, %struct.dwarf_fde* %112, %struct.dwarf_fde* %108)		; <i32> [#uses=1]
	%114 = icmp sgt i32 %113, 0		; <i1> [#uses=1]
	br i1 %114, label %bb2.i14.i.i, label %bb5.i.i.i

bb5.i.i.i:		; preds = %bb4.i18.i.i, %bb3.i17.i.i
	%tmp29.i.i.i = add i32 %i1.0.i.i.i, %tmp100		; <i32> [#uses=1]
	%115 = getelementptr %struct.fde_vector* %101, i32 0, i32 2, i32 %tmp29.i.i.i		; <%struct.dwarf_fde**> [#uses=1]
	store %struct.dwarf_fde* %108, %struct.dwarf_fde** %115, align 4
	%indvar.next16.i.i.i = add i32 %indvar15.i.i.i, 1		; <i32> [#uses=2]
	%exitcond98 = icmp eq i32 %indvar.next16.i.i.i, %104		; <i1> [#uses=1]
	br i1 %exitcond98, label %bb6.i.i.i, label %bb1.i13.i.i

bb6.i.i.i:		; preds = %bb5.i.i.i
	call void @llvm.dbg.stoppoint(i32 558, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%116 = load i32* %106, align 4		; <i32> [#uses=1]
	%117 = load i32* %103, align 4		; <i32> [#uses=1]
	%118 = add i32 %117, %116		; <i32> [#uses=1]
	store i32 %118, i32* %106, align 4
	%.pre.i.i = load %struct.fde_vector** %39, align 4		; <%struct.fde_vector*> [#uses=1]
	br label %fde_merge.exit.i.i

fde_merge.exit.i.i:		; preds = %bb6.i.i.i, %bb10.i.i
	%119 = phi %struct.fde_vector* [ %.pre.i.i, %bb6.i.i.i ], [ %102, %bb10.i.i ]		; <%struct.fde_vector*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 582, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram219 to { }*))
	free %struct.fde_vector* %119
	br label %end_fde_sort.exit.i

bb11.i.i:		; preds = %bb7.i.i
	call void @llvm.dbg.stoppoint(i32 588, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call arm_apcscc  void @frame_heapsort(%struct.object* %ob, i32 (%struct.object*, %struct.dwarf_fde*, %struct.dwarf_fde*)* %fde_compare.0.i.i, %struct.fde_vector* %55)
	br label %end_fde_sort.exit.i

end_fde_sort.exit.i:		; preds = %bb11.i.i, %fde_merge.exit.i.i
	call void @llvm.dbg.stoppoint(i32 759, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram278 to { }*))
	%120 = load %struct.fde_vector** %33, align 4		; <%struct.fde_vector*> [#uses=1]
	%121 = load %struct.dwarf_fde** %45, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%122 = getelementptr %struct.fde_vector* %120, i32 0, i32 0		; <i8**> [#uses=1]
	%123 = bitcast %struct.dwarf_fde* %121 to i8*		; <i8*> [#uses=1]
	store i8* %123, i8** %122, align 4
	call void @llvm.dbg.stoppoint(i32 760, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%124 = load %struct.fde_vector** %33, align 4		; <%struct.fde_vector*> [#uses=1]
	%.c.i = bitcast %struct.fde_vector* %124 to %struct.dwarf_fde*		; <%struct.dwarf_fde*> [#uses=1]
	store %struct.dwarf_fde* %.c.i, %struct.dwarf_fde** %45
	call void @llvm.dbg.stoppoint(i32 762, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%125 = load i32* %0		; <i32> [#uses=1]
	%126 = or i32 %125, 1		; <i32> [#uses=2]
	store i32 %126, i32* %0
	br label %init_object.exit

init_object.exit:		; preds = %end_fde_sort.exit.i, %bb1.i.i, %bb8.i
	%.rle120 = phi i32 [ %126, %end_fde_sort.exit.i ], [ %.rle121, %bb8.i ], [ %.rle121, %bb1.i.i ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 938, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram282 to { }*))
	%127 = getelementptr %struct.object* %ob, i32 0, i32 0		; <i8**> [#uses=1]
	%128 = load i8** %127, align 4		; <i8*> [#uses=1]
	%129 = icmp ugt i8* %128, %pc		; <i1> [#uses=1]
	br i1 %129, label %bb16, label %bb2

bb2:		; preds = %init_object.exit, %entry
	%130 = phi i32 [ %1, %entry ], [ %.rle120, %init_object.exit ]		; <i32> [#uses=5]
	call void @llvm.dbg.stoppoint(i32 942, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%131 = and i32 %130, 1		; <i32> [#uses=1]
	%132 = icmp eq i32 %131, 0		; <i1> [#uses=1]
	br i1 %132, label %bb8, label %bb3

bb3:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 944, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%133 = and i32 %130, 4		; <i32> [#uses=1]
	%134 = icmp eq i32 %133, 0		; <i1> [#uses=1]
	br i1 %134, label %bb5, label %bb4

bb4:		; preds = %bb3
	call void @llvm.dbg.stoppoint(i32 945, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram258 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 898, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%135 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	%136 = load %struct.dwarf_fde** %135		; <%struct.dwarf_fde*> [#uses=2]
	%137 = bitcast %struct.dwarf_fde* %136 to %struct.fde_vector*		; <%struct.fde_vector*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 901, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%138 = getelementptr %struct.dwarf_fde* %136, i32 0, i32 1		; <i32*> [#uses=1]
	%139 = load i32* %138, align 4		; <i32> [#uses=1]
	%140 = ptrtoint i8* %pc to i32		; <i32> [#uses=2]
	%141 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%142 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	br label %bb5.i23

bb.i18:		; preds = %bb5.i23
	%143 = add i32 %lo.0.i, %hi.0.ph.i.ph		; <i32> [#uses=1]
	%144 = lshr i32 %143, 1		; <i32> [#uses=3]
	%145 = getelementptr %struct.fde_vector* %137, i32 0, i32 2, i32 %144		; <%struct.dwarf_fde**> [#uses=1]
	%146 = load %struct.dwarf_fde** %145, align 4		; <%struct.dwarf_fde*> [#uses=3]
	%147 = getelementptr %struct.dwarf_fde* %146, i32 0, i32 1		; <i32*> [#uses=2]
	%148 = bitcast i32* %147 to i8*		; <i8*> [#uses=1]
	%149 = load i32* %147, align 1		; <i32> [#uses=1]
	%150 = sub i32 0, %149		; <i32> [#uses=1]
	%151 = getelementptr i8* %148, i32 %150		; <i8*> [#uses=1]
	%152 = bitcast i8* %151 to %struct.dwarf_cie*		; <%struct.dwarf_cie*> [#uses=1]
	%153 = call arm_apcscc  i32 @get_cie_encoding(%struct.dwarf_cie* %152) nounwind		; <i32> [#uses=2]
	%154 = trunc i32 %153 to i8		; <i8> [#uses=3]
	%155 = icmp eq i8 %154, -1		; <i1> [#uses=1]
	br i1 %155, label %base_from_object.exit.i, label %bb1.i.i19

bb1.i.i19:		; preds = %bb.i18
	%156 = and i32 %153, 112		; <i32> [#uses=1]
	switch i32 %156, label %bb5.i.i [
		i32 0, label %base_from_object.exit.i
		i32 16, label %base_from_object.exit.i
		i32 32, label %bb3.i.i
		i32 48, label %bb4.i.i20
		i32 80, label %base_from_object.exit.i
	]

bb3.i.i:		; preds = %bb1.i.i19
	%157 = load i8** %141, align 4		; <i8*> [#uses=1]
	%158 = ptrtoint i8* %157 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit.i

bb4.i.i20:		; preds = %bb1.i.i19
	%159 = load i8** %142, align 4		; <i8*> [#uses=1]
	%160 = ptrtoint i8* %159 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit.i

bb5.i.i:		; preds = %bb1.i.i19
	call void @llvm.dbg.stoppoint(i32 909, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram248 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 312, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 163, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*)) nounwind
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 910, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram248 to { }*)) nounwind
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

base_from_object.exit.i:		; preds = %bb4.i.i20, %bb3.i.i, %bb1.i.i19, %bb1.i.i19, %bb1.i.i19, %bb.i18
	%161 = phi i32 [ %160, %bb4.i.i20 ], [ %158, %bb3.i.i ], [ 0, %bb.i18 ], [ 0, %bb1.i.i19 ], [ 0, %bb1.i.i19 ], [ 0, %bb1.i.i19 ]		; <i32> [#uses=1]
	%162 = getelementptr %struct.dwarf_fde* %146, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%163 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %154, i32 %161, i8* %162, i32* %pc_begin.i) nounwind		; <i8*> [#uses=1]
	%164 = and i8 %154, 15		; <i8> [#uses=1]
	%165 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %164, i32 0, i8* %163, i32* %pc_range.i) nounwind		; <i8*> [#uses=0]
	%166 = load i32* %pc_begin.i, align 4		; <i32> [#uses=2]
	%167 = icmp ult i32 %140, %166		; <i1> [#uses=1]
	br i1 %167, label %bb5.i23, label %bb2.i21

bb2.i21:		; preds = %base_from_object.exit.i
	%168 = load i32* %pc_range.i, align 4		; <i32> [#uses=1]
	%169 = add i32 %168, %166		; <i32> [#uses=1]
	%170 = icmp ult i32 %140, %169		; <i1> [#uses=1]
	br i1 %170, label %binary_search_mixed_encoding_fdes.exit, label %bb3.i22

bb3.i22:		; preds = %bb2.i21
	%171 = add i32 %144, 1		; <i32> [#uses=1]
	br label %bb5.i23

bb5.i23:		; preds = %bb3.i22, %base_from_object.exit.i, %bb4
	%hi.0.ph.i.ph = phi i32 [ %139, %bb4 ], [ %144, %base_from_object.exit.i ], [ %hi.0.ph.i.ph, %bb3.i22 ]		; <i32> [#uses=3]
	%lo.0.i = phi i32 [ %171, %bb3.i22 ], [ 0, %bb4 ], [ %lo.0.i, %base_from_object.exit.i ]		; <i32> [#uses=3]
	%172 = icmp ult i32 %lo.0.i, %hi.0.ph.i.ph		; <i1> [#uses=1]
	br i1 %172, label %bb.i18, label %binary_search_mixed_encoding_fdes.exit

binary_search_mixed_encoding_fdes.exit:		; preds = %bb5.i23, %bb2.i21
	%.0.i = phi %struct.dwarf_fde* [ %146, %bb2.i21 ], [ null, %bb5.i23 ]		; <%struct.dwarf_fde*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 923, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram284 to { }*))
	ret %struct.dwarf_fde* %.0.i

bb5:		; preds = %bb3
	call void @llvm.dbg.stoppoint(i32 946, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%173 = and i32 %130, 2040		; <i32> [#uses=1]
	%174 = icmp eq i32 %173, 0		; <i1> [#uses=1]
	%175 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	%176 = load %struct.dwarf_fde** %175		; <%struct.dwarf_fde*> [#uses=3]
	%177 = bitcast %struct.dwarf_fde* %176 to %struct.fde_vector*		; <%struct.fde_vector*> [#uses=2]
	br i1 %174, label %bb6, label %bb7

bb6:		; preds = %bb5
	call void @llvm.dbg.stoppoint(i32 947, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram223 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 844, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%178 = getelementptr %struct.dwarf_fde* %176, i32 0, i32 1		; <i32*> [#uses=1]
	%179 = load i32* %178, align 4		; <i32> [#uses=1]
	br label %bb5.i30

bb.i25:		; preds = %bb5.i30
	%180 = add i32 %lo.0.i29, %hi.0.ph.i28.ph		; <i32> [#uses=1]
	%181 = lshr i32 %180, 1		; <i32> [#uses=3]
	%182 = getelementptr %struct.fde_vector* %177, i32 0, i32 2, i32 %181		; <%struct.dwarf_fde**> [#uses=1]
	%183 = load %struct.dwarf_fde** %182, align 4		; <%struct.dwarf_fde*> [#uses=2]
	%184 = getelementptr %struct.dwarf_fde* %183, i32 0, i32 2		; <[0 x i8]*> [#uses=2]
	%185 = bitcast [0 x i8]* %184 to i8**		; <i8**> [#uses=1]
	%186 = load i8** %185, align 4		; <i8*> [#uses=2]
	%187 = icmp ugt i8* %186, %pc		; <i1> [#uses=1]
	br i1 %187, label %bb5.i30, label %bb2.i26

bb2.i26:		; preds = %bb.i25
	%188 = bitcast [0 x i8]* %184 to i32*		; <i32*> [#uses=1]
	%189 = getelementptr i32* %188, i32 1		; <i32*> [#uses=1]
	%190 = load i32* %189, align 4		; <i32> [#uses=1]
	%191 = getelementptr i8* %186, i32 %190		; <i8*> [#uses=1]
	%192 = icmp ugt i8* %191, %pc		; <i1> [#uses=1]
	br i1 %192, label %binary_search_unencoded_fdes.exit, label %bb3.i27

bb3.i27:		; preds = %bb2.i26
	%193 = add i32 %181, 1		; <i32> [#uses=1]
	br label %bb5.i30

bb5.i30:		; preds = %bb3.i27, %bb.i25, %bb6
	%hi.0.ph.i28.ph = phi i32 [ %179, %bb6 ], [ %181, %bb.i25 ], [ %hi.0.ph.i28.ph, %bb3.i27 ]		; <i32> [#uses=3]
	%lo.0.i29 = phi i32 [ %193, %bb3.i27 ], [ 0, %bb6 ], [ %lo.0.i29, %bb.i25 ]		; <i32> [#uses=3]
	%194 = icmp ult i32 %lo.0.i29, %hi.0.ph.i28.ph		; <i1> [#uses=1]
	br i1 %194, label %bb.i25, label %binary_search_unencoded_fdes.exit

binary_search_unencoded_fdes.exit:		; preds = %bb5.i30, %bb2.i26
	%.0.i31 = phi %struct.dwarf_fde* [ %183, %bb2.i26 ], [ null, %bb5.i30 ]		; <%struct.dwarf_fde*> [#uses=1]
	tail call void @llvm.dbg.stoppoint(i32 862, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret %struct.dwarf_fde* %.0.i31

bb7:		; preds = %bb5
	call void @llvm.dbg.stoppoint(i32 949, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram260 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 869, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%195 = lshr i32 %130, 3		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 870, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%196 = trunc i32 %195 to i8		; <i8> [#uses=3]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 242, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%197 = icmp eq i8 %196, -1		; <i1> [#uses=1]
	br i1 %197, label %base_from_object.exit.i39, label %bb1.i.i35

bb1.i.i35:		; preds = %bb7
	call void @llvm.dbg.stoppoint(i32 245, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%198 = and i32 %195, 112		; <i32> [#uses=1]
	switch i32 %198, label %bb5.i.i38 [
		i32 0, label %base_from_object.exit.i39
		i32 16, label %base_from_object.exit.i39
		i32 32, label %bb3.i.i36
		i32 48, label %bb4.i.i37
		i32 80, label %base_from_object.exit.i39
	]

bb3.i.i36:		; preds = %bb1.i.i35
	call void @llvm.dbg.stoppoint(i32 253, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%199 = getelementptr %struct.object* %ob, i32 0, i32 1		; <i8**> [#uses=1]
	%200 = load i8** %199, align 4		; <i8*> [#uses=1]
	%201 = ptrtoint i8* %200 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit.i39

bb4.i.i37:		; preds = %bb1.i.i35
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%202 = getelementptr %struct.object* %ob, i32 0, i32 2		; <i8**> [#uses=1]
	%203 = load i8** %202, align 4		; <i8*> [#uses=1]
	%204 = ptrtoint i8* %203 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit.i39

bb5.i.i38:		; preds = %bb1.i.i35
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

base_from_object.exit.i39:		; preds = %bb4.i.i37, %bb3.i.i36, %bb1.i.i35, %bb1.i.i35, %bb1.i.i35, %bb7
	%205 = phi i32 [ %204, %bb4.i.i37 ], [ %201, %bb3.i.i36 ], [ 0, %bb7 ], [ 0, %bb1.i.i35 ], [ 0, %bb1.i.i35 ], [ 0, %bb1.i.i35 ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 873, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	%206 = getelementptr %struct.dwarf_fde* %176, i32 0, i32 1		; <i32*> [#uses=1]
	%207 = load i32* %206, align 4		; <i32> [#uses=1]
	%208 = and i8 %196, 15		; <i8> [#uses=1]
	%209 = ptrtoint i8* %pc to i32		; <i32> [#uses=2]
	br label %bb5.i45

bb.i40:		; preds = %bb5.i45
	%210 = add i32 %lo.0.i44, %hi.0.ph.i43.ph		; <i32> [#uses=1]
	%211 = lshr i32 %210, 1		; <i32> [#uses=3]
	%212 = getelementptr %struct.fde_vector* %177, i32 0, i32 2, i32 %211		; <%struct.dwarf_fde**> [#uses=1]
	%213 = load %struct.dwarf_fde** %212, align 4		; <%struct.dwarf_fde*> [#uses=2]
	%214 = getelementptr %struct.dwarf_fde* %213, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%215 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %196, i32 %205, i8* %214, i32* %pc_begin.i34) nounwind		; <i8*> [#uses=1]
	%216 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %208, i32 0, i8* %215, i32* %pc_range.i33) nounwind		; <i8*> [#uses=0]
	%217 = load i32* %pc_begin.i34, align 4		; <i32> [#uses=2]
	%218 = icmp ult i32 %209, %217		; <i1> [#uses=1]
	br i1 %218, label %bb5.i45, label %bb2.i41

bb2.i41:		; preds = %bb.i40
	%219 = load i32* %pc_range.i33, align 4		; <i32> [#uses=1]
	%220 = add i32 %219, %217		; <i32> [#uses=1]
	%221 = icmp ult i32 %209, %220		; <i1> [#uses=1]
	br i1 %221, label %binary_search_single_encoding_fdes.exit, label %bb3.i42

bb3.i42:		; preds = %bb2.i41
	%222 = add i32 %211, 1		; <i32> [#uses=1]
	br label %bb5.i45

bb5.i45:		; preds = %bb3.i42, %bb.i40, %base_from_object.exit.i39
	%hi.0.ph.i43.ph = phi i32 [ %207, %base_from_object.exit.i39 ], [ %211, %bb.i40 ], [ %hi.0.ph.i43.ph, %bb3.i42 ]		; <i32> [#uses=3]
	%lo.0.i44 = phi i32 [ %222, %bb3.i42 ], [ 0, %base_from_object.exit.i39 ], [ %lo.0.i44, %bb.i40 ]		; <i32> [#uses=3]
	%223 = icmp ult i32 %lo.0.i44, %hi.0.ph.i43.ph		; <i1> [#uses=1]
	br i1 %223, label %bb.i40, label %binary_search_single_encoding_fdes.exit

binary_search_single_encoding_fdes.exit:		; preds = %bb5.i45, %bb2.i41
	%.0.i46 = phi %struct.dwarf_fde* [ %213, %bb2.i41 ], [ null, %bb5.i45 ]		; <%struct.dwarf_fde*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 892, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	ret %struct.dwarf_fde* %.0.i46

bb8:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 954, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%224 = and i32 %130, 2		; <i32> [#uses=1]
	%225 = icmp eq i32 %224, 0		; <i1> [#uses=1]
	%226 = getelementptr %struct.object* %ob, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=2]
	br i1 %225, label %bb15, label %bb9

bb9:		; preds = %bb8
	call void @llvm.dbg.stoppoint(i32 957, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%227 = load %struct.dwarf_fde** %226		; <%struct.dwarf_fde*> [#uses=1]
	%228 = bitcast %struct.dwarf_fde* %227 to i8*		; <i8*> [#uses=1]
	br label %bb13

bb10:		; preds = %bb13
	%229 = call arm_apcscc  %struct.dwarf_fde* @linear_search_fdes(%struct.object* %ob, %struct.dwarf_fde* %231, i8* %pc)		; <%struct.dwarf_fde*> [#uses=2]
	%230 = icmp eq %struct.dwarf_fde* %229, null		; <i1> [#uses=1]
	br i1 %230, label %bb12, label %bb16

bb12:		; preds = %bb10
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb13

bb13:		; preds = %bb12, %bb9
	%indvar = phi i32 [ 0, %bb9 ], [ %indvar.next, %bb12 ]		; <i32> [#uses=2]
	%tmp117 = shl i32 %indvar, 2		; <i32> [#uses=1]
	%scevgep118 = getelementptr i8* %228, i32 %tmp117		; <i8*> [#uses=1]
	%p.0 = bitcast i8* %scevgep118 to %struct.dwarf_fde**		; <%struct.dwarf_fde**> [#uses=1]
	%231 = load %struct.dwarf_fde** %p.0, align 4		; <%struct.dwarf_fde*> [#uses=2]
	%232 = icmp eq %struct.dwarf_fde* %231, null		; <i1> [#uses=1]
	br i1 %232, label %bb16, label %bb10

bb15:		; preds = %bb8
	call void @llvm.dbg.stoppoint(i32 966, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%233 = load %struct.dwarf_fde** %226, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%234 = call arm_apcscc  %struct.dwarf_fde* @linear_search_fdes(%struct.object* %ob, %struct.dwarf_fde* %233, i8* %pc)		; <%struct.dwarf_fde*> [#uses=1]
	ret %struct.dwarf_fde* %234

bb16:		; preds = %bb13, %bb10, %init_object.exit
	%.0 = phi %struct.dwarf_fde* [ null, %init_object.exit ], [ %229, %bb10 ], [ null, %bb13 ]		; <%struct.dwarf_fde*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 966, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret %struct.dwarf_fde* %.0
}

define arm_apcscc %struct.dwarf_fde* @_Unwind_Find_FDE(i8* %pc, %struct.dwarf_eh_bases* nocapture %bases) {
entry:
	%func = alloca i32, align 4		; <i32*> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram297 to { }*))
	call void @llvm.dbg.stoppoint(i32 977, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	call void @llvm.dbg.stoppoint(i32 982, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram166 to { }*))
	br label %bb3

bb:		; preds = %bb3
	%0 = getelementptr %struct.object* %ob.0, i32 0, i32 0		; <i8**> [#uses=1]
	%1 = load i8** %0, align 4		; <i8*> [#uses=1]
	%2 = icmp ugt i8* %1, %pc		; <i1> [#uses=1]
	br i1 %2, label %bb2, label %bb1

bb1:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 985, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%3 = call arm_apcscc  %struct.dwarf_fde* @search_object(%struct.object* %ob.0, i8* %pc)		; <%struct.dwarf_fde*> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 986, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%4 = icmp eq %struct.dwarf_fde* %3, null		; <i1> [#uses=1]
	br i1 %4, label %bb9, label %fini

bb2:		; preds = %bb
	%5 = getelementptr %struct.object* %ob.0, i32 0, i32 5		; <%struct.object**> [#uses=1]
	br label %bb3

bb3:		; preds = %bb2, %entry
	%ob.0.in = phi %struct.object** [ @seen_objects, %entry ], [ %5, %bb2 ]		; <%struct.object**> [#uses=1]
	%ob.0 = load %struct.object** %ob.0.in		; <%struct.object*> [#uses=5]
	%6 = icmp eq %struct.object* %ob.0, null		; <i1> [#uses=1]
	br i1 %6, label %bb9, label %bb

bb4:		; preds = %bb9
	%7 = getelementptr %struct.object* %19, i32 0, i32 5		; <%struct.object**> [#uses=2]
	%8 = load %struct.object** %7, align 4		; <%struct.object*> [#uses=1]
	store %struct.object* %8, %struct.object** @unseen_objects, align 4
	%9 = call arm_apcscc  %struct.dwarf_fde* @search_object(%struct.object* %19, i8* %pc)		; <%struct.dwarf_fde*> [#uses=3]
	%10 = getelementptr %struct.object* %19, i32 0, i32 0		; <i8**> [#uses=1]
	br label %bb7

bb5:		; preds = %bb7
	%11 = getelementptr %struct.object* %16, i32 0, i32 0		; <i8**> [#uses=1]
	%12 = load i8** %11, align 4		; <i8*> [#uses=1]
	%13 = load i8** %10, align 4		; <i8*> [#uses=1]
	%14 = icmp ult i8* %12, %13		; <i1> [#uses=1]
	br i1 %14, label %bb8, label %bb6

bb6:		; preds = %bb5
	%15 = getelementptr %struct.object* %16, i32 0, i32 5		; <%struct.object**> [#uses=1]
	br label %bb7

bb7:		; preds = %bb6, %bb4
	%p.0 = phi %struct.object** [ @seen_objects, %bb4 ], [ %15, %bb6 ]		; <%struct.object**> [#uses=2]
	%16 = load %struct.object** %p.0, align 4		; <%struct.object*> [#uses=4]
	%17 = icmp eq %struct.object* %16, null		; <i1> [#uses=1]
	br i1 %17, label %bb8, label %bb5

bb8:		; preds = %bb7, %bb5
	store %struct.object* %16, %struct.object** %7, align 4
	store %struct.object* %19, %struct.object** %p.0, align 4
	%18 = icmp eq %struct.dwarf_fde* %9, null		; <i1> [#uses=1]
	br i1 %18, label %bb9, label %fini

bb9:		; preds = %bb8, %bb3, %bb1
	%f.0 = phi %struct.dwarf_fde* [ %9, %bb8 ], [ %3, %bb1 ], [ null, %bb3 ]		; <%struct.dwarf_fde*> [#uses=1]
	%19 = load %struct.object** @unseen_objects, align 4		; <%struct.object*> [#uses=7]
	%20 = icmp eq %struct.object* %19, null		; <i1> [#uses=1]
	br i1 %20, label %fini, label %bb4

fini:		; preds = %bb9, %bb8, %bb1
	%f.1 = phi %struct.dwarf_fde* [ %3, %bb1 ], [ %9, %bb8 ], [ %f.0, %bb9 ]		; <%struct.dwarf_fde*> [#uses=5]
	%ob.1 = phi %struct.object* [ %ob.0, %bb1 ], [ %19, %bb9 ], [ %19, %bb8 ]		; <%struct.object*> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 1011, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	call void @llvm.dbg.stoppoint(i32 1013, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	%21 = icmp eq %struct.dwarf_fde* %f.1, null		; <i1> [#uses=1]
	br i1 %21, label %bb13, label %bb10

bb10:		; preds = %fini
	call void @llvm.dbg.stoppoint(i32 1018, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%22 = getelementptr %struct.object* %ob.1, i32 0, i32 1		; <i8**> [#uses=2]
	%23 = load i8** %22, align 4		; <i8*> [#uses=1]
	%24 = getelementptr %struct.dwarf_eh_bases* %bases, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* %23, i8** %24, align 4
	call void @llvm.dbg.stoppoint(i32 1019, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%25 = getelementptr %struct.object* %ob.1, i32 0, i32 2		; <i8**> [#uses=2]
	%26 = load i8** %25, align 4		; <i8*> [#uses=1]
	%27 = getelementptr %struct.dwarf_eh_bases* %bases, i32 0, i32 1		; <i8**> [#uses=1]
	store i8* %26, i8** %27, align 4
	call void @llvm.dbg.stoppoint(i32 1021, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%28 = getelementptr %struct.object* %ob.1, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	%29 = load i32* %28		; <i32> [#uses=2]
	%30 = lshr i32 %29, 3		; <i32> [#uses=1]
	%31 = and i32 %30, 255		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 1022, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%32 = and i32 %29, 4		; <i32> [#uses=1]
	%33 = icmp eq i32 %32, 0		; <i1> [#uses=1]
	br i1 %33, label %bb12, label %bb11

bb11:		; preds = %bb10
	call void @llvm.dbg.stoppoint(i32 1023, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram248 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 312, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 163, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit30 to { }*)) nounwind
	%34 = getelementptr %struct.dwarf_fde* %f.1, i32 0, i32 1		; <i32*> [#uses=2]
	%35 = bitcast i32* %34 to i8*		; <i8*> [#uses=1]
	%36 = load i32* %34, align 1		; <i32> [#uses=1]
	%37 = sub i32 0, %36		; <i32> [#uses=1]
	%38 = getelementptr i8* %35, i32 %37		; <i8*> [#uses=1]
	%39 = bitcast i8* %38 to %struct.dwarf_cie*		; <%struct.dwarf_cie*> [#uses=1]
	%40 = call arm_apcscc  i32 @get_cie_encoding(%struct.dwarf_cie* %39) nounwind		; <i32> [#uses=1]
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram76 to { }*)) nounwind
	br label %bb12

bb12:		; preds = %bb11, %bb10
	%encoding.0 = phi i32 [ %40, %bb11 ], [ %31, %bb10 ]		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 1024, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%41 = trunc i32 %encoding.0 to i8		; <i8> [#uses=2]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 242, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%42 = icmp eq i8 %41, -1		; <i1> [#uses=1]
	br i1 %42, label %base_from_object.exit, label %bb1.i

bb1.i:		; preds = %bb12
	call void @llvm.dbg.stoppoint(i32 245, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%43 = and i32 %encoding.0, 112		; <i32> [#uses=1]
	switch i32 %43, label %bb5.i [
		i32 0, label %base_from_object.exit
		i32 16, label %base_from_object.exit
		i32 32, label %bb3.i
		i32 48, label %bb4.i
		i32 80, label %base_from_object.exit
	]

bb3.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 253, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%44 = load i8** %22, align 4		; <i8*> [#uses=1]
	%45 = ptrtoint i8* %44 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb4.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 255, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%46 = load i8** %25, align 4		; <i8*> [#uses=1]
	%47 = ptrtoint i8* %46 to i32		; <i32> [#uses=1]
	br label %base_from_object.exit

bb5.i:		; preds = %bb1.i
	call void @llvm.dbg.stoppoint(i32 257, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call arm_apcscc  void @abort() noreturn nounwind
	unreachable

base_from_object.exit:		; preds = %bb4.i, %bb3.i, %bb1.i, %bb1.i, %bb1.i, %bb12
	%48 = phi i32 [ %47, %bb4.i ], [ %45, %bb3.i ], [ 0, %bb12 ], [ 0, %bb1.i ], [ 0, %bb1.i ], [ 0, %bb1.i ]		; <i32> [#uses=1]
	%49 = getelementptr %struct.dwarf_fde* %f.1, i32 0, i32 2, i32 0		; <i8*> [#uses=1]
	%50 = call arm_apcscc  i8* @read_encoded_value_with_base(i8 zeroext %41, i32 %48, i8* %49, i32* %func)		; <i8*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 1026, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram231 to { }*))
	%51 = load i32* %func, align 4		; <i32> [#uses=1]
	%52 = inttoptr i32 %51 to i8*		; <i8*> [#uses=1]
	%53 = getelementptr %struct.dwarf_eh_bases* %bases, i32 0, i32 2		; <i8**> [#uses=1]
	store i8* %52, i8** %53, align 4
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram297 to { }*))
	ret %struct.dwarf_fde* %f.1

bb13:		; preds = %fini
	call void @llvm.dbg.stoppoint(i32 1029, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret %struct.dwarf_fde* %f.1
}

define arm_apcscc i8* @__deregister_frame_info_bases(i8* %begin) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram301 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 180, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = icmp eq i8* %begin, null		; <i1> [#uses=1]
	br i1 %0, label %bb17, label %bb

bb:		; preds = %entry
	%1 = bitcast i8* %begin to i32*		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %bb17, label %bb6

bb3:		; preds = %bb6
	%4 = getelementptr %struct.object* %10, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=1]
	%5 = load %struct.dwarf_fde** %4, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%6 = bitcast %struct.dwarf_fde* %5 to i8*		; <i8*> [#uses=1]
	%7 = icmp eq i8* %6, %begin		; <i1> [#uses=1]
	%8 = getelementptr %struct.object* %10, i32 0, i32 5		; <%struct.object**> [#uses=2]
	br i1 %7, label %bb4, label %bb6

bb4:		; preds = %bb3
	tail call void @llvm.dbg.stoppoint(i32 190, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%9 = load %struct.object** %8, align 4		; <%struct.object*> [#uses=1]
	store %struct.object* %9, %struct.object** %p.0, align 4
	br label %out

bb6:		; preds = %bb3, %bb
	%p.0 = phi %struct.object** [ @unseen_objects, %bb ], [ %8, %bb3 ]		; <%struct.object**> [#uses=2]
	%10 = load %struct.object** %p.0, align 4		; <%struct.object*> [#uses=4]
	%11 = icmp eq %struct.object* %10, null		; <i1> [#uses=1]
	br i1 %11, label %bb14, label %bb3

bb8:		; preds = %bb14
	%12 = getelementptr %struct.object* %30, i32 0, i32 4, i32 0		; <i32*> [#uses=1]
	%13 = load i32* %12		; <i32> [#uses=1]
	%14 = and i32 %13, 1		; <i32> [#uses=1]
	%15 = icmp eq i32 %14, 0		; <i1> [#uses=1]
	%16 = getelementptr %struct.object* %30, i32 0, i32 3, i32 0		; <%struct.dwarf_fde**> [#uses=3]
	br i1 %15, label %bb11, label %bb9

bb9:		; preds = %bb8
	%17 = load %struct.dwarf_fde** %16		; <%struct.dwarf_fde*> [#uses=1]
	%18 = bitcast %struct.dwarf_fde* %17 to i8**		; <i8**> [#uses=1]
	%19 = load i8** %18, align 4		; <i8*> [#uses=1]
	%20 = icmp eq i8* %19, %begin		; <i1> [#uses=1]
	br i1 %20, label %bb10, label %bb13

bb10:		; preds = %bb9
	tail call void @llvm.dbg.stoppoint(i32 200, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%21 = getelementptr %struct.object* %30, i32 0, i32 5		; <%struct.object**> [#uses=1]
	%22 = load %struct.object** %21, align 4		; <%struct.object*> [#uses=1]
	store %struct.object* %22, %struct.object** %p.1, align 4
	tail call void @llvm.dbg.stoppoint(i32 201, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%23 = load %struct.dwarf_fde** %16		; <%struct.dwarf_fde*> [#uses=1]
	free %struct.dwarf_fde* %23
	br label %out

bb11:		; preds = %bb8
	%24 = load %struct.dwarf_fde** %16, align 4		; <%struct.dwarf_fde*> [#uses=1]
	%25 = bitcast %struct.dwarf_fde* %24 to i8*		; <i8*> [#uses=1]
	%26 = icmp eq i8* %25, %begin		; <i1> [#uses=1]
	br i1 %26, label %bb12, label %bb13

bb12:		; preds = %bb11
	tail call void @llvm.dbg.stoppoint(i32 210, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%27 = getelementptr %struct.object* %30, i32 0, i32 5		; <%struct.object**> [#uses=1]
	%28 = load %struct.object** %27, align 4		; <%struct.object*> [#uses=1]
	store %struct.object* %28, %struct.object** %p.1, align 4
	br label %out

bb13:		; preds = %bb11, %bb9
	%29 = getelementptr %struct.object* %30, i32 0, i32 5		; <%struct.object**> [#uses=1]
	br label %bb14

bb14:		; preds = %bb13, %bb6
	%p.1 = phi %struct.object** [ %29, %bb13 ], [ @seen_objects, %bb6 ]		; <%struct.object**> [#uses=3]
	%30 = load %struct.object** %p.1, align 4		; <%struct.object*> [#uses=8]
	%31 = icmp eq %struct.object* %30, null		; <i1> [#uses=1]
	br i1 %31, label %bb15, label %bb8

out:		; preds = %bb12, %bb10, %bb4
	%ob.0 = phi %struct.object* [ %10, %bb4 ], [ %30, %bb10 ], [ %30, %bb12 ]		; <%struct.object*> [#uses=2]
	tail call void @llvm.dbg.stoppoint(i32 216, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 217, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram168 to { }*))
	%32 = icmp eq %struct.object* %ob.0, null		; <i1> [#uses=1]
	br i1 %32, label %bb15, label %bb16

bb15:		; preds = %out, %bb14
	tail call void @llvm.dbg.stoppoint(i32 217, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call arm_apcscc  void @abort() noreturn nounwind
	unreachable

bb16:		; preds = %out
	tail call void @llvm.dbg.stoppoint(i32 218, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%33 = bitcast %struct.object* %ob.0 to i8*		; <i8*> [#uses=1]
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram301 to { }*))
	ret i8* %33

bb17:		; preds = %bb, %entry
	tail call void @llvm.dbg.stoppoint(i32 218, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret i8* null
}

define arm_apcscc i8* @__deregister_frame_info(i8* %begin) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram303 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 224, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = tail call arm_apcscc  i8* @__deregister_frame_info_bases(i8* %begin)		; <i8*> [#uses=1]
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram303 to { }*))
	ret i8* %0
}

define arm_apcscc void @__deregister_frame(i8* %begin) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram305 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 231, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = bitcast i8* %begin to i32*		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb

bb:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 232, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram303 to { }*)) nounwind
	tail call void @llvm.dbg.stoppoint(i32 224, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%3 = tail call arm_apcscc  i8* @__deregister_frame_info_bases(i8* %begin) nounwind		; <i8*> [#uses=1]
	free i8* %3
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram305 to { }*))
	ret void

return:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 233, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void
}
