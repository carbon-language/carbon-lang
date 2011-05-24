; RUN: opt < %s -scalarrepl -S | not grep alloca
; RUN: opt < %s -scalarrepl-ssa -S | not grep alloca
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
	%struct.Sphere = type { %struct.Vec }
	%struct.Vec = type { i32, i32, i32 }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [6 x i8] c"r.cpp\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1 = internal constant [5 x i8] c"/tmp\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str2 = internal constant [55 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build 00)\00", section "llvm.metadata"		; <[55 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([55 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [4 x i8] c"Vec\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str4 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str5 = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str6 = internal constant [2 x i8] c"y\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype7 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str6, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, i64 32, i64 32, i64 32, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str8 = internal constant [2 x i8] c"z\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype9 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, i64 32, i64 32, i64 64, i32 0, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype10 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype11 = internal constant %llvm.dbg.derivedtype.type { i32 458790, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 96, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype12 = internal constant %llvm.dbg.derivedtype.type { i32 458768, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype10 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite13 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite13 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array14 = internal constant [5 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype10 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) ], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite15 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array14 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram16 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 5, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite15 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array17 = internal constant [5 x { }*] [ { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype7 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to { }*) ], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite18 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, i64 96, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array17 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype19 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array20 = internal constant [5 x { }*] [ { }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype19 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) ], section "llvm.metadata"		; <[5 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([5 x { }*]* @llvm.dbg.array20 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str21 = internal constant [13 x i8] c"__comp_ctor \00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str22 = internal constant [14 x i8] c"_ZN3VecC1Eiii\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.array32 = internal constant [3 x { }*] [ { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite33 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array32 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str34 = internal constant [10 x i8] c"operator-\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str35 = internal constant [14 x i8] c"_ZmiRK3VecS1_\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str41 = internal constant [7 x i8] c"Sphere\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str43 = internal constant [7 x i8] c"center\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype44 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str43, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 14, i64 96, i64 32, i64 0, i32 1, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype45 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite52 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array46 = internal constant [3 x { }*] [ { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype45 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite47 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array46 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str48 = internal constant [11 x i8] c"ray_sphere\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str49 = internal constant [30 x i8] c"_ZN6Sphere10ray_sphereERK3Vec\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram50 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str48, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str48, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str49, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 16, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite47 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array51 = internal constant [2 x { }*] [ { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype44 to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram50 to { }*) ], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite52 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str41, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 12, i64 96, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array51 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype53 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite52 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array54 = internal constant [3 x { }*] [ { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype53 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*) ], section "llvm.metadata"		; <[3 x { }*]*> [#uses=1]
@llvm.dbg.composite55 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([3 x { }*]* @llvm.dbg.array54 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram56 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([11 x i8]* @.str48, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str48, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str49, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 16, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite55 to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str61 = internal constant [2 x i8] c"v\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable62 = internal constant %llvm.dbg.variable.type { i32 459008, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram56 to { }*), i8* getelementptr ([2 x i8]* @.str61, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 17, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite18 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind

define i32 @_ZN6Sphere10ray_sphereERK3Vec(%struct.Sphere* %this, %struct.Vec* %Orig) nounwind {
entry:
	%v = alloca %struct.Vec, align 8		; <%struct.Vec*> [#uses=4]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram56 to { }*))
	%0 = bitcast %struct.Vec* %v to { }*		; <{ }*> [#uses=1]
	call void @llvm.dbg.declare({ }* %0, { }* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable62 to { }*))
	%1 = getelementptr %struct.Sphere* %this, i32 0, i32 0, i32 2		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	%3 = getelementptr %struct.Vec* %Orig, i32 0, i32 2		; <i32*> [#uses=1]
	%4 = load i32* %3, align 4		; <i32> [#uses=1]
	%5 = sub i32 %2, %4		; <i32> [#uses=1]
	%6 = getelementptr %struct.Sphere* %this, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%7 = load i32* %6, align 4		; <i32> [#uses=1]
	%8 = getelementptr %struct.Vec* %Orig, i32 0, i32 1		; <i32*> [#uses=1]
	%9 = load i32* %8, align 4		; <i32> [#uses=1]
	%10 = sub i32 %7, %9		; <i32> [#uses=1]
	%11 = getelementptr %struct.Sphere* %this, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%12 = load i32* %11, align 4		; <i32> [#uses=1]
	%13 = getelementptr %struct.Vec* %Orig, i32 0, i32 0		; <i32*> [#uses=1]
	%14 = load i32* %13, align 4		; <i32> [#uses=1]
	%15 = sub i32 %12, %14		; <i32> [#uses=1]
	%16 = getelementptr %struct.Vec* %v, i32 0, i32 0		; <i32*> [#uses=2]
	store i32 %15, i32* %16, align 8
	%17 = getelementptr %struct.Vec* %v, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %10, i32* %17, align 4
	%18 = getelementptr %struct.Vec* %v, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %5, i32* %18, align 8
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	call void @llvm.dbg.stoppoint(i32 9, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*)) nounwind
	%19 = load i32* %16, align 8		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 18, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram56 to { }*))
	ret i32 %19
}
