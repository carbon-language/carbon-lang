; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep alloca
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

	type { }		; type %0
	type <{ i8 }>		; type %1
	type { i32 (...)**, %3 }		; type %2
	type { %4, %2*, i8, i8, %10*, %11*, %12*, %12* }		; type %3
	type { i32 (...)**, i32, i32, i32, i32, i32, %5*, %6, [8 x %6], i32, %6*, %7 }		; type %4
	type { %5*, void (i32, %4*, i32)*, i32, i32 }		; type %5
	type { i8*, i32 }		; type %6
	type { %8* }		; type %7
	type { i32, %9**, i32, %9**, i8** }		; type %8
	type { i32 (...)**, i32 }		; type %9
	type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %7 }		; type %10
	type { %9, i32*, i8, i32*, i32*, i32*, i8, [256 x i8], [256 x i8], i8 }		; type %11
	type { %9 }		; type %12
	type { i32, void ()* }		; type %13
	type { %15 }		; type %14
	type { %16 }		; type %15
	type { %17 }		; type %16
	type { i32*, i32*, i32* }		; type %17
	type { %19 }		; type %18
	type { %20 }		; type %19
	type { %21 }		; type %20
	type { %14*, %14*, %14* }		; type %21
	type { i32 }		; type %22
	type { i8 }		; type %23
	type { i32* }		; type %24
	type { %14* }		; type %25
	type { %27 }		; type %26
	type { i8* }		; type %27
	type { %29, %30, %3 }		; type %28
	type { i32 (...)** }		; type %29
	type { %10, i32, %26 }		; type %30
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, %0*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0*, %0*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0* }
	%llvm.dbg.enumerator.type = type { i32, i8*, i64 }
	%llvm.dbg.global_variable.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1, %0* }
	%llvm.dbg.subprogram.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1 }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%llvm.dbg.variable.type = type { i32, %0*, i8*, %0*, i32, %0* }

@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
internal constant [11 x i8] c"bigfib.cpp\00", section "llvm.metadata"		; <[11 x i8]*>:0 [#uses=1]
internal constant [84 x i8] c"/Volumes/Nanpura/mainline/llvm/projects/llvm-test/SingleSource/Benchmarks/Misc-C++/\00", section "llvm.metadata"		; <[84 x i8]*>:1 [#uses=1]
internal constant [57 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build 2099)\00", section "llvm.metadata"		; <[57 x i8]*>:2 [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([11 x i8]* @0, i32 0, i32 0), i8* getelementptr ([84 x i8]* @1, i32 0, i32 0), i8* getelementptr ([57 x i8]* @2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
internal constant [23 x i8] c"/usr/include/c++/4.0.0\00", section "llvm.metadata"		; <[23 x i8]*>:3 [#uses=1]


internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*>:4 [#uses=1]
@llvm.dbg.basictype103 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @4, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
internal constant [8 x i8] c"iomanip\00", section "llvm.metadata"		; <[8 x i8]*>:5 [#uses=1]
@llvm.dbg.compile_unit1548 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([8 x i8]* @5, i32 0, i32 0), i8* getelementptr ([23 x i8]* @3, i32 0, i32 0), i8* getelementptr ([57 x i8]* @2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
internal constant [6 x i8] c"_Setw\00", section "llvm.metadata"		; <[6 x i8]*>:6 [#uses=1]
internal constant [5 x i8] c"_M_n\00", section "llvm.metadata"		; <[5 x i8]*>:7 [#uses=1]
@llvm.dbg.derivedtype1552 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @7, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1548 to %0*), i32 232, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype103 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1553 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1552 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite1554 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @6, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1548 to %0*), i32 232, i64 32, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array1553 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1555 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1554 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype103 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1556 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1555 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
internal constant [5 x i8] c"setw\00", section "llvm.metadata"		; <[5 x i8]*>:8 [#uses=2]
internal constant [11 x i8] c"_ZSt4setwi\00", section "llvm.metadata"		; <[11 x i8]*>:9 [#uses=1]
@llvm.dbg.subprogram1559 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @8, i32 0, i32 0), i8* getelementptr ([5 x i8]* @8, i32 0, i32 0), i8* getelementptr ([11 x i8]* @9, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1548 to %0*), i32 242, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1556 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
internal constant [4 x i8] c"__x\00", section "llvm.metadata"		; <[4 x i8]*>:10 [#uses=1]
@llvm.dbg.variable1563 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1559 to %0*), i8* getelementptr ([4 x i8]* @10, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1548 to %0*), i32 244, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1554 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]

define linkonce i32 @_ZSt4setwi(i32) nounwind {
	%2 = alloca %22		; <%22*> [#uses=2]
	%3 = alloca %22		; <%22*> [#uses=3]
	%4 = alloca %22		; <%22*> [#uses=2]
	%5 = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1559 to %0*))
	%6 = bitcast %22* %3 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %6, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1563 to %0*))
	call void @llvm.dbg.stoppoint(i32 245, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1548 to %0*))
	%7 = getelementptr %22* %3, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %0, i32* %7, align 4
	call void @llvm.dbg.stoppoint(i32 246, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1548 to %0*))
	%8 = getelementptr %22* %4, i32 0, i32 0		; <i32*> [#uses=1]
	%9 = getelementptr %22* %3, i32 0, i32 0		; <i32*> [#uses=1]
	%10 = load i32* %9, align 4		; <i32> [#uses=1]
	store i32 %10, i32* %8, align 4
	%11 = getelementptr %22* %2, i32 0, i32 0		; <i32*> [#uses=1]
	%12 = getelementptr %22* %4, i32 0, i32 0		; <i32*> [#uses=1]
	%13 = load i32* %12, align 4		; <i32> [#uses=1]
	store i32 %13, i32* %11, align 4
	%14 = bitcast %22* %2 to i32*		; <i32*> [#uses=1]
	%15 = load i32* %14		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 246, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1548 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1559 to %0*))
	ret i32 %15
}

declare void @llvm.dbg.func.start(%0*) nounwind

declare void @llvm.dbg.declare(%0*, %0*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, %0*) nounwind

declare void @llvm.dbg.region.end(%0*) nounwind

