; RUN: llvm-as < %s | llc -march=x86
; <rdar://problem/6968283>
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%struct.A = type <{ i8 }>
	%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
	%struct.__type_info_pseudo = type { i8*, i8* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [12 x i8] c"testcase.ii\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str1 = internal constant [22 x i8] c"/Volumes/Sandbox/swb/\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@.str2 = internal constant [57 x i8] c"4.2.1 (Based on Apple Inc. build 5646) (LLVM build 2110)\00", section "llvm.metadata"		; <[57 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 4, i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([57 x i8]* @.str2, i32 0, i32 0), i1 true, i1 true, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [2 x i8] c"A\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 64, i64 64, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite4 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str5 = internal constant [5 x i8] c"Func\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str6 = internal constant [13 x i8] c"_ZN1A4FuncEv\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([5 x i8]* @.str5, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str5, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str6, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite4 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str7 = internal constant [4 x i8] c"Bar\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str8 = internal constant [12 x i8] c"_ZN1A3BarEv\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram9 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 7, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite4 to { }*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array10 = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram9 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite11 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, i64 8, i64 8, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array10 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype12 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 64, i64 64, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite11 to { }*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array13 = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype12 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array13 to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram14 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 7, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@_ZTI6Error1 = weak_odr constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i64 add (i64 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i64), i64 16) to i8*), i8* getelementptr ([8 x i8]* @_ZTS6Error1, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTVN10__cxxabiv117__class_type_infoE = external constant [0 x i32 (...)*]		; <[0 x i32 (...)*]*> [#uses=1]
@_ZTS6Error1 = weak_odr constant [8 x i8] c"6Error1\00"		; <[8 x i8]*> [#uses=1]
@_ZTI6Error2 = weak_odr constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i64 add (i64 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i64), i64 16) to i8*), i8* getelementptr ([8 x i8]* @_ZTS6Error2, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTS6Error2 = weak_odr constant [8 x i8] c"6Error2\00"		; <[8 x i8]*> [#uses=1]

define void @_ZN1A3BarEv(%struct.A* %this) ssp {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram14 to { }*))
	tail call void @llvm.dbg.stoppoint(i32 9, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	invoke void @_ZN1A4FuncEv(%struct.A* %this)
			to label %return unwind label %lpad

bb:		; preds = %lpad
	tail call void @llvm.dbg.stoppoint(i32 10, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = tail call i8* @__cxa_begin_catch(i8* %eh_ptr) nounwind		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch()
	ret void

return:		; preds = %entry
	tail call void @llvm.dbg.stoppoint(i32 14, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	ret void

lpad:		; preds = %entry
	%eh_ptr = tail call i8* @llvm.eh.exception()		; <i8*> [#uses=3]
	tail call void @llvm.dbg.stoppoint(i32 14, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%eh_select = tail call i64 (i8*, i8*, ...)* @llvm.eh.selector.i64(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI6Error1 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI6Error2 to i8*), i8* null)		; <i64> [#uses=1]
	%eh_typeid = tail call i64 @llvm.eh.typeid.for.i64(i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI6Error1 to i8*))		; <i64> [#uses=1]
	%1 = icmp eq i64 %eh_select, %eh_typeid		; <i1> [#uses=1]
	br i1 %1, label %bb, label %ppad7

ppad7:		; preds = %lpad
	%eh_typeid8 = tail call i64 @llvm.eh.typeid.for.i64(i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI6Error2 to i8*))		; <i64> [#uses=0]
	%2 = tail call i8* @__cxa_begin_catch(i8* %eh_ptr) nounwind		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch()
	ret void
}

declare void @llvm.dbg.func.start({ }*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind readnone

declare void @_ZN1A4FuncEv(%struct.A*)

declare i8* @__cxa_begin_catch(i8*) nounwind

declare i8* @llvm.eh.exception() nounwind

declare i64 @llvm.eh.selector.i64(i8*, i8*, ...) nounwind

declare i64 @llvm.eh.typeid.for.i64(i8*) nounwind

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...)
