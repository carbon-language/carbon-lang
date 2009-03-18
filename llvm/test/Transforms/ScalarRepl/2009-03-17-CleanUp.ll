; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep store | not grep undef

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	type { }		; type %0
	type { double, double }		; type %1
	type { i32, void ()* }		; type %2
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, %0*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0*, %0*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0* }
	%llvm.dbg.global_variable.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1, %0* }
	%llvm.dbg.subprogram.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1 }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%llvm.dbg.variable.type = type { i32, %0*, i8*, %0*, i32, %0* }
	%struct..0._50 = type { i32 }
	%struct..1__pthread_mutex_s = type { i32, i32, i32, i32, i32, %struct..0._50 }
	%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
	%struct.__locale_struct = type { [13 x %struct.locale_data*], i16*, i32*, i32*, [13 x i8*] }
	%struct.__pthread_slist_t = type { %struct.__pthread_slist_t* }
	%struct.__si_class_type_info_pseudo = type { %struct.__type_info_pseudo, %"struct.std::type_info"* }
	%struct.__type_info_pseudo = type { i8*, i8* }
	%struct.locale_data = type opaque
	%"struct.polynomial<double>" = type { i32 (...)**, double*, i32 }
	%"struct.polynomial<std::complex<double> >" = type { i32 (...)**, %"struct.std::complex<double>"*, i32 }
	%struct.pthread_attr_t = type { i32, [32 x i8] }
	%struct.pthread_mutex_t = type { %struct..1__pthread_mutex_s }
	%struct.pthread_mutexattr_t = type { i32 }
	%"struct.std::allocator<char>" = type <{ i8 }>
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i8, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"* }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" = type { i8* }
	%"struct.std::complex<double>" = type { %1 }
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", %struct.__locale_struct*, i8, i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::exception" = type { i32 (...)** }
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
	%"struct.std::ios_base::Init" = type <{ i8 }>
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::ios_base::_Words" = type { i8*, i32 }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
	%"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >" = type { %"struct.std::locale::facet" }
	%"struct.std::num_put<char,std::ostreambuf_iterator<char, std::char_traits<char> > >" = type { %"struct.std::locale::facet" }
	%"struct.std::overflow_error" = type { %"struct.std::runtime_error" }
	%"struct.std::runtime_error" = type { %"struct.std::exception", %"struct.std::string" }
	%"struct.std::string" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" }
	%"struct.std::type_info" = type { i32 (...)**, i8* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [13 x i8] c"fftbench.cpp\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str1 = internal constant [42 x i8] c"/developer/home2/zsth/test/debug/tmp3/X3/\00", section "llvm.metadata"		; <[42 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5641) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([13 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([42 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [8 x i8] c"complex\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str4 = internal constant [110 x i8] c"/developer/home2/zsth/projects/llvm.org/install/lib/gcc/i686-pc-linux-gnu/4.2.1/../../../../include/c++/4.2.1\00", section "llvm.metadata"		; <[110 x i8]*> [#uses=1]
@llvm.dbg.compile_unit5 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([110 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str6 = internal constant [16 x i8] c"complex<double>\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str7 = internal constant [15 x i8] c"complex double\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str7, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 128, i64 64, i64 0, i32 0, i32 3 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str8 = internal constant [9 x i8] c"_M_value\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str8, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1195, i64 128, i64 64, i64 0, i32 1, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype9 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite10 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1161, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite10 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str11 = internal constant [7 x i8] c"double\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.basictype12 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str11, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 64, i64 0, i32 0, i32 4 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array13 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite14 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array13 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram15 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1215, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite14 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str16 = internal constant [15 x i8] c"complex<float>\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@.str18 = internal constant [14 x i8] c"complex float\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.basictype19 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str18, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 32, i64 0, i32 0, i32 3 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype20 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str8, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1042, i64 64, i64 32, i64 0, i32 1, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype19 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype21 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite171 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array22 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype19 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite23 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array22 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram24 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1007, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite23 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str25 = internal constant [6 x i8] c"float\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.basictype26 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str25, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 4 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array27 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype26 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype26 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite28 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array27 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram29 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1062, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite28 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype30 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 128, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype31 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype30 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array32 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite33 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array32 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram34 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1464, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite33 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str35 = internal constant [21 x i8] c"complex<long double>\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@.str37 = internal constant [20 x i8] c"complex long double\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.basictype38 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([20 x i8]* @.str37, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 192, i64 32, i64 0, i32 0, i32 3 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype39 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str8, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1348, i64 192, i64 32, i64 0, i32 1, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype38 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype40 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite122 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array41 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype40 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype38 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite42 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array41 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram43 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1314, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite42 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str44 = internal constant [12 x i8] c"long double\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.basictype45 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str44, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 96, i64 32, i64 0, i32 0, i32 4 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array46 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype40 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype45 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype45 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite47 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array46 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram48 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1352, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite47 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype49 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite171 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype50 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype49 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array51 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype40 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype50 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite52 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array51 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram53 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1480, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite52 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array54 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype40 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite55 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array54 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram56 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1484, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite55 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype57 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype45 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array58 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype57 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype40 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite59 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array58 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str60 = internal constant [5 x i8] c"real\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str61 = internal constant [24 x i8] c"_ZNSt7complexIeE4realEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram62 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str61, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1359, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite59 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str63 = internal constant [9 x i8] c"stddef.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str64 = internal constant [88 x i8] c"/developer/home2/zsth/projects/llvm.org/install/lib/gcc/i686-pc-linux-gnu/4.2.1/include\00", section "llvm.metadata"		; <[88 x i8]*> [#uses=1]
@llvm.dbg.compile_unit65 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([9 x i8]* @.str63, i32 0, i32 0), i8* getelementptr ([88 x i8]* @.str64, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str66 = internal constant [8 x i8] c"float_t\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype67 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str66, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit65 to %0*), i32 214, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype45 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str68 = internal constant [10 x i8] c"mathdef.h\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str69 = internal constant [18 x i8] c"/usr/include/bits\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.compile_unit70 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([10 x i8]* @.str68, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str69, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str71 = internal constant [9 x i8] c"double_t\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype72 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str71, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit70 to %0*), i32 36, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype67 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype73 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 96, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype72 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype74 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype73 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype75 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 192, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite122 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype76 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype75 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array77 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype74 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype76 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite78 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array77 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str79 = internal constant [25 x i8] c"_ZNKSt7complexIeE4realEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram80 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str79, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1363, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite78 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str81 = internal constant [5 x i8] c"imag\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str82 = internal constant [24 x i8] c"_ZNSt7complexIeE4imagEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram83 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str82, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1367, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite59 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str84 = internal constant [25 x i8] c"_ZNKSt7complexIeE4imagEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram85 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str84, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1371, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite78 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype86 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite122 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array87 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype86 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype40 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype45 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite88 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array87 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str89 = internal constant [10 x i8] c"operator=\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str90 = internal constant [21 x i8] c"_ZNSt7complexIeEaSEe\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram91 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str90, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1375, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite88 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str92 = internal constant [11 x i8] c"operator+=\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str93 = internal constant [21 x i8] c"_ZNSt7complexIeEpLEe\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram94 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str93, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1383, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite88 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str95 = internal constant [11 x i8] c"operator-=\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str96 = internal constant [21 x i8] c"_ZNSt7complexIeEmIEe\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram97 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str96, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1390, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite88 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str98 = internal constant [11 x i8] c"operator*=\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str99 = internal constant [21 x i8] c"_ZNSt7complexIeEmLEe\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram100 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str99, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1397, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite88 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str101 = internal constant [11 x i8] c"operator/=\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str102 = internal constant [21 x i8] c"_ZNSt7complexIeEdVEe\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram103 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str102, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1404, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite88 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite104 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 55, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype105 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite104 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype106 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype105 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array107 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype86 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype40 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype106 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite108 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array107 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram109 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1335, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite108 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram110 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1337, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite108 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram111 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1339, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite108 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram112 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1341, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite108 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram113 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1343, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite108 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype114 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 192, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype38 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype115 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype114 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array116 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype115 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype76 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite117 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array116 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str118 = internal constant [6 x i8] c"__rep\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str119 = internal constant [26 x i8] c"_ZNKSt7complexIeE5__repEv\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram120 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str119, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1345, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite117 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array121 = internal constant [20 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype39 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram43 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram48 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram53 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram56 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram62 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram80 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram83 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram85 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram91 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram94 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram97 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram100 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram103 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram109 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram110 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram111 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram112 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram113 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram120 to %0*)], section "llvm.metadata"		; <[20 x %0*]*> [#uses=1]
@llvm.dbg.composite122 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([21 x i8]* @.str35, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1310, i64 192, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([20 x %0*]* @llvm.dbg.array121 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype123 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 192, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite122 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype124 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype123 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array125 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype124 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite126 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array125 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram127 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1468, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite126 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype128 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype26 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array129 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype128 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite130 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array129 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str131 = internal constant [24 x i8] c"_ZNSt7complexIfE4realEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram132 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str131, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1046, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite130 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype133 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype26 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype134 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype133 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype135 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype49 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array136 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype134 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype135 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite137 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array136 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str138 = internal constant [25 x i8] c"_ZNKSt7complexIfE4realEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram139 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str138, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1050, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite137 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str140 = internal constant [24 x i8] c"_ZNSt7complexIfE4imagEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram141 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str140, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1054, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite130 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str142 = internal constant [25 x i8] c"_ZNKSt7complexIfE4imagEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram143 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str142, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1058, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite137 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype144 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite171 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array145 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype144 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype26 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite146 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array145 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str147 = internal constant [21 x i8] c"_ZNSt7complexIfEaSEf\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram148 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str147, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1069, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite146 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str149 = internal constant [21 x i8] c"_ZNSt7complexIfEpLEf\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram150 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str149, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1077, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite146 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str151 = internal constant [21 x i8] c"_ZNSt7complexIfEmIEf\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram152 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str151, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1084, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite146 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str153 = internal constant [21 x i8] c"_ZNSt7complexIfEmLEf\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram154 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str153, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1091, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite146 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str155 = internal constant [21 x i8] c"_ZNSt7complexIfEdVEf\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram156 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str155, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1098, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite146 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array157 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype144 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype21 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype106 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite158 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array157 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram159 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1029, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite158 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram160 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1031, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite158 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram161 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1033, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite158 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram162 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1035, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite158 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram163 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1037, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite158 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype164 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype19 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype165 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype164 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array166 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype165 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype135 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite167 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array166 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str168 = internal constant [26 x i8] c"_ZNKSt7complexIfE5__repEv\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram169 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str168, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1039, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite167 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array170 = internal constant [20 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype20 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram24 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram29 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram34 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram127 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram132 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram139 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram141 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram143 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram148 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram150 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram152 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram154 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram156 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram159 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram160 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram161 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram162 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram163 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram169 to %0*)], section "llvm.metadata"		; <[20 x %0*]*> [#uses=1]
@llvm.dbg.composite171 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str16, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1003, i64 64, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([20 x %0*]* @llvm.dbg.array170 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype172 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite171 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype173 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype172 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array174 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype173 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite175 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array174 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram176 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1472, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite175 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array177 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype124 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite178 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array177 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram179 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1476, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite178 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype180 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array181 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype180 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite182 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array181 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str183 = internal constant [24 x i8] c"_ZNSt7complexIdE4realEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram184 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str183, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1199, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite182 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype185 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 64, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype186 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype185 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype187 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype30 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array188 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype186 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype187 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite189 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array188 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str190 = internal constant [25 x i8] c"_ZNKSt7complexIdE4realEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram191 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str190, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1203, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite189 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str192 = internal constant [24 x i8] c"_ZNSt7complexIdE4imagEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram193 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str192, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1207, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite182 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str194 = internal constant [25 x i8] c"_ZNKSt7complexIdE4imagEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram195 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str194, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1211, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite189 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype196 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array197 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype196 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite198 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array197 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str199 = internal constant [21 x i8] c"_ZNSt7complexIdEaSEd\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram200 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str199, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1222, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite198 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str201 = internal constant [21 x i8] c"_ZNSt7complexIdEpLEd\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram202 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str201, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1230, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite198 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str203 = internal constant [21 x i8] c"_ZNSt7complexIdEmIEd\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram204 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str203, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1237, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite198 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str205 = internal constant [21 x i8] c"_ZNSt7complexIdEmLEd\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram206 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str205, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1244, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite198 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str207 = internal constant [21 x i8] c"_ZNSt7complexIdEdVEd\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram208 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str207, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1251, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite198 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array209 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype196 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype9 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype106 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite210 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array209 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram211 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1182, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite210 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram212 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1184, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite210 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram213 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1186, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite210 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram214 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1188, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite210 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram215 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1190, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite210 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype216 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 128, i64 64, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype217 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype216 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array218 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype217 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype187 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite219 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array218 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str220 = internal constant [26 x i8] c"_ZNKSt7complexIdE5__repEv\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram221 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str220, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1192, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite219 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array222 = internal constant [20 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram15 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram176 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram179 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram184 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram191 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram193 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram195 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram200 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram202 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram204 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram206 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram208 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram211 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram212 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram213 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram214 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram215 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram221 to %0*)], section "llvm.metadata"		; <[20 x %0*]*> [#uses=1]
@llvm.dbg.composite223 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str6, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1157, i64 128, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([20 x %0*]* @llvm.dbg.array222 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype224 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array225 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype224 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array225 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str226 = internal constant [22 x i8] c"_ZNSt7complexIdEC1ECd\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram227 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str226, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1161, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str230 = internal constant [4 x i8] c"__z\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.variable231 = internal constant %llvm.dbg.variable.type { i32 459009, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram227 to %0*), i8* getelementptr ([4 x i8]* @.str230, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1161, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram232 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str118, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str220, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1192, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite219 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram235 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str183, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1199, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite182 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram237 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str190, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1203, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite189 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram239 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str81, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str194, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1211, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite189 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str241 = internal constant [22 x i8] c"_ZNSt7complexIdEC1Edd\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram242 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str3, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str241, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1215, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite14 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str244 = internal constant [4 x i8] c"__r\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.subprogram248 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str199, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1222, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite198 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram252 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str101, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str207, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1251, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite198 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array255 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite256 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array255 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str257 = internal constant [14 x i8] c"random_double\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram258 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str257, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str257, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 55, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite256 to %0*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=0]
@.str259 = internal constant [7 x i8] c"result\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@_ZZL13random_doublevE4seed = internal global i32 1325		; <i32*> [#uses=14]
@.str266 = internal constant [19 x i8] c"polynomial<double>\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@.str268 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype269 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str268, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array270 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite271 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array270 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype272 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite271 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype273 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype272 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str274 = internal constant [17 x i8] c"_vptr.polynomial\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype275 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str274, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 84, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype273 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype276 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str277 = internal constant [8 x i8] c"m_coeff\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype278 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str277, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 164, i64 32, i64 32, i64 32, i32 2, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype276 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str279 = internal constant [13 x i8] c"unsigned int\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.basictype280 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str279, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str281 = internal constant [7 x i8] c"size_t\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype282 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str281, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit65 to %0*), i32 152, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str283 = internal constant [9 x i8] c"m_degree\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype284 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str283, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 167, i64 32, i64 32, i64 64, i32 2, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype285 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array286 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite287 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array286 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str288 = internal constant [11 x i8] c"polynomial\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram289 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 211, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite287 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype290 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype185 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array291 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype290 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite292 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array291 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram293 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 220, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite292 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array294 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype186 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite295 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array294 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram296 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 232, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite295 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype297 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 96, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype298 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype297 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array299 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype298 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite300 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array299 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram301 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 242, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite300 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array302 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite303 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array302 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str304 = internal constant [12 x i8] c"~polynomial\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram305 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 252, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite303 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype306 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array307 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype306 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype298 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite308 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array307 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str309 = internal constant [27 x i8] c"_ZN10polynomialIdEaSERKS0_\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram310 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str309, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 259, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite308 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array311 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype186 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite312 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array311 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str313 = internal constant [11 x i8] c"initialize\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str314 = internal constant [35 x i8] c"_ZN10polynomialIdE10initializeERKd\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram315 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str313, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str313, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str314, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 203, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite312 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array316 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype306 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite317 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array316 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str318 = internal constant [8 x i8] c"stretch\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str319 = internal constant [29 x i8] c"_ZN10polynomialIdE7stretchEj\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram320 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str318, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str318, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str319, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 276, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite317 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype321 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype297 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array322 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype321 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite323 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array322 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str324 = internal constant [7 x i8] c"degree\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str325 = internal constant [29 x i8] c"_ZNK10polynomialIdE6degreeEv\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram326 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str325, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 111, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite323 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array327 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype321 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite328 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array327 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str329 = internal constant [4 x i8] c"get\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str330 = internal constant [26 x i8] c"_ZNK10polynomialIdE3getEj\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram331 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str330, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 300, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite328 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array332 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype180 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite333 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array332 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str334 = internal constant [11 x i8] c"operator[]\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str335 = internal constant [23 x i8] c"_ZN10polynomialIdEixEj\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.subprogram336 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str335, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 306, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite333 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array337 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype12 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype321 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype186 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite338 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array337 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str339 = internal constant [11 x i8] c"operator()\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str340 = internal constant [26 x i8] c"_ZNK10polynomialIdEclERKd\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram341 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str339, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str339, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str340, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 313, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite338 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array342 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype321 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite343 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array342 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str344 = internal constant [10 x i8] c"operator-\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str345 = internal constant [24 x i8] c"_ZNK10polynomialIdEngEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram346 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str345, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 335, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite343 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str347 = internal constant [10 x i8] c"operator+\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str348 = internal constant [24 x i8] c"_ZNK10polynomialIdEpsEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram349 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str348, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 346, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite343 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array350 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype321 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype298 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite351 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array350 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str352 = internal constant [28 x i8] c"_ZNK10polynomialIdEplERKS0_\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram353 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str352, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 353, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite351 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str354 = internal constant [28 x i8] c"_ZNK10polynomialIdEmiERKS0_\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram355 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str354, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 376, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite351 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array356 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite357 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array356 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str358 = internal constant [5 x i8] c"log2\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str359 = internal constant [26 x i8] c"_ZN10polynomialIdE4log2Ej\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram360 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str358, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str358, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str359, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 404, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite357 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array361 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite362 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array361 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str363 = internal constant [10 x i8] c"flip_bits\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str364 = internal constant [32 x i8] c"_ZN10polynomialIdE9flip_bitsEjj\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram365 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str363, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str363, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str364, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 423, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite362 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array366 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite367 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array366 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str368 = internal constant [12 x i8] c"stretch_fft\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str369 = internal constant [34 x i8] c"_ZN10polynomialIdE11stretch_fftEv\00", section "llvm.metadata"		; <[34 x i8]*> [#uses=1]
@llvm.dbg.subprogram370 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str368, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str368, i32 0, i32 0), i8* getelementptr ([34 x i8]* @.str369, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 443, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite367 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str371 = internal constant [34 x i8] c"polynomial<std::complex<double> >\00", section "llvm.metadata"		; <[34 x i8]*> [#uses=1]
@llvm.dbg.derivedtype373 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str274, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 84, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype273 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype374 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str277, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 164, i64 32, i64 32, i64 32, i32 2, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype224 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype375 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str283, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 167, i64 32, i64 32, i64 64, i32 2, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype376 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array377 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite378 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array377 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram379 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 211, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite378 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array380 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype187 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite381 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array380 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram382 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 220, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite381 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array383 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite384 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array383 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram385 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 232, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite384 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype386 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 96, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype387 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype386 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array388 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype387 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite389 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array388 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram390 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 242, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite389 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array391 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite392 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array391 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram393 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 252, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite392 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype394 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array395 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype394 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype387 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite396 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array395 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str397 = internal constant [39 x i8] c"_ZN10polynomialISt7complexIdEEaSERKS2_\00", section "llvm.metadata"		; <[39 x i8]*> [#uses=1]
@llvm.dbg.subprogram398 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([39 x i8]* @.str397, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 259, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite396 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array399 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite400 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array399 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str401 = internal constant [49 x i8] c"_ZN10polynomialISt7complexIdEE10initializeERKS1_\00", section "llvm.metadata"		; <[49 x i8]*> [#uses=1]
@llvm.dbg.subprogram402 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str313, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str313, i32 0, i32 0), i8* getelementptr ([49 x i8]* @.str401, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 203, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite400 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array403 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype394 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite404 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array403 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str405 = internal constant [41 x i8] c"_ZN10polynomialISt7complexIdEE7stretchEj\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram406 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str318, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str318, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str405, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 276, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite404 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype407 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype386 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array408 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype407 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite409 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array408 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str410 = internal constant [41 x i8] c"_ZNK10polynomialISt7complexIdEE6degreeEv\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram411 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str410, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 111, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite409 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array412 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype407 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite413 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array412 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str414 = internal constant [38 x i8] c"_ZNK10polynomialISt7complexIdEE3getEj\00", section "llvm.metadata"		; <[38 x i8]*> [#uses=1]
@llvm.dbg.subprogram415 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([38 x i8]* @.str414, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 300, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite413 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array416 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype196 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite417 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array416 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str418 = internal constant [35 x i8] c"_ZN10polynomialISt7complexIdEEixEj\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram419 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str418, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 306, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite417 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array420 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype407 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite421 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array420 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str422 = internal constant [40 x i8] c"_ZNK10polynomialISt7complexIdEEclERKS1_\00", section "llvm.metadata"		; <[40 x i8]*> [#uses=1]
@llvm.dbg.subprogram423 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str339, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str339, i32 0, i32 0), i8* getelementptr ([40 x i8]* @.str422, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 313, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite421 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array424 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype407 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite425 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array424 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str426 = internal constant [36 x i8] c"_ZNK10polynomialISt7complexIdEEngEv\00", section "llvm.metadata"		; <[36 x i8]*> [#uses=1]
@llvm.dbg.subprogram427 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([36 x i8]* @.str426, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 335, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite425 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str428 = internal constant [36 x i8] c"_ZNK10polynomialISt7complexIdEEpsEv\00", section "llvm.metadata"		; <[36 x i8]*> [#uses=1]
@llvm.dbg.subprogram429 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([36 x i8]* @.str428, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 346, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite425 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array430 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype407 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype387 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite431 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array430 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str432 = internal constant [40 x i8] c"_ZNK10polynomialISt7complexIdEEplERKS2_\00", section "llvm.metadata"		; <[40 x i8]*> [#uses=1]
@llvm.dbg.subprogram433 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([40 x i8]* @.str432, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 353, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite431 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str434 = internal constant [40 x i8] c"_ZNK10polynomialISt7complexIdEEmiERKS2_\00", section "llvm.metadata"		; <[40 x i8]*> [#uses=1]
@llvm.dbg.subprogram435 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([40 x i8]* @.str434, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 376, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite431 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str436 = internal constant [38 x i8] c"_ZN10polynomialISt7complexIdEE4log2Ej\00", section "llvm.metadata"		; <[38 x i8]*> [#uses=1]
@llvm.dbg.subprogram437 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str358, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str358, i32 0, i32 0), i8* getelementptr ([38 x i8]* @.str436, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 404, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite357 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str438 = internal constant [44 x i8] c"_ZN10polynomialISt7complexIdEE9flip_bitsEjj\00", section "llvm.metadata"		; <[44 x i8]*> [#uses=1]
@llvm.dbg.subprogram439 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str363, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str363, i32 0, i32 0), i8* getelementptr ([44 x i8]* @.str438, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 423, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite362 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array440 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite441 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array440 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str442 = internal constant [46 x i8] c"_ZN10polynomialISt7complexIdEE11stretch_fftEv\00", section "llvm.metadata"		; <[46 x i8]*> [#uses=1]
@llvm.dbg.subprogram443 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str368, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str368, i32 0, i32 0), i8* getelementptr ([46 x i8]* @.str442, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 443, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite441 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str444 = internal constant [49 x i8] c"polynomial<std::complex<std::complex<double> > >\00", section "llvm.metadata"		; <[49 x i8]*> [#uses=1]
@llvm.dbg.composite445 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([49 x i8]* @.str444, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 84, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array446 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite445 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype387 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite447 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array446 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str448 = internal constant [12 x i8] c"bit_reverse\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str449 = internal constant [50 x i8] c"_ZN10polynomialISt7complexIdEE11bit_reverseERKS2_\00", section "llvm.metadata"		; <[50 x i8]*> [#uses=1]
@llvm.dbg.subprogram450 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([50 x i8]* @.str449, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 469, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite447 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype451 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite445 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype452 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype451 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array453 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite445 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype452 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite454 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array453 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str455 = internal constant [59 x i8] c"_ZN10polynomialISt7complexIdEE11bit_reverseERKS_IS0_IS1_EE\00", section "llvm.metadata"		; <[59 x i8]*> [#uses=1]
@llvm.dbg.subprogram456 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([59 x i8]* @.str455, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 483, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite454 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str457 = internal constant [4 x i8] c"fft\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str458 = internal constant [41 x i8] c"_ZN10polynomialISt7complexIdEE3fftERKS2_\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram459 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str457, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str457, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str458, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 497, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite447 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str460 = internal constant [12 x i8] c"inverse_fft\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str461 = internal constant [59 x i8] c"_ZN10polynomialISt7complexIdEE11inverse_fftERKS_IS0_IS1_EE\00", section "llvm.metadata"		; <[59 x i8]*> [#uses=1]
@llvm.dbg.subprogram462 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str460, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str460, i32 0, i32 0), i8* getelementptr ([59 x i8]* @.str461, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 535, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite454 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str463 = internal constant [10 x i8] c"operator*\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str464 = internal constant [40 x i8] c"_ZNK10polynomialISt7complexIdEEmlERKS2_\00", section "llvm.metadata"		; <[40 x i8]*> [#uses=1]
@llvm.dbg.subprogram465 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([40 x i8]* @.str464, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 576, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite431 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str466 = internal constant [39 x i8] c"_ZN10polynomialISt7complexIdEEpLERKS2_\00", section "llvm.metadata"		; <[39 x i8]*> [#uses=1]
@llvm.dbg.subprogram467 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([39 x i8]* @.str466, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 625, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite396 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str468 = internal constant [39 x i8] c"_ZN10polynomialISt7complexIdEEmIERKS2_\00", section "llvm.metadata"		; <[39 x i8]*> [#uses=1]
@llvm.dbg.subprogram469 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([39 x i8]* @.str468, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 636, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite396 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str470 = internal constant [39 x i8] c"_ZN10polynomialISt7complexIdEEmLERKS2_\00", section "llvm.metadata"		; <[39 x i8]*> [#uses=1]
@llvm.dbg.subprogram471 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([39 x i8]* @.str470, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 647, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite396 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array472 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite473 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array472 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str474 = internal constant [8 x i8] c"acquire\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str475 = internal constant [41 x i8] c"_ZN10polynomialISt7complexIdEE7acquireEv\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram476 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str475, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 181, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite473 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str477 = internal constant [8 x i8] c"release\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str478 = internal constant [41 x i8] c"_ZN10polynomialISt7complexIdEE7releaseEv\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram479 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str478, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 188, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite473 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array480 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype376 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype187 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite481 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array480 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str482 = internal constant [10 x i8] c"deep_copy\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str483 = internal constant [47 x i8] c"_ZN10polynomialISt7complexIdEE9deep_copyEPKS1_\00", section "llvm.metadata"		; <[47 x i8]*> [#uses=1]
@llvm.dbg.subprogram484 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([47 x i8]* @.str483, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 195, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite481 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array485 = internal constant [33 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype373 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype374 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype375 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram379 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram382 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram385 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram390 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram393 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram398 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram402 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram406 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram411 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram415 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram419 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram423 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram427 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram429 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram433 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram435 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram437 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram439 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram443 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram450 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram456 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram459 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram462 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram465 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram467 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram469 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram471 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram476 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram479 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram484 to %0*)], section "llvm.metadata"		; <[33 x %0*]*> [#uses=1]
@llvm.dbg.composite486 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([34 x i8]* @.str371, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 84, i64 96, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([33 x %0*]* @llvm.dbg.array485 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array487 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype298 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite488 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array487 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str489 = internal constant [38 x i8] c"_ZN10polynomialIdE11bit_reverseERKS0_\00", section "llvm.metadata"		; <[38 x i8]*> [#uses=1]
@llvm.dbg.subprogram490 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([38 x i8]* @.str489, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 469, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite488 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array491 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype387 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite492 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array491 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str493 = internal constant [52 x i8] c"_ZN10polynomialIdE11bit_reverseERKS_ISt7complexIdEE\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.subprogram494 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str493, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 483, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite492 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str495 = internal constant [29 x i8] c"_ZN10polynomialIdE3fftERKS0_\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram496 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str457, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str457, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str495, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 497, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite488 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str497 = internal constant [52 x i8] c"_ZN10polynomialIdE11inverse_fftERKS_ISt7complexIdEE\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.subprogram498 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str460, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str460, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str497, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 535, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite492 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str499 = internal constant [28 x i8] c"_ZNK10polynomialIdEmlERKS0_\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram500 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str499, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 576, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite351 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str501 = internal constant [27 x i8] c"_ZN10polynomialIdEpLERKS0_\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram502 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str501, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 625, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite308 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str503 = internal constant [27 x i8] c"_ZN10polynomialIdEmIERKS0_\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram504 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str503, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 636, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite308 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str505 = internal constant [27 x i8] c"_ZN10polynomialIdEmLERKS0_\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram506 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str98, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str505, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 647, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite308 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array507 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite508 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array507 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str509 = internal constant [29 x i8] c"_ZN10polynomialIdE7acquireEv\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram510 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str509, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 181, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite508 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str511 = internal constant [29 x i8] c"_ZN10polynomialIdE7releaseEv\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram512 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str511, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 188, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite508 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array513 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype290 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite514 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array513 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str515 = internal constant [33 x i8] c"_ZN10polynomialIdE9deep_copyEPKd\00", section "llvm.metadata"		; <[33 x i8]*> [#uses=1]
@llvm.dbg.subprogram516 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([33 x i8]* @.str515, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 195, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite514 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array517 = internal constant [33 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype275 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype278 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype284 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram289 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram293 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram296 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram301 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram305 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram310 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram315 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram320 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram326 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram331 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram336 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram341 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram346 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram349 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram353 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram355 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram360 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram365 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram370 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram490 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram494 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram496 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram498 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram500 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram502 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram504 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram506 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram510 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram512 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram516 to %0*)], section "llvm.metadata"		; <[33 x %0*]*> [#uses=1]
@llvm.dbg.composite518 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([19 x i8]* @.str266, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 84, i64 96, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([33 x %0*]* @llvm.dbg.array517 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype519 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array520 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype180 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype519 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite521 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array520 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram522 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str335, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 306, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite521 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram527 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str325, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 111, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite323 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram530 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str418, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 306, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite417 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array534 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype196 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype224 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite535 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array534 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str536 = internal constant [19 x i8] c"operator*=<double>\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@.str537 = internal constant [35 x i8] c"_ZNSt7complexIdEmLIdEERS0_RKS_IT_E\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram538 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([19 x i8]* @.str536, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str536, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str537, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1286, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite535 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str541 = internal constant [4 x i8] c"__t\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.variable542 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram538 to %0*), i8* getelementptr ([4 x i8]* @.str541, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1288, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram543 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([33 x i8]* @.str515, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 195, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite514 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram549 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str358, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str358, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str359, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 404, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite357 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array555 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite556 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array555 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str557 = internal constant [18 x i8] c"operator*<double>\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@.str558 = internal constant [32 x i8] c"_ZStmlIdESt7complexIT_ERKS2_S4_\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram559 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str557, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str557, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str558, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 378, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite556 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable564 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram559 to %0*), i8* getelementptr ([4 x i8]* @.str244, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 380, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram565 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str482, i32 0, i32 0), i8* getelementptr ([47 x i8]* @.str483, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 195, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite481 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram569 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str324, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str410, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 111, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite409 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array572 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype31 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite573 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array572 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str574 = internal constant [18 x i8] c"operator-<double>\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@.str575 = internal constant [29 x i8] c"_ZStngIdESt7complexIT_ERKS2_\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram576 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str574, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str574, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str575, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 443, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite573 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram578 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str330, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 300, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite328 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram581 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str363, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str363, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str364, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 423, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite362 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str591 = internal constant [19 x i8] c"operator/=<double>\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@.str592 = internal constant [35 x i8] c"_ZNSt7complexIdEdVIdEERS0_RKS_IT_E\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram593 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([19 x i8]* @.str591, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str591, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str592, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1297, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite535 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable596 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram593 to %0*), i8* getelementptr ([4 x i8]* @.str541, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1299, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str597 = internal constant [18 x i8] c"operator/<double>\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@.str598 = internal constant [32 x i8] c"_ZStdvIdESt7complexIT_ERKS2_S4_\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram599 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str597, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str597, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str598, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 408, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite556 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable602 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram599 to %0*), i8* getelementptr ([4 x i8]* @.str244, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 410, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str603 = internal constant [19 x i8] c"operator+=<double>\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@.str604 = internal constant [35 x i8] c"_ZNSt7complexIdEpLIdEERS0_RKS_IT_E\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram605 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([19 x i8]* @.str603, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str603, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str604, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1268, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite535 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str608 = internal constant [18 x i8] c"operator+<double>\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@.str609 = internal constant [32 x i8] c"_ZStplIdESt7complexIT_ERKS2_S4_\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram610 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str608, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str608, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str609, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 318, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite556 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable613 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram610 to %0*), i8* getelementptr ([4 x i8]* @.str244, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 320, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str614 = internal constant [19 x i8] c"operator-=<double>\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@.str615 = internal constant [35 x i8] c"_ZNSt7complexIdEmIIdEERS0_RKS_IT_E\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram616 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([19 x i8]* @.str614, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str614, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str615, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 1277, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite535 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str619 = internal constant [32 x i8] c"_ZStmiIdESt7complexIT_ERKS2_S4_\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram620 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str574, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str574, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str619, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 348, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite556 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable623 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram620 to %0*), i8* getelementptr ([4 x i8]* @.str244, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 350, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram624 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str329, i32 0, i32 0), i8* getelementptr ([38 x i8]* @.str414, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 300, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite413 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array627 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite628 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array627 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str629 = internal constant [42 x i8] c"__static_initialization_and_destruction_0\00", section "llvm.metadata"		; <[42 x i8]*> [#uses=1]
@llvm.dbg.subprogram630 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([42 x i8]* @.str629, i32 0, i32 0), i8* getelementptr ([42 x i8]* @.str629, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 703, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite628 to %0*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=0]
@.str635 = internal constant [9 x i8] c"iostream\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit636 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([9 x i8]* @.str635, i32 0, i32 0), i8* getelementptr ([110 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@_ZStL8__ioinit = internal global %"struct.std::allocator<char>" zeroinitializer		; <%"struct.std::allocator<char>"*> [#uses=2]
@.str638 = internal constant [115 x i8] c"/developer/home2/zsth/projects/llvm.org/install/lib/gcc/i686-pc-linux-gnu/4.2.1/../../../../include/c++/4.2.1/bits\00", section "llvm.metadata"		; <[115 x i8]*> [#uses=1]
@__dso_handle = external global i8*		; <i8**> [#uses=1]
@_ZGVN10polynomialIdE4PI2IE = weak global i64 0, align 8		; <i64*> [#uses=1]
@_ZN10polynomialIdE4PI2IE = weak global %"struct.std::complex<double>" zeroinitializer		; <%"struct.std::complex<double>"*> [#uses=3]
@llvm.dbg.array654 = internal constant [1 x %0*] zeroinitializer, section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite655 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array654 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str656 = internal constant [16 x i8] c"_GLOBAL__I_main\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram657 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str656, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str656, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 704, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite655 to %0*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype658 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array659 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype658 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite660 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array659 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str661 = internal constant [8 x i8] c"__tcf_0\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.subprogram662 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str661, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str661, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit636 to %0*), i32 77, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite660 to %0*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram665 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str511, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 188, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite508 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str667 = internal constant [23 x i8] c"_ZN10polynomialIdED0Ev\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.subprogram668 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str667, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 252, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite508 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@_ZTV10polynomialIdE = weak constant [4 x i32 (...)*] [i32 (...)* null, i32 (...)* bitcast (%struct.__class_type_info_pseudo* @_ZTI10polynomialIdE to i32 (...)*), i32 (...)* bitcast (void (%"struct.polynomial<double>"*)* @_ZN10polynomialIdED1Ev to i32 (...)*), i32 (...)* bitcast (void (%"struct.polynomial<double>"*)* @_ZN10polynomialIdED0Ev to i32 (...)*)], align 8		; <[4 x i32 (...)*]*> [#uses=1]
@_ZTI10polynomialIdE = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([16 x i8]* @_ZTS10polynomialIdE, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTVN10__cxxabiv117__class_type_infoE = external constant [0 x i32 (...)*]		; <[0 x i32 (...)*]*> [#uses=1]
@_ZTS10polynomialIdE = weak constant [16 x i8] c"10polynomialIdE\00"		; <[16 x i8]*> [#uses=1]
@.str671 = internal constant [5 x i8] c"char\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.basictype672 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str671, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 6 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str676 = internal constant [11 x i8] c"<built-in>\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.compile_unit677 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([11 x i8]* @.str676, i32 0, i32 0), i8* getelementptr ([42 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@llvm.dbg.array680 = internal constant [0 x %0*] zeroinitializer, section "llvm.metadata"		; <[0 x %0*]*> [#uses=1]
@.str690 = internal constant [23 x i8] c"_ZN10polynomialIdED1Ev\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.subprogram691 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str690, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 252, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite508 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram693 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str477, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str478, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 188, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite473 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str695 = internal constant [35 x i8] c"_ZN10polynomialISt7complexIdEED0Ev\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram696 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str695, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 252, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite473 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@_ZTV10polynomialISt7complexIdEE = weak constant [4 x i32 (...)*] [i32 (...)* null, i32 (...)* bitcast (%struct.__class_type_info_pseudo* @_ZTI10polynomialISt7complexIdEE to i32 (...)*), i32 (...)* bitcast (void (%"struct.polynomial<std::complex<double> >"*)* @_ZN10polynomialISt7complexIdEED1Ev to i32 (...)*), i32 (...)* bitcast (void (%"struct.polynomial<std::complex<double> >"*)* @_ZN10polynomialISt7complexIdEED0Ev to i32 (...)*)], align 8		; <[4 x i32 (...)*]*> [#uses=1]
@_ZTI10polynomialISt7complexIdEE = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([28 x i8]* @_ZTS10polynomialISt7complexIdEE, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTS10polynomialISt7complexIdEE = weak constant [28 x i8] c"10polynomialISt7complexIdEE\00"		; <[28 x i8]*> [#uses=1]
@.str707 = internal constant [35 x i8] c"_ZN10polynomialISt7complexIdEED1Ev\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram708 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str304, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str707, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 252, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite473 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram710 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str509, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 181, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite508 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str712 = internal constant [23 x i8] c"_ZN10polynomialIdEC1Ej\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.subprogram713 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str712, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 211, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite287 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str716 = internal constant [27 x i8] c"_ZN10polynomialIdEC1ERKS0_\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram717 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str716, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 242, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite300 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram720 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str318, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str318, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str319, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 276, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite317 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram729 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str309, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 259, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite308 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram732 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str474, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str475, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 181, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite473 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str734 = internal constant [35 x i8] c"_ZN10polynomialISt7complexIdEEC1Ej\00", section "llvm.metadata"		; <[35 x i8]*> [#uses=1]
@llvm.dbg.subprogram735 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str288, i32 0, i32 0), i8* getelementptr ([35 x i8]* @.str734, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 211, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite378 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram738 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([38 x i8]* @.str489, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 469, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite488 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable744 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram738 to %0*), i8* getelementptr ([7 x i8]* @.str259, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 473, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram747 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str448, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str493, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 483, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite492 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable751 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram747 to %0*), i8* getelementptr ([7 x i8]* @.str259, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 487, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram753 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([39 x i8]* @.str397, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 259, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite396 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram756 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str368, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str368, i32 0, i32 0), i8* getelementptr ([34 x i8]* @.str369, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 443, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite367 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str759 = internal constant [35 x i8] c"overflow in fft polynomial stretch\00"		; <[35 x i8]*> [#uses=1]
@_ZTISt14overflow_error = weak constant %struct.__si_class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv120__si_class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([19 x i8]* @_ZTSSt14overflow_error, i32 0, i32 0) }, %"struct.std::type_info"* bitcast (%struct.__si_class_type_info_pseudo* @_ZTISt13runtime_error to %"struct.std::type_info"*) }		; <%struct.__si_class_type_info_pseudo*> [#uses=2]
@_ZTVN10__cxxabiv120__si_class_type_infoE = external constant [0 x i32 (...)*]		; <[0 x i32 (...)*]*> [#uses=1]
@_ZTSSt14overflow_error = weak constant [19 x i8] c"St14overflow_error\00"		; <[19 x i8]*> [#uses=1]
@_ZTISt13runtime_error = external constant %struct.__si_class_type_info_pseudo		; <%struct.__si_class_type_info_pseudo*> [#uses=1]
@.str769 = internal constant [10 x i8] c"stdexcept\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.compile_unit770 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([10 x i8]* @.str769, i32 0, i32 0), i8* getelementptr ([110 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str773 = internal constant [15 x i8] c"overflow_error\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@.str775 = internal constant [14 x i8] c"runtime_error\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str777 = internal constant [10 x i8] c"exception\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.compile_unit778 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([10 x i8]* @.str777, i32 0, i32 0), i8* getelementptr ([110 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str780 = internal constant [16 x i8] c"_vptr.exception\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.derivedtype781 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str780, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit778 to %0*), i32 57, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype273 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype782 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite800 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array783 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype782 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite784 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array783 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram785 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str777, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str777, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit778 to %0*), i32 59, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite784 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array786 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype782 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite787 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array786 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str788 = internal constant [11 x i8] c"~exception\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram789 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str788, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str788, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit778 to %0*), i32 60, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite787 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype790 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype791 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype790 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype792 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite800 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype793 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype792 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array794 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype793 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite795 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array794 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str796 = internal constant [5 x i8] c"what\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str797 = internal constant [24 x i8] c"_ZNKSt9exception4whatEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram798 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str796, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str796, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str797, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit778 to %0*), i32 63, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite795 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array799 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype781 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram785 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram789 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram798 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite800 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str777, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit778 to %0*), i32 57, i64 32, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array799 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype801 = internal constant %llvm.dbg.derivedtype.type { i32 458780, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite800 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str802 = internal constant [12 x i8] c"stringfwd.h\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.compile_unit803 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([12 x i8]* @.str802, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str804 = internal constant [64 x i8] c"basic_string<char,std::char_traits<char>,std::allocator<char> >\00", section "llvm.metadata"		; <[64 x i8]*> [#uses=1]
@.str806 = internal constant [15 x i8] c"basic_string.h\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.compile_unit807 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([15 x i8]* @.str806, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str808 = internal constant [13 x i8] c"_Alloc_hider\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str810 = internal constant [16 x i8] c"allocator<char>\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str812 = internal constant [16 x i8] c"new_allocator.h\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str813 = internal constant [114 x i8] c"/developer/home2/zsth/projects/llvm.org/install/lib/gcc/i686-pc-linux-gnu/4.2.1/../../../../include/c++/4.2.1/ext\00", section "llvm.metadata"		; <[114 x i8]*> [#uses=1]
@llvm.dbg.compile_unit814 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([16 x i8]* @.str812, i32 0, i32 0), i8* getelementptr ([114 x i8]* @.str813, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str815 = internal constant [20 x i8] c"new_allocator<char>\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.derivedtype817 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite877 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array818 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite819 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array818 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str820 = internal constant [14 x i8] c"new_allocator\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram821 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str820, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str820, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 68, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite819 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype822 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite877 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype823 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype822 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array824 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype823 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite825 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array824 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram826 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str820, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str820, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 70, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite825 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite827 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 54, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype828 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite827 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype829 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype828 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array830 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype829 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite831 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array830 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram832 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str820, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str820, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 73, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite831 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array833 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite834 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array833 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str835 = internal constant [15 x i8] c"~new_allocator\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram836 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str835, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str835, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 75, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite834 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype837 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype838 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype822 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype839 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array840 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype838 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype839 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite841 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array840 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str842 = internal constant [8 x i8] c"address\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str843 = internal constant [44 x i8] c"_ZNK9__gnu_cxx13new_allocatorIcE7addressERc\00", section "llvm.metadata"		; <[44 x i8]*> [#uses=1]
@llvm.dbg.subprogram844 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str842, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str842, i32 0, i32 0), i8* getelementptr ([44 x i8]* @.str843, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 78, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite841 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype845 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype790 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array846 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype838 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype845 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite847 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array846 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str848 = internal constant [45 x i8] c"_ZNK9__gnu_cxx13new_allocatorIcE7addressERKc\00", section "llvm.metadata"		; <[45 x i8]*> [#uses=1]
@llvm.dbg.subprogram849 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str842, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str842, i32 0, i32 0), i8* getelementptr ([45 x i8]* @.str848, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 81, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite847 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype850 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array851 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype850 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite852 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array851 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str853 = internal constant [9 x i8] c"allocate\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str854 = internal constant [46 x i8] c"_ZN9__gnu_cxx13new_allocatorIcE8allocateEjPKv\00", section "llvm.metadata"		; <[46 x i8]*> [#uses=1]
@llvm.dbg.subprogram855 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str853, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str853, i32 0, i32 0), i8* getelementptr ([46 x i8]* @.str854, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 86, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite852 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array856 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite857 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array856 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str858 = internal constant [11 x i8] c"deallocate\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str859 = internal constant [48 x i8] c"_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcj\00", section "llvm.metadata"		; <[48 x i8]*> [#uses=1]
@llvm.dbg.subprogram860 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str858, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str858, i32 0, i32 0), i8* getelementptr ([48 x i8]* @.str859, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 96, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite857 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array861 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype838 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite862 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array861 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str863 = internal constant [9 x i8] c"max_size\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str864 = internal constant [44 x i8] c"_ZNK9__gnu_cxx13new_allocatorIcE8max_sizeEv\00", section "llvm.metadata"		; <[44 x i8]*> [#uses=1]
@llvm.dbg.subprogram865 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str863, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str863, i32 0, i32 0), i8* getelementptr ([44 x i8]* @.str864, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 100, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite862 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array866 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype845 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite867 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array866 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str868 = internal constant [10 x i8] c"construct\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str869 = internal constant [48 x i8] c"_ZN9__gnu_cxx13new_allocatorIcE9constructEPcRKc\00", section "llvm.metadata"		; <[48 x i8]*> [#uses=1]
@llvm.dbg.subprogram870 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str868, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str868, i32 0, i32 0), i8* getelementptr ([48 x i8]* @.str869, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 106, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite867 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array871 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype817 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite872 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array871 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str873 = internal constant [8 x i8] c"destroy\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str874 = internal constant [43 x i8] c"_ZN9__gnu_cxx13new_allocatorIcE7destroyEPc\00", section "llvm.metadata"		; <[43 x i8]*> [#uses=1]
@llvm.dbg.subprogram875 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str873, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str873, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str874, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 110, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite872 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array876 = internal constant [11 x %0*] [%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram821 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram826 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram832 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram836 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram844 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram849 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram855 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram860 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram865 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram870 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram875 to %0*)], section "llvm.metadata"		; <[11 x %0*]*> [#uses=1]
@llvm.dbg.composite877 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([20 x i8]* @.str815, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit814 to %0*), i32 54, i64 8, i64 8, i64 0, i32 0, %0* null, %0* bitcast ([11 x %0*]* @llvm.dbg.array876 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype878 = internal constant %llvm.dbg.derivedtype.type { i32 458780, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit803 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite877 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype879 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite902 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array880 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype879 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite881 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array880 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str882 = internal constant [12 x i8] c"allocator.h\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.compile_unit883 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([12 x i8]* @.str882, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str884 = internal constant [10 x i8] c"allocator\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.subprogram885 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str884, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str884, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit883 to %0*), i32 100, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite881 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype886 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite902 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype887 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype886 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array888 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype879 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite889 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array888 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram890 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str884, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str884, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit883 to %0*), i32 102, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite889 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite891 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit803 to %0*), i32 49, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype892 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite891 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype893 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype892 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array894 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype879 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype893 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite895 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array894 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram896 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str884, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str884, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit883 to %0*), i32 106, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite895 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array897 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype879 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite898 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array897 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str899 = internal constant [11 x i8] c"~allocator\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram900 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str899, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str899, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit883 to %0*), i32 108, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite898 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array901 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype878 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram885 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram890 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram896 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram900 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite902 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str810, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit803 to %0*), i32 49, i64 8, i64 8, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array901 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype903 = internal constant %llvm.dbg.derivedtype.type { i32 458780, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite902 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str904 = internal constant [5 x i8] c"_M_p\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.derivedtype905 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str904, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 264, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype906 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite911 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array907 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype906 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite908 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array907 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram909 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str808, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str808, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 261, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite908 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array910 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype903 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype905 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram909 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite911 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str808, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 260, i64 32, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array910 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str912 = internal constant [12 x i8] c"_M_dataplus\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype913 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str912, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 276, i64 32, i64 32, i64 0, i32 1, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite911 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str914 = internal constant [7 x i8] c"string\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype915 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str914, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit803 to %0*), i32 56, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1693 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype916 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype915 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype917 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype916 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array918 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite919 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array918 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str920 = internal constant [8 x i8] c"_M_data\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str921 = internal constant [17 x i8] c"_ZNKSs7_M_dataEv\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram922 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str920, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str920, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str921, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 279, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite919 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype923 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1693 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array924 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite925 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array924 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str926 = internal constant [17 x i8] c"_ZNSs7_M_dataEPc\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram927 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str920, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str920, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str926, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 283, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite925 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str928 = internal constant [5 x i8] c"_Rep\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str930 = internal constant [10 x i8] c"_Rep_base\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str932 = internal constant [10 x i8] c"_M_length\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype933 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str932, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 149, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str934 = internal constant [12 x i8] c"_M_capacity\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype935 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str934, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 150, i64 32, i64 32, i64 32, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str936 = internal constant [10 x i8] c"ptrdiff_t\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype937 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str936, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit677 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype938 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype937 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str939 = internal constant [8 x i8] c"types.h\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.compile_unit940 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([8 x i8]* @.str939, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str69, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str941 = internal constant [10 x i8] c"__int32_t\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype942 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str941, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 43, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype938 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str943 = internal constant [8 x i8] c"__pid_t\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype944 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str943, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 145, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype942 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str945 = internal constant [10 x i8] c"__daddr_t\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype946 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str945, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 154, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype944 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str947 = internal constant [8 x i8] c"__key_t\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype948 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str947, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 157, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype946 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str949 = internal constant [12 x i8] c"__clockid_t\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype950 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str949, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 158, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype948 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str951 = internal constant [10 x i8] c"__ssize_t\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype952 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str951, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 181, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype950 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str953 = internal constant [11 x i8] c"__intptr_t\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype954 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str953, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 189, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype952 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str955 = internal constant [12 x i8] c"_G_config.h\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str956 = internal constant [13 x i8] c"/usr/include\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.compile_unit957 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([12 x i8]* @.str955, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str958 = internal constant [11 x i8] c"_G_int32_t\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype959 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str958, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit957 to %0*), i32 55, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype954 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str960 = internal constant [11 x i8] c"nl_types.h\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.compile_unit961 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([11 x i8]* @.str960, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str962 = internal constant [8 x i8] c"nl_item\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype963 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str962, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit961 to %0*), i32 34, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype959 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str964 = internal constant [7 x i8] c"time.h\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.compile_unit965 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([7 x i8]* @.str964, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str966 = internal constant [10 x i8] c"clockid_t\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype967 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str966, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit965 to %0*), i32 77, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype963 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str968 = internal constant [6 x i8] c"pid_t\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype969 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str968, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit965 to %0*), i32 105, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype967 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str970 = internal constant [15 x i8] c"__sig_atomic_t\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype971 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str970, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit965 to %0*), i32 413, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype969 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str972 = internal constant [15 x i8] c"pthreadtypes.h\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.compile_unit973 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([15 x i8]* @.str972, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str69, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str974 = internal constant [15 x i8] c"pthread_once_t\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype975 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str974, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit973 to %0*), i32 109, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype971 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype976 = internal constant %llvm.dbg.derivedtype.type { i32 458805, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype975 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str977 = internal constant [19 x i8] c"pthread_spinlock_t\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.derivedtype978 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([19 x i8]* @.str977, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit973 to %0*), i32 142, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype976 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str979 = internal constant [10 x i8] c"pthread.h\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.compile_unit980 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([10 x i8]* @.str979, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str981 = internal constant [8 x i8] c"ssize_t\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype982 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str981, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit980 to %0*), i32 1101, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype978 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str983 = internal constant [9 x i8] c"unistd.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit984 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([9 x i8]* @.str983, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str985 = internal constant [9 x i8] c"intptr_t\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.derivedtype986 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str985, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit984 to %0*), i32 226, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype982 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str987 = internal constant [15 x i8] c"gthr-default.h\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@.str988 = internal constant [133 x i8] c"/developer/home2/zsth/projects/llvm.org/install/lib/gcc/i686-pc-linux-gnu/4.2.1/../../../../include/c++/4.2.1/i686-pc-linux-gnu/bits\00", section "llvm.metadata"		; <[133 x i8]*> [#uses=1]
@llvm.dbg.compile_unit989 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([15 x i8]* @.str987, i32 0, i32 0), i8* getelementptr ([133 x i8]* @.str988, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str990 = internal constant [17 x i8] c"__gthread_once_t\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.derivedtype991 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str990, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit989 to %0*), i32 46, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype986 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype992 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str941, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 43, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype991 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str993 = internal constant [9 x i8] c"stdint.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit994 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([9 x i8]* @.str993, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str995 = internal constant [8 x i8] c"int32_t\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype996 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str995, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit994 to %0*), i32 38, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype992 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str997 = internal constant [14 x i8] c"int_least32_t\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.derivedtype998 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str997, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit994 to %0*), i32 67, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype996 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str999 = internal constant [13 x i8] c"int_fast16_t\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1000 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str999, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit994 to %0*), i32 91, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype998 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1001 = internal constant [13 x i8] c"int_fast32_t\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1002 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1001, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit994 to %0*), i32 97, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1000 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1003 = internal constant [11 x i8] c"postypes.h\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1004 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([11 x i8]* @.str1003, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1005 = internal constant [11 x i8] c"streamsize\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1006 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1005, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1004 to %0*), i32 72, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1002 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1007 = internal constant [17 x i8] c"/usr/include/sys\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1008 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([8 x i8]* @.str939, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1007, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1009 = internal constant [8 x i8] c"daddr_t\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1010 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1009, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1008 to %0*), i32 105, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1006 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1011 = internal constant [6 x i8] c"key_t\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1012 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1011, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1008 to %0*), i32 117, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1010 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1013 = internal constant [11 x i8] c"register_t\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1014 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1013, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1008 to %0*), i32 204, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1012 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1015 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str936, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit677 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1014 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1016 = internal constant [9 x i8] c"stdlib.h\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1017 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([9 x i8]* @.str1016, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1018 = internal constant [13 x i8] c"_Atomic_word\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1019 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1018, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1017 to %0*), i32 962, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1015 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1020 = internal constant [12 x i8] c"_M_refcount\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1021 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str1020, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 151, i64 32, i64 32, i64 64, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1019 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1022 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype933 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype935 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1021 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1023 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str930, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 148, i64 96, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1022 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype1024 = internal constant %llvm.dbg.derivedtype.type { i32 458780, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1023 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1025 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1091 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1026 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1025 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite1027 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array1026 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1028 = internal constant [13 x i8] c"_S_empty_rep\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str1029 = internal constant [27 x i8] c"_ZNSs4_Rep12_S_empty_repEv\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram1030 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1028, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1028, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str1029, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 180, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1027 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1031 = internal constant [5 x i8] c"bool\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.basictype1032 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1031, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 2 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype1033 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 96, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1091 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1034 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1033 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1035 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype1032 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1034 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1036 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1035 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1037 = internal constant [13 x i8] c"_M_is_leaked\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str1038 = internal constant [28 x i8] c"_ZNKSs4_Rep12_M_is_leakedEv\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram1039 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1037, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1037, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str1038, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 190, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1036 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1040 = internal constant [13 x i8] c"_M_is_shared\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str1041 = internal constant [28 x i8] c"_ZNKSs4_Rep12_M_is_sharedEv\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram1042 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1040, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1040, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str1041, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 194, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1036 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype1043 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1091 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1044 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1043 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1045 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1044 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1046 = internal constant [14 x i8] c"_M_set_leaked\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str1047 = internal constant [28 x i8] c"_ZNSs4_Rep13_M_set_leakedEv\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram1048 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1046, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1046, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str1047, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 198, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1045 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1049 = internal constant [16 x i8] c"_M_set_sharable\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str1050 = internal constant [30 x i8] c"_ZNSs4_Rep15_M_set_sharableEv\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram1051 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str1049, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1049, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1050, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 202, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1045 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1052 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1043 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1053 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1052 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1054 = internal constant [27 x i8] c"_M_set_length_and_sharable\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@.str1055 = internal constant [41 x i8] c"_ZNSs4_Rep26_M_set_length_and_sharableEj\00", section "llvm.metadata"		; <[41 x i8]*> [#uses=1]
@llvm.dbg.subprogram1056 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([27 x i8]* @.str1054, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str1054, i32 0, i32 0), i8* getelementptr ([41 x i8]* @.str1055, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 206, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1053 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1057 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1043 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1058 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1057 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1059 = internal constant [11 x i8] c"_M_refdata\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str1060 = internal constant [25 x i8] c"_ZNSs4_Rep10_M_refdataEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram1061 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1059, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1059, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str1060, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 216, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1058 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1062 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1043 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1063 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1062 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1064 = internal constant [8 x i8] c"_M_grab\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1065 = internal constant [30 x i8] c"_ZNSs4_Rep7_M_grabERKSaIcES2_\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram1066 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1064, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1064, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1065, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 220, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1063 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1067 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1043 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1068 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1067 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1069 = internal constant [17 x i8] c"basic_string.tcc\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1070 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([17 x i8]* @.str1069, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1071 = internal constant [10 x i8] c"_S_create\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str1072 = internal constant [31 x i8] c"_ZNSs4_Rep9_S_createEjjRKSaIcE\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram1073 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str1071, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str1071, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str1072, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 529, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1068 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1074 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1043 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1075 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1074 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1076 = internal constant [11 x i8] c"_M_dispose\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str1077 = internal constant [31 x i8] c"_ZNSs4_Rep10_M_disposeERKSaIcE\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram1078 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1076, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1076, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str1077, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 231, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1075 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1079 = internal constant [11 x i8] c"_M_destroy\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str1080 = internal constant [31 x i8] c"_ZNSs4_Rep10_M_destroyERKSaIcE\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram1081 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1079, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1079, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str1080, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 427, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1075 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1082 = internal constant [11 x i8] c"_M_refcopy\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str1083 = internal constant [25 x i8] c"_ZNSs4_Rep10_M_refcopyEv\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram1084 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1082, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1082, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str1083, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 245, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1058 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1085 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1043 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1086 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1085 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1087 = internal constant [9 x i8] c"_M_clone\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str1088 = internal constant [29 x i8] c"_ZNSs4_Rep8_M_cloneERKSaIcEj\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram1089 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str1087, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str1087, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str1088, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 606, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1086 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1090 = internal constant [14 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1024 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1030 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1039 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1042 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1048 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1051 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1056 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1061 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1066 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1073 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1078 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1081 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1084 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1089 to %0*)], section "llvm.metadata"		; <[14 x %0*]*> [#uses=1]
@llvm.dbg.composite1091 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str928, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 155, i64 96, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([14 x %0*]* @llvm.dbg.array1090 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype1092 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1091 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1093 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1092 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1094 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1093 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1095 = internal constant [16 x i8] c"_ZNKSs6_M_repEv\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1096 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str1087, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str1087, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1095, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 287, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1094 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1097 = internal constant [15 x i8] c"stl_iterator.h\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1098 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([15 x i8]* @.str1097, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1099 = internal constant [97 x i8] c"__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >\00", section "llvm.metadata"		; <[97 x i8]*> [#uses=1]
@.str1101 = internal constant [11 x i8] c"_M_current\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1102 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1101, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 639, i64 32, i64 32, i64 0, i32 2, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1103 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1104 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1103 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1105 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1104 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1106 = internal constant [18 x i8] c"__normal_iterator\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1107 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str1106, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1106, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 650, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1105 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1108 = internal constant [10 x i8] c"__caddr_t\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1109 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str1108, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit940 to %0*), i32 188, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1110 = internal constant [15 x i8] c"__gnuc_va_list\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1111 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str1110, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit957 to %0*), i32 58, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1109 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1112 = internal constant [8 x i8] c"libio.h\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1113 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([8 x i8]* @.str1112, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str956, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1114 = internal constant [8 x i8] c"va_list\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1115 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1114, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1113 to %0*), i32 491, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1111 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1116 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1115 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1117 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1116 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1118 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1103 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1117 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1119 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1118 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1120 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str1106, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1106, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 653, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1119 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite1121 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 637, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype1122 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1121 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1123 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1122 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1124 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1103 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1123 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1125 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1124 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1126 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str1106, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1106, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 660, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1125 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype1127 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1128 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1127 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1129 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype839 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1128 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1130 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1129 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1131 = internal constant [44 x i8] c"_ZNK9__gnu_cxx17__normal_iteratorIPcSsEdeEv\00", section "llvm.metadata"		; <[44 x i8]*> [#uses=1]
@llvm.dbg.subprogram1132 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([44 x i8]* @.str1131, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 665, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1130 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1133 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1128 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1134 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1133 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1135 = internal constant [11 x i8] c"operator->\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str1136 = internal constant [44 x i8] c"_ZNK9__gnu_cxx17__normal_iteratorIPcSsEptEv\00", section "llvm.metadata"		; <[44 x i8]*> [#uses=1]
@llvm.dbg.subprogram1137 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1135, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1135, i32 0, i32 0), i8* getelementptr ([44 x i8]* @.str1136, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 669, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1134 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype1138 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1139 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1138 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1103 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1140 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1139 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1141 = internal constant [11 x i8] c"operator++\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str1142 = internal constant [43 x i8] c"_ZN9__gnu_cxx17__normal_iteratorIPcSsEppEv\00", section "llvm.metadata"		; <[43 x i8]*> [#uses=1]
@llvm.dbg.subprogram1143 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1141, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1141, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str1142, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 673, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1140 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1144 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1103 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1145 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1144 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1146 = internal constant [43 x i8] c"_ZN9__gnu_cxx17__normal_iteratorIPcSsEppEi\00", section "llvm.metadata"		; <[43 x i8]*> [#uses=1]
@llvm.dbg.subprogram1147 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1141, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1141, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str1146, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 680, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1145 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1148 = internal constant [11 x i8] c"operator--\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@.str1149 = internal constant [43 x i8] c"_ZN9__gnu_cxx17__normal_iteratorIPcSsEmmEv\00", section "llvm.metadata"		; <[43 x i8]*> [#uses=1]
@llvm.dbg.subprogram1150 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1148, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1148, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str1149, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 685, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1140 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1151 = internal constant [43 x i8] c"_ZN9__gnu_cxx17__normal_iteratorIPcSsEmmEi\00", section "llvm.metadata"		; <[43 x i8]*> [#uses=1]
@llvm.dbg.subprogram1152 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str1148, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1148, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str1151, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 692, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1145 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype1153 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1015 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1154 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype839 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1128 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1153 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1155 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1154 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1156 = internal constant [46 x i8] c"_ZNK9__gnu_cxx17__normal_iteratorIPcSsEixERKi\00", section "llvm.metadata"		; <[46 x i8]*> [#uses=1]
@llvm.dbg.subprogram1157 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([46 x i8]* @.str1156, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 697, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1155 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1158 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1138 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1103 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1153 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1159 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1158 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1160 = internal constant [45 x i8] c"_ZN9__gnu_cxx17__normal_iteratorIPcSsEpLERKi\00", section "llvm.metadata"		; <[45 x i8]*> [#uses=1]
@llvm.dbg.subprogram1161 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([45 x i8]* @.str1160, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 701, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1159 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1162 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1128 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1153 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1163 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1162 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1164 = internal constant [46 x i8] c"_ZNK9__gnu_cxx17__normal_iteratorIPcSsEplERKi\00", section "llvm.metadata"		; <[46 x i8]*> [#uses=1]
@llvm.dbg.subprogram1165 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str347, i32 0, i32 0), i8* getelementptr ([46 x i8]* @.str1164, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 705, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1163 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1166 = internal constant [45 x i8] c"_ZN9__gnu_cxx17__normal_iteratorIPcSsEmIERKi\00", section "llvm.metadata"		; <[45 x i8]*> [#uses=1]
@llvm.dbg.subprogram1167 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str95, i32 0, i32 0), i8* getelementptr ([45 x i8]* @.str1166, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 709, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1159 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1168 = internal constant [46 x i8] c"_ZNK9__gnu_cxx17__normal_iteratorIPcSsEmiERKi\00", section "llvm.metadata"		; <[46 x i8]*> [#uses=1]
@llvm.dbg.subprogram1169 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str344, i32 0, i32 0), i8* getelementptr ([46 x i8]* @.str1168, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 713, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1163 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1170 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1117 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1128 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1171 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1170 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1172 = internal constant [5 x i8] c"base\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str1173 = internal constant [47 x i8] c"_ZNK9__gnu_cxx17__normal_iteratorIPcSsE4baseEv\00", section "llvm.metadata"		; <[47 x i8]*> [#uses=1]
@llvm.dbg.subprogram1174 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1172, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1172, i32 0, i32 0), i8* getelementptr ([47 x i8]* @.str1173, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 717, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1171 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1175 = internal constant [16 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1102 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1107 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1120 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1126 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1132 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1137 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1143 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1147 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1150 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1152 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1157 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1161 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1165 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1167 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1169 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1174 to %0*)], section "llvm.metadata"		; <[16 x %0*]*> [#uses=1]
@llvm.dbg.composite1176 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([97 x i8]* @.str1099, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 637, i64 32, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([16 x %0*]* @llvm.dbg.array1175 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1177 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1178 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1177 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1179 = internal constant [19 x i8] c"_ZNKSs9_M_ibeginEv\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1180 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1172, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1172, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1179, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 293, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1178 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1181 = internal constant [8 x i8] c"_M_iend\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1182 = internal constant [17 x i8] c"_ZNKSs7_M_iendEv\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1183 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1181, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1181, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1182, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 297, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1178 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1184 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1185 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1184 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1186 = internal constant [8 x i8] c"_M_leak\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1187 = internal constant [16 x i8] c"_ZNSs7_M_leakEv\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1188 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1186, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1186, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1187, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 301, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1185 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1189 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1190 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1189 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1191 = internal constant [9 x i8] c"_M_check\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str1192 = internal constant [21 x i8] c"_ZNKSs8_M_checkEjPKc\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1193 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str1191, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str1191, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1192, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 308, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1190 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1194 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1195 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1194 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1196 = internal constant [16 x i8] c"_M_check_length\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str1197 = internal constant [30 x i8] c"_ZNKSs15_M_check_lengthEjjPKc\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram1198 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str1196, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1196, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1197, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 316, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1195 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1199 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1200 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1199 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1201 = internal constant [9 x i8] c"_M_limit\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str1202 = internal constant [19 x i8] c"_ZNKSs8_M_limitEjj\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1203 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str1201, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str1201, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1202, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 324, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1200 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1204 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype1032 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1205 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1204 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1206 = internal constant [12 x i8] c"_M_disjunct\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str1207 = internal constant [24 x i8] c"_ZNKSs11_M_disjunctEPKc\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram1208 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str1206, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str1206, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str1207, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 332, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1205 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1209 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1210 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1209 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1211 = internal constant [8 x i8] c"_M_copy\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1212 = internal constant [21 x i8] c"_ZNSs7_M_copyEPcPKcj\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1213 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1211, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1211, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1212, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 341, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1210 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1214 = internal constant [8 x i8] c"_M_move\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1215 = internal constant [21 x i8] c"_ZNSs7_M_moveEPcPKcj\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1216 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1214, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1214, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1215, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 350, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1210 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1217 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1218 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1217 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1219 = internal constant [10 x i8] c"_M_assign\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str1220 = internal constant [21 x i8] c"_ZNSs9_M_assignEPcjc\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1221 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str1219, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str1219, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1220, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 359, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1218 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1222 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* null, %0* null], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1223 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1222 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1224 = internal constant [14 x i8] c"_S_copy_chars\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram1225 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 371, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1223 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1226 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1227 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1226 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1228 = internal constant [64 x i8] c"_ZNSs13_S_copy_charsEPcN9__gnu_cxx17__normal_iteratorIS_SsEES2_\00", section "llvm.metadata"		; <[64 x i8]*> [#uses=1]
@llvm.dbg.subprogram1229 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([64 x i8]* @.str1228, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 378, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1227 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1230 = internal constant [103 x i8] c"__normal_iterator<const char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >\00", section "llvm.metadata"		; <[103 x i8]*> [#uses=1]
@llvm.dbg.composite1231 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([103 x i8]* @.str1230, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 637, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1232 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1231 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1231 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1233 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1232 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1234 = internal constant [65 x i8] c"_ZNSs13_S_copy_charsEPcN9__gnu_cxx17__normal_iteratorIPKcSsEES4_\00", section "llvm.metadata"		; <[65 x i8]*> [#uses=1]
@llvm.dbg.subprogram1235 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([65 x i8]* @.str1234, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 382, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1233 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1236 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1237 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1236 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1238 = internal constant [28 x i8] c"_ZNSs13_S_copy_charsEPcS_S_\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram1239 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str1238, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 386, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1237 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1240 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1241 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1240 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1242 = internal constant [30 x i8] c"_ZNSs13_S_copy_charsEPcPKcS1_\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram1243 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1224, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1242, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 390, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1241 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1244 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1245 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1244 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1246 = internal constant [10 x i8] c"_M_mutate\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str1247 = internal constant [20 x i8] c"_ZNSs9_M_mutateEjjj\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.subprogram1248 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str1246, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str1246, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1247, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 451, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1245 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1249 = internal constant [13 x i8] c"_M_leak_hard\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str1250 = internal constant [22 x i8] c"_ZNSs12_M_leak_hardEv\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram1251 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1249, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1249, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str1250, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 437, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1185 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1252 = internal constant [22 x i8] c"_ZNSs12_S_empty_repEv\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram1253 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1028, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1028, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str1252, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 400, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1027 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1254 = internal constant [13 x i8] c"basic_string\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram1255 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 2055, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1185 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1256 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1257 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1256 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1258 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 191, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1257 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype1259 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype916 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1260 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1261 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1260 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1262 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 183, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1261 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1263 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1264 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1263 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1265 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 197, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1264 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1266 = internal constant [6 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1267 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1266 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1268 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 208, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1267 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1269 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1270 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1269 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1271 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 219, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1270 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1272 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1273 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1272 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1274 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 226, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1273 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1275 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1276 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1275 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1277 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 233, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1276 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1278 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1279 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1278 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1280 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1254, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 477, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1279 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1281 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1282 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1281 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1283 = internal constant [14 x i8] c"~basic_string\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram1284 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1283, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1283, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 482, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1282 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype1285 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1693 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1286 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1287 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1286 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1288 = internal constant [13 x i8] c"_ZNSsaSERKSs\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram1289 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1288, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 490, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1287 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1290 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1291 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1290 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1292 = internal constant [12 x i8] c"_ZNSsaSEPKc\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram1293 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str1292, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 498, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1291 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1294 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1295 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1294 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1296 = internal constant [10 x i8] c"_ZNSsaSEc\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.subprogram1297 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str89, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str1296, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 509, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1295 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1298 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1299 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1298 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1300 = internal constant [6 x i8] c"begin\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1301 = internal constant [14 x i8] c"_ZNSs5beginEv\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram1302 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1300, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1300, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1301, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 521, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1299 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1303 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1231 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1304 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1303 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1305 = internal constant [15 x i8] c"_ZNKSs5beginEv\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1306 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1300, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1300, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1305, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 532, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1304 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1307 = internal constant [4 x i8] c"end\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1308 = internal constant [12 x i8] c"_ZNSs3endEv\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram1309 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str1307, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str1307, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str1308, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 540, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1299 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1310 = internal constant [13 x i8] c"_ZNKSs3endEv\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram1311 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str1307, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str1307, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1310, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 551, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1304 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1312 = internal constant [128 x i8] c"reverse_iterator<__gnu_cxx::__normal_iterator<char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >\00", section "llvm.metadata"		; <[128 x i8]*> [#uses=1]
@llvm.dbg.composite1313 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([128 x i8]* @.str1312, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 100, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1314 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1313 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1315 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1314 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1316 = internal constant [7 x i8] c"rbegin\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1317 = internal constant [15 x i8] c"_ZNSs6rbeginEv\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1318 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1316, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1316, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1317, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 560, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1315 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1319 = internal constant [134 x i8] c"reverse_iterator<__gnu_cxx::__normal_iterator<const char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >\00", section "llvm.metadata"		; <[134 x i8]*> [#uses=1]
@llvm.dbg.composite1320 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([134 x i8]* @.str1319, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1098 to %0*), i32 100, i64 0, i64 0, i64 0, i32 4, %0* null, %0* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1321 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1320 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1322 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1321 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1323 = internal constant [16 x i8] c"_ZNKSs6rbeginEv\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1324 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1316, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1316, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1323, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 569, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1322 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1325 = internal constant [5 x i8] c"rend\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str1326 = internal constant [13 x i8] c"_ZNSs4rendEv\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram1327 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1325, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1325, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1326, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 578, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1315 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1328 = internal constant [14 x i8] c"_ZNKSs4rendEv\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram1329 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1325, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1325, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1328, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 587, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1322 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1330 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1331 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1330 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1332 = internal constant [5 x i8] c"size\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str1333 = internal constant [14 x i8] c"_ZNKSs4sizeEv\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram1334 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1332, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1332, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1333, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 595, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1331 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1335 = internal constant [7 x i8] c"length\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1336 = internal constant [16 x i8] c"_ZNKSs6lengthEv\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1337 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1335, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1335, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1336, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 601, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1331 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1338 = internal constant [18 x i8] c"_ZNKSs8max_sizeEv\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1339 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str863, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str863, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1338, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 606, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1331 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1340 = internal constant [4 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1341 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1340 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1342 = internal constant [7 x i8] c"resize\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1343 = internal constant [16 x i8] c"_ZNSs6resizeEjc\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1344 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1342, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1342, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1343, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 622, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1341 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1345 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1346 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1345 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1347 = internal constant [15 x i8] c"_ZNSs6resizeEj\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1348 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1342, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1342, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1347, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 633, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1346 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1349 = internal constant [9 x i8] c"capacity\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@.str1350 = internal constant [18 x i8] c"_ZNKSs8capacityEv\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1351 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str1349, i32 0, i32 0), i8* getelementptr ([9 x i8]* @.str1349, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1350, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 641, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1331 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite1352 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1345 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1353 = internal constant [8 x i8] c"reserve\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1354 = internal constant [16 x i8] c"_ZNSs7reserveEj\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1355 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1353, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1353, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1354, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 484, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1352 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1356 = internal constant [6 x i8] c"clear\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1357 = internal constant [14 x i8] c"_ZNSs5clearEv\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram1358 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1356, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1356, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1357, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 668, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1185 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1359 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype1032 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1360 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1359 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1361 = internal constant [6 x i8] c"empty\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1362 = internal constant [15 x i8] c"_ZNKSs5emptyEv\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1363 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1361, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1361, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1362, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 675, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1360 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1364 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype845 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1365 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1364 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1366 = internal constant [11 x i8] c"_ZNKSsixEj\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram1367 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1366, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 690, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1365 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1368 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype839 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1369 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1368 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1370 = internal constant [10 x i8] c"_ZNSsixEj\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.subprogram1371 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str334, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str1370, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 707, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1369 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1372 = internal constant [3 x i8] c"at\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@.str1373 = internal constant [12 x i8] c"_ZNKSs2atEj\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram1374 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([3 x i8]* @.str1372, i32 0, i32 0), i8* getelementptr ([3 x i8]* @.str1372, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str1373, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 728, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1365 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1375 = internal constant [11 x i8] c"_ZNSs2atEj\00", section "llvm.metadata"		; <[11 x i8]*> [#uses=1]
@llvm.dbg.subprogram1376 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([3 x i8]* @.str1372, i32 0, i32 0), i8* getelementptr ([3 x i8]* @.str1372, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str1375, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 747, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1369 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1377 = internal constant [13 x i8] c"_ZNSspLERKSs\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram1378 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1377, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 762, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1287 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1379 = internal constant [12 x i8] c"_ZNSspLEPKc\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.subprogram1380 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str1379, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 771, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1291 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1381 = internal constant [10 x i8] c"_ZNSspLEc\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@llvm.dbg.subprogram1382 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str1381, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 780, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1295 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1383 = internal constant [7 x i8] c"append\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1384 = internal constant [18 x i8] c"_ZNSs6appendERKSs\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1385 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1384, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 330, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1287 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1386 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1387 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1386 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1388 = internal constant [20 x i8] c"_ZNSs6appendERKSsjj\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.subprogram1389 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1388, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 347, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1387 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1390 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1391 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1390 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1392 = internal constant [18 x i8] c"_ZNSs6appendEPKcj\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1393 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1392, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 303, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1391 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1394 = internal constant [17 x i8] c"_ZNSs6appendEPKc\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1395 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1394, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 824, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1291 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1396 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1397 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1396 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1398 = internal constant [16 x i8] c"_ZNSs6appendEjc\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1399 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1398, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 286, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1397 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1400 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* null, %0* null], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1401 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1400 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1402 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1383, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 851, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1401 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1403 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1404 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1403 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1405 = internal constant [10 x i8] c"push_back\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str1406 = internal constant [18 x i8] c"_ZNSs9push_backEc\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1407 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str1405, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str1405, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1406, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 859, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1404 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1408 = internal constant [7 x i8] c"assign\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1409 = internal constant [18 x i8] c"_ZNSs6assignERKSs\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1410 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1409, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 248, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1287 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1411 = internal constant [20 x i8] c"_ZNSs6assignERKSsjj\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.subprogram1412 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1411, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 889, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1387 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1413 = internal constant [18 x i8] c"_ZNSs6assignEPKcj\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1414 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1413, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 264, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1391 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1415 = internal constant [17 x i8] c"_ZNSs6assignEPKc\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1416 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1415, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 917, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1291 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1417 = internal constant [16 x i8] c"_ZNSs6assignEjc\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1418 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1417, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 933, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1397 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite1419 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1400 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1420 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1408, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 946, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1419 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1421 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1422 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1421 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1423 = internal constant [7 x i8] c"insert\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1424 = internal constant [53 x i8] c"_ZNSs6insertEN9__gnu_cxx17__normal_iteratorIPcSsEEjc\00", section "llvm.metadata"		; <[53 x i8]*> [#uses=1]
@llvm.dbg.subprogram1425 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([53 x i8]* @.str1424, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 962, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1422 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1426 = internal constant [5 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* null, %0* null], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1427 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1426 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1428 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 978, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1427 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1429 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1430 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1429 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1431 = internal constant [19 x i8] c"_ZNSs6insertEjRKSs\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1432 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1431, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 993, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1430 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1433 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1434 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1433 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1435 = internal constant [21 x i8] c"_ZNSs6insertEjRKSsjj\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1436 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1435, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1016, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1434 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1437 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1438 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1437 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1439 = internal constant [19 x i8] c"_ZNSs6insertEjPKcj\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1440 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1439, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 365, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1438 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1441 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1442 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1441 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1443 = internal constant [18 x i8] c"_ZNSs6insertEjPKc\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1444 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1443, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1056, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1442 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1445 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1446 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1445 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1447 = internal constant [17 x i8] c"_ZNSs6insertEjjc\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1448 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1447, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1079, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1446 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1449 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1450 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1449 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1451 = internal constant [52 x i8] c"_ZNSs6insertEN9__gnu_cxx17__normal_iteratorIPcSsEEc\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.subprogram1452 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1423, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str1451, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1096, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1450 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1453 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1454 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1453 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1455 = internal constant [6 x i8] c"erase\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1456 = internal constant [15 x i8] c"_ZNSs5eraseEjj\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1457 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1455, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1455, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1456, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1120, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1454 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1458 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1459 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1458 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1460 = internal constant [50 x i8] c"_ZNSs5eraseEN9__gnu_cxx17__normal_iteratorIPcSsEE\00", section "llvm.metadata"		; <[50 x i8]*> [#uses=1]
@llvm.dbg.subprogram1461 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1455, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1455, i32 0, i32 0), i8* getelementptr ([50 x i8]* @.str1460, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1136, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1459 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1462 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1463 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1462 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1464 = internal constant [53 x i8] c"_ZNSs5eraseEN9__gnu_cxx17__normal_iteratorIPcSsEES2_\00", section "llvm.metadata"		; <[53 x i8]*> [#uses=1]
@llvm.dbg.subprogram1465 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1455, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1455, i32 0, i32 0), i8* getelementptr ([53 x i8]* @.str1464, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1156, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1463 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1466 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1467 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1466 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1468 = internal constant [8 x i8] c"replace\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1469 = internal constant [21 x i8] c"_ZNSs7replaceEjjRKSs\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1470 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1469, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1183, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1467 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1471 = internal constant [7 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[7 x %0*]*> [#uses=1]
@llvm.dbg.composite1472 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([7 x %0*]* @llvm.dbg.array1471 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1473 = internal constant [23 x i8] c"_ZNSs7replaceEjjRKSsjj\00", section "llvm.metadata"		; <[23 x i8]*> [#uses=1]
@llvm.dbg.subprogram1474 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([23 x i8]* @.str1473, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1206, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1472 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1475 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1476 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1475 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1477 = internal constant [21 x i8] c"_ZNSs7replaceEjjPKcj\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1478 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1477, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 397, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1476 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1479 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1480 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1479 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1481 = internal constant [20 x i8] c"_ZNSs7replaceEjjPKc\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.subprogram1482 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1481, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1248, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1480 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1483 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1484 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1483 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1485 = internal constant [19 x i8] c"_ZNSs7replaceEjjjc\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1486 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1485, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1271, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1484 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1487 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1488 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1487 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1489 = internal constant [59 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_RKSs\00", section "llvm.metadata"		; <[59 x i8]*> [#uses=1]
@llvm.dbg.subprogram1490 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([59 x i8]* @.str1489, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1289, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1488 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1491 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1492 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1491 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1493 = internal constant [59 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_PKcj\00", section "llvm.metadata"		; <[59 x i8]*> [#uses=1]
@llvm.dbg.subprogram1494 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([59 x i8]* @.str1493, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1307, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1492 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1495 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1496 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1495 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1497 = internal constant [58 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_PKc\00", section "llvm.metadata"		; <[58 x i8]*> [#uses=1]
@llvm.dbg.subprogram1498 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([58 x i8]* @.str1497, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1328, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1496 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1499 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1500 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1499 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1501 = internal constant [57 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_jc\00", section "llvm.metadata"		; <[57 x i8]*> [#uses=1]
@llvm.dbg.subprogram1502 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([57 x i8]* @.str1501, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1349, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1500 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1503 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* null, %0* null], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1504 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1503 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1505 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1373, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1504 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1506 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1507 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1506 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1508 = internal constant [61 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_S1_S1_\00", section "llvm.metadata"		; <[61 x i8]*> [#uses=1]
@llvm.dbg.subprogram1509 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1508, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1385, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1507 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1510 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1511 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1510 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1512 = internal constant [61 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_PKcS4_\00", section "llvm.metadata"		; <[61 x i8]*> [#uses=1]
@llvm.dbg.subprogram1513 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1512, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1396, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1511 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1514 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1515 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1514 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1516 = internal constant [61 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_S2_S2_\00", section "llvm.metadata"		; <[61 x i8]*> [#uses=1]
@llvm.dbg.subprogram1517 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([61 x i8]* @.str1516, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1406, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1515 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1518 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1231 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1231 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1519 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1518 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1520 = internal constant [70 x i8] c"_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_NS0_IPKcSsEES5_\00", section "llvm.metadata"		; <[70 x i8]*> [#uses=1]
@llvm.dbg.subprogram1521 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1468, i32 0, i32 0), i8* getelementptr ([70 x i8]* @.str1520, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1417, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1519 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1522 = internal constant [18 x i8] c"cpp_type_traits.h\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1523 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([18 x i8]* @.str1522, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1524 = internal constant [12 x i8] c"__true_type\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@llvm.dbg.composite1526 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str1524, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1523 to %0*), i32 97, i64 8, i64 8, i64 0, i32 0, %0* null, %0* bitcast ([0 x %0*]* @llvm.dbg.array680 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1527 = internal constant [7 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1526 to %0*)], section "llvm.metadata"		; <[7 x %0*]*> [#uses=1]
@llvm.dbg.composite1528 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([7 x %0*]* @llvm.dbg.array1527 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1529 = internal constant [20 x i8] c"_M_replace_dispatch\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.subprogram1530 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([20 x i8]* @.str1529, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1529, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1430, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1528 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1531 = internal constant [13 x i8] c"__false_type\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.composite1533 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1531, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1523 to %0*), i32 98, i64 8, i64 8, i64 0, i32 0, %0* null, %0* bitcast ([0 x %0*]* @llvm.dbg.array680 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1534 = internal constant [7 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1176 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1533 to %0*)], section "llvm.metadata"		; <[7 x %0*]*> [#uses=1]
@llvm.dbg.composite1535 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([7 x %0*]* @llvm.dbg.array1534 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1536 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([20 x i8]* @.str1529, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1529, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1436, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1535 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1537 = internal constant [15 x i8] c"_M_replace_aux\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@.str1538 = internal constant [27 x i8] c"_ZNSs14_M_replace_auxEjjjc\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram1539 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str1537, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1537, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str1538, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 651, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1484 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1540 = internal constant [16 x i8] c"_M_replace_safe\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str1541 = internal constant [30 x i8] c"_ZNSs15_M_replace_safeEjjPKcj\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram1542 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str1540, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1540, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1541, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 664, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1476 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1543 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1533 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1544 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1543 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1545 = internal constant [17 x i8] c"_S_construct_aux\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1546 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str1545, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1545, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1451, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1544 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1547 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1526 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1548 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1547 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1549 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str1545, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1545, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1460, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1548 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1550 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1551 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1550 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1552 = internal constant [13 x i8] c"_S_construct\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram1553 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1466, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1551 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1554 = internal constant [26 x i8] c"stl_iterator_base_types.h\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.compile_unit1555 = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 4, i8* getelementptr ([26 x i8]* @.str1554, i32 0, i32 0), i8* getelementptr ([115 x i8]* @.str638, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 false, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str1556 = internal constant [19 x i8] c"input_iterator_tag\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.composite1558 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([19 x i8]* @.str1556, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1555 to %0*), i32 80, i64 8, i64 8, i64 0, i32 0, %0* null, %0* bitcast ([0 x %0*]* @llvm.dbg.array680 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1559 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1558 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1560 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1559 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1561 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1476, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1560 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1562 = internal constant [21 x i8] c"forward_iterator_tag\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1564 = internal constant %llvm.dbg.derivedtype.type { i32 458780, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1555 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1558 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1565 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1564 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite1566 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([21 x i8]* @.str1562, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1555 to %0*), i32 84, i64 8, i64 8, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array1565 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.array1567 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* null, %0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*), %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1566 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1568 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1567 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1569 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1483, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1568 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1570 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype887 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1571 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1570 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1572 = internal constant [30 x i8] c"_ZNSs12_S_constructEjcRKSaIcE\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram1573 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1552, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1572, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 166, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1571 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1574 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1575 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1574 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1576 = internal constant [5 x i8] c"copy\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str1577 = internal constant [17 x i8] c"_ZNKSs4copyEPcjj\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1578 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1576, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1576, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1577, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 705, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1575 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1579 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype923 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1285 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1580 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1579 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1581 = internal constant [5 x i8] c"swap\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str1582 = internal constant [15 x i8] c"_ZNSs4swapERSs\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1583 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1581, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1581, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1582, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 501, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1580 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1584 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1585 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1584 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1586 = internal constant [6 x i8] c"c_str\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1587 = internal constant [15 x i8] c"_ZNKSs5c_strEv\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1588 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1586, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1586, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1587, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1522, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1585 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1589 = internal constant [5 x i8] c"data\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str1590 = internal constant [14 x i8] c"_ZNKSs4dataEv\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram1591 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1589, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1589, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1590, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1532, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1585 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1592 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite902 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1593 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1592 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1594 = internal constant [14 x i8] c"get_allocator\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str1595 = internal constant [24 x i8] c"_ZNKSs13get_allocatorEv\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram1596 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1594, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1594, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str1595, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1539, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1593 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1597 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1598 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1597 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1599 = internal constant [5 x i8] c"find\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@.str1600 = internal constant [18 x i8] c"_ZNKSs4findEPKcjj\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1601 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1600, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 719, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1598 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1602 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1603 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1602 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1604 = internal constant [18 x i8] c"_ZNKSs4findERKSsj\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1605 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1604, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1567, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1603 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1606 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1607 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1606 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1608 = internal constant [17 x i8] c"_ZNKSs4findEPKcj\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1609 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1608, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1581, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1607 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1610 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype282 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype672 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1611 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1610 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1612 = internal constant [15 x i8] c"_ZNKSs4findEcj\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1613 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1599, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1612, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 742, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1611 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite1614 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1602 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1615 = internal constant [6 x i8] c"rfind\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str1616 = internal constant [19 x i8] c"_ZNKSs5rfindERKSsj\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1617 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1616, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1611, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1614 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1618 = internal constant [19 x i8] c"_ZNKSs5rfindEPKcjj\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1619 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1618, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 760, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1598 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite1620 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1606 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1621 = internal constant [18 x i8] c"_ZNKSs5rfindEPKcj\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.subprogram1622 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1621, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1639, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1620 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.composite1623 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1610 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1624 = internal constant [16 x i8] c"_ZNKSs5rfindEcj\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@llvm.dbg.subprogram1625 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1615, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1624, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 781, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1623 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1626 = internal constant [14 x i8] c"find_first_of\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str1627 = internal constant [28 x i8] c"_ZNKSs13find_first_ofERKSsj\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram1628 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str1627, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1669, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1603 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1629 = internal constant [28 x i8] c"_ZNKSs13find_first_ofEPKcjj\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram1630 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str1629, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 798, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1598 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1631 = internal constant [27 x i8] c"_ZNKSs13find_first_ofEPKcj\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram1632 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str1631, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1697, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1607 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1633 = internal constant [25 x i8] c"_ZNKSs13find_first_ofEcj\00", section "llvm.metadata"		; <[25 x i8]*> [#uses=1]
@llvm.dbg.subprogram1634 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1626, i32 0, i32 0), i8* getelementptr ([25 x i8]* @.str1633, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1716, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1611 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1635 = internal constant [13 x i8] c"find_last_of\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@.str1636 = internal constant [27 x i8] c"_ZNKSs12find_last_ofERKSsj\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram1637 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str1636, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1730, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1614 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1638 = internal constant [27 x i8] c"_ZNKSs12find_last_ofEPKcjj\00", section "llvm.metadata"		; <[27 x i8]*> [#uses=1]
@llvm.dbg.subprogram1639 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([27 x i8]* @.str1638, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 813, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1598 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1640 = internal constant [26 x i8] c"_ZNKSs12find_last_ofEPKcj\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram1641 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1640, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1758, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1620 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1642 = internal constant [24 x i8] c"_ZNKSs12find_last_ofEcj\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram1643 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str1635, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str1642, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1777, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1623 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1644 = internal constant [18 x i8] c"find_first_not_of\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@.str1645 = internal constant [32 x i8] c"_ZNKSs17find_first_not_ofERKSsj\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram1646 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str1645, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1791, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1603 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1647 = internal constant [32 x i8] c"_ZNKSs17find_first_not_ofEPKcjj\00", section "llvm.metadata"		; <[32 x i8]*> [#uses=1]
@llvm.dbg.subprogram1648 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([32 x i8]* @.str1647, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 834, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1598 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1649 = internal constant [31 x i8] c"_ZNKSs17find_first_not_ofEPKcj\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram1650 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str1649, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1820, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1607 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1651 = internal constant [29 x i8] c"_ZNKSs17find_first_not_ofEcj\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram1652 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([18 x i8]* @.str1644, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str1651, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 846, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1611 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1653 = internal constant [17 x i8] c"find_last_not_of\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@.str1654 = internal constant [31 x i8] c"_ZNKSs16find_last_not_ofERKSsj\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram1655 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str1654, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1850, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1614 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1656 = internal constant [31 x i8] c"_ZNKSs16find_last_not_ofEPKcjj\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram1657 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str1656, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 857, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1598 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1658 = internal constant [30 x i8] c"_ZNKSs16find_last_not_ofEPKcj\00", section "llvm.metadata"		; <[30 x i8]*> [#uses=1]
@llvm.dbg.subprogram1659 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([30 x i8]* @.str1658, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1879, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1620 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1660 = internal constant [28 x i8] c"_ZNKSs16find_last_not_ofEcj\00", section "llvm.metadata"		; <[28 x i8]*> [#uses=1]
@llvm.dbg.subprogram1661 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1653, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str1660, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 878, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1623 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1662 = internal constant [4 x %0*] [%0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1693 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[4 x %0*]*> [#uses=1]
@llvm.dbg.composite1663 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([4 x %0*]* @llvm.dbg.array1662 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1664 = internal constant [7 x i8] c"substr\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@.str1665 = internal constant [17 x i8] c"_ZNKSs6substrEjj\00", section "llvm.metadata"		; <[17 x i8]*> [#uses=1]
@llvm.dbg.subprogram1666 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1664, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str1664, i32 0, i32 0), i8* getelementptr ([17 x i8]* @.str1665, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1911, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1663 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1667 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1668 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1667 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1669 = internal constant [8 x i8] c"compare\00", section "llvm.metadata"		; <[8 x i8]*> [#uses=1]
@.str1670 = internal constant [20 x i8] c"_ZNKSs7compareERKSs\00", section "llvm.metadata"		; <[20 x i8]*> [#uses=1]
@llvm.dbg.subprogram1671 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([20 x i8]* @.str1670, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit807 to %0*), i32 1929, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1668 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1672 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1673 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1672 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1674 = internal constant [22 x i8] c"_ZNKSs7compareEjjRKSs\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram1675 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str1674, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 898, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1673 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1676 = internal constant [7 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1259 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[7 x %0*]*> [#uses=1]
@llvm.dbg.composite1677 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([7 x %0*]* @llvm.dbg.array1676 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1678 = internal constant [24 x i8] c"_ZNKSs7compareEjjRKSsjj\00", section "llvm.metadata"		; <[24 x i8]*> [#uses=1]
@llvm.dbg.subprogram1679 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([24 x i8]* @.str1678, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 914, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1677 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1680 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1681 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1680 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1682 = internal constant [19 x i8] c"_ZNKSs7compareEPKc\00", section "llvm.metadata"		; <[19 x i8]*> [#uses=1]
@llvm.dbg.subprogram1683 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([19 x i8]* @.str1682, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 931, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1681 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1684 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1685 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1684 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1686 = internal constant [21 x i8] c"_ZNKSs7compareEjjPKc\00", section "llvm.metadata"		; <[21 x i8]*> [#uses=1]
@llvm.dbg.subprogram1687 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([21 x i8]* @.str1686, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 946, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1685 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1688 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype917 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype280 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite1689 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array1688 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1690 = internal constant [22 x i8] c"_ZNKSs7compareEjjPKcj\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram1691 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([8 x i8]* @.str1669, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str1690, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1070 to %0*), i32 963, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1689 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1692 = internal constant [143 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype913 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram922 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram927 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1096 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1180 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1183 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1188 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1193 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1198 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1203 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1208 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1213 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1216 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1221 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1225 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1229 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1235 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1239 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1243 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1248 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1251 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1253 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1255 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1258 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1262 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1265 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1268 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1271 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1274 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1277 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1280 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1284 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1289 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1293 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1297 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1302 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1306 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1309 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1311 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1318 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1324 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1327 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1329 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1334 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1337 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1339 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1344 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1348 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1351 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1355 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1358 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1363 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1367 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1371 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1374 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1376 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1378 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1380 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1382 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1385 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1389 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1393 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1395 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1399 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1402 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1407 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1410 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1412 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1414 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1416 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1418 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1420 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1425 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1428 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1432 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1436 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1440 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1444 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1448 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1452 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1457 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1461 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1465 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1470 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1474 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1478 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1482 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1486 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1490 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1494 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1498 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1502 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1505 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1509 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1513 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1517 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1521 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1530 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1536 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1539 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1542 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1546 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1549 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1553 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1561 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1569 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1573 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1578 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1583 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1588 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1591 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1596 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1601 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1605 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1609 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1613 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1617 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1619 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1622 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1625 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1628 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1630 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1632 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1634 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1637 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1639 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1641 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1643 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1646 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1648 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1650 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1652 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1655 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1657 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1659 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1661 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1666 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1671 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1675 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1679 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1683 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1687 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1691 to %0*)], section "llvm.metadata"		; <[143 x %0*]*> [#uses=1]
@llvm.dbg.composite1693 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([64 x i8]* @.str804, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit803 to %0*), i32 56, i64 32, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([143 x %0*]* @llvm.dbg.array1692 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype1694 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str914, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit803 to %0*), i32 56, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1693 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str1695 = internal constant [7 x i8] c"_M_msg\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype1696 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str1695, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 109, i64 32, i64 32, i64 32, i32 1, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1694 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1697 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1714 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1698 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str914, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit803 to %0*), i32 56, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype916 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1699 = internal constant %llvm.dbg.derivedtype.type { i32 458768, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1698 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1700 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1697 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1699 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1701 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1700 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1702 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str775, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str775, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 114, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1701 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1703 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1697 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1704 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1703 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1705 = internal constant [15 x i8] c"~runtime_error\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@llvm.dbg.subprogram1706 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str1705, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str1705, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 117, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1704 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.derivedtype1707 = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1714 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1708 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1707 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1709 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype791 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1708 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1710 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1709 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1711 = internal constant [29 x i8] c"_ZNKSt13runtime_error4whatEv\00", section "llvm.metadata"		; <[29 x i8]*> [#uses=1]
@llvm.dbg.subprogram1712 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str796, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str796, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str1711, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 122, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1710 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1713 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype801 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1696 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1702 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1706 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1712 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite1714 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str775, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 108, i64 64, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array1713 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype1715 = internal constant %llvm.dbg.derivedtype.type { i32 458780, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1714 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype1716 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1721 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1717 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1716 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1699 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1718 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1717 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprogram1719 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str773, i32 0, i32 0), i8* getelementptr ([15 x i8]* @.str773, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 136, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1718 to %0*), i1 false, i1 false }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1720 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1715 to %0*), %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1719 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1721 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([15 x i8]* @.str773, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 134, i64 64, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1720 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype1722 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1721 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1723 = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1722 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1724 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1723 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1725 = internal constant [16 x i8] c"~overflow_error\00", section "llvm.metadata"		; <[16 x i8]*> [#uses=1]
@.str1726 = internal constant [26 x i8] c"_ZNSt14overflow_errorD1Ev\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram1727 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str1725, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1725, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1726, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*), i32 134, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1724 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@_ZTVSt14overflow_error = weak constant [5 x i32 (...)*] [i32 (...)* null, i32 (...)* bitcast (%struct.__si_class_type_info_pseudo* @_ZTISt14overflow_error to i32 (...)*), i32 (...)* bitcast (void (%"struct.std::overflow_error"*)* @_ZNSt14overflow_errorD1Ev to i32 (...)*), i32 (...)* bitcast (void (%"struct.std::overflow_error"*)* @_ZNSt14overflow_errorD0Ev to i32 (...)*), i32 (...)* bitcast (i8* (%"struct.std::runtime_error"*)* @_ZNKSt13runtime_error4whatEv to i32 (...)*)], align 8		; <[5 x i32 (...)*]*> [#uses=1]
@.str1735 = internal constant [26 x i8] c"_ZNSt14overflow_errorD0Ev\00", section "llvm.metadata"		; <[26 x i8]*> [#uses=1]
@llvm.dbg.subprogram1736 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([16 x i8]* @.str1725, i32 0, i32 0), i8* getelementptr ([16 x i8]* @.str1725, i32 0, i32 0), i8* getelementptr ([26 x i8]* @.str1735, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 702, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1724 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array1738 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite1739 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array1738 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1740 = internal constant [14 x i8] c"__complex_exp\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@.str1741 = internal constant [22 x i8] c"_ZSt13__complex_expCd\00", section "llvm.metadata"		; <[22 x i8]*> [#uses=1]
@llvm.dbg.subprogram1742 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str1740, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str1740, i32 0, i32 0), i8* getelementptr ([22 x i8]* @.str1741, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 730, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1739 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1744 = internal constant [12 x i8] c"exp<double>\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str1745 = internal constant [31 x i8] c"_ZSt3expIdESt7complexIT_ERKS2_\00", section "llvm.metadata"		; <[31 x i8]*> [#uses=1]
@llvm.dbg.subprogram1746 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str1744, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str1744, i32 0, i32 0), i8* getelementptr ([31 x i8]* @.str1745, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*), i32 738, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite573 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.subprogram1748 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str457, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str457, i32 0, i32 0), i8* getelementptr ([29 x i8]* @.str495, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 497, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite488 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable1751 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1748 to %0*), i8* getelementptr ([7 x i8]* @.str259, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 503, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1752 = internal constant [2 x i8] c"u\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable1753 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1748 to %0*), i8* getelementptr ([2 x i8]* @.str1752, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 501, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1754 = internal constant [2 x i8] c"t\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable1755 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1748 to %0*), i8* getelementptr ([2 x i8]* @.str1754, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 501, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1756 = internal constant [2 x i8] c"w\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable1757 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1748 to %0*), i8* getelementptr ([2 x i8]* @.str1756, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 501, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1758 = internal constant [3 x i8] c"wm\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.variable1759 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1748 to %0*), i8* getelementptr ([3 x i8]* @.str1758, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 501, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram1771 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @.str460, i32 0, i32 0), i8* getelementptr ([12 x i8]* @.str460, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str497, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 535, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite492 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable1774 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1771 to %0*), i8* getelementptr ([7 x i8]* @.str259, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 541, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable1775 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1771 to %0*), i8* getelementptr ([2 x i8]* @.str1752, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 539, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable1776 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1771 to %0*), i8* getelementptr ([2 x i8]* @.str1754, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 539, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable1777 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1771 to %0*), i8* getelementptr ([2 x i8]* @.str1756, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 539, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.variable1778 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1771 to %0*), i8* getelementptr ([3 x i8]* @.str1758, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 539, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite223 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.subprogram1785 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([10 x i8]* @.str463, i32 0, i32 0), i8* getelementptr ([28 x i8]* @.str499, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 576, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite351 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.variable1791 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1785 to %0*), i8* getelementptr ([7 x i8]* @.str259, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 614, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1794 = internal constant [5 x i8] c"dft2\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable1795 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1785 to %0*), i8* getelementptr ([5 x i8]* @.str1794, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 601, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1796 = internal constant [5 x i8] c"dft1\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.variable1797 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1785 to %0*), i8* getelementptr ([5 x i8]* @.str1796, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 600, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite486 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1798 = internal constant [3 x i8] c"a2\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.variable1799 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1785 to %0*), i8* getelementptr ([3 x i8]* @.str1798, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 591, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1800 = internal constant [3 x i8] c"a1\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.variable1801 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1785 to %0*), i8* getelementptr ([3 x i8]* @.str1800, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 590, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@llvm.dbg.derivedtype1802 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype837 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array1803 = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype269 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype1802 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite1804 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array1803 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str1805 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram1806 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str1805, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str1805, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 654, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite1804 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str1816 = internal constant [6 x i8] c"poly3\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.variable1817 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1806 to %0*), i8* getelementptr ([6 x i8]* @.str1816, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 674, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1818 = internal constant [6 x i8] c"poly2\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.variable1819 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1806 to %0*), i8* getelementptr ([6 x i8]* @.str1818, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 673, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1820 = internal constant [6 x i8] c"poly1\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.variable1821 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1806 to %0*), i8* getelementptr ([6 x i8]* @.str1820, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 672, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite518 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
@.str1824 = internal constant [4 x i8] c"-ga\00", align 4		; <[4 x i8]*> [#uses=0]
@_ZSt4cout = external global %"struct.std::basic_ostream<char,std::char_traits<char> >"		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=3]
@.str1825 = internal constant [32 x i8] c"\0Afftbench (Std. C++) run time: \00"		; <[32 x i8]*> [#uses=1]
@.str1826 = internal constant [3 x i8] c"\0A\0A\00"		; <[3 x i8]*> [#uses=1]
@llvm.global_ctors = appending global [1 x %2] [%2 { i32 65535, void ()* @_GLOBAL__I_main }]		; <[1 x %2]*> [#uses=0]

@_ZL20__gthrw_pthread_oncePiPFvvE = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._50*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._50*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = alias weak i32 (%struct..0._50*)* @pthread_mutexattr_init		; <i32 (%struct..0._50*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = alias weak i32 (%struct..0._50*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._50*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = alias weak i32 (%struct..0._50*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._50*)*> [#uses=0]

define i32 @main(i32 %argc, i8** nocapture %argv) {
entry:
	%n.0.reg2mem = alloca i32		; <i32*> [#uses=5]
	%poly3 = alloca %"struct.polynomial<double>", align 8		; <%"struct.polynomial<double>"*> [#uses=5]
	%poly2 = alloca %"struct.polynomial<double>", align 8		; <%"struct.polynomial<double>"*> [#uses=6]
	%poly1 = alloca %"struct.polynomial<double>", align 8		; <%"struct.polynomial<double>"*> [#uses=6]
	%0 = alloca %"struct.polynomial<double>", align 8		; <%"struct.polynomial<double>"*> [#uses=4]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1806 to %0*))
	%1 = bitcast %"struct.polynomial<double>"* %poly3 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %1, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1817 to %0*))
	%2 = bitcast %"struct.polynomial<double>"* %poly2 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %2, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1819 to %0*))
	%3 = bitcast %"struct.polynomial<double>"* %poly1 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %3, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1821 to %0*))
	call void @llvm.dbg.stoppoint(i32 659, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%4 = icmp sgt i32 %argc, 1		; <i1> [#uses=1]
	br i1 %4, label %bb4, label %bb5

bb1:		; preds = %bb4
	call void @llvm.dbg.stoppoint(i32 663, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%5 = getelementptr i8** %argv, i32 1		; <i8**> [#uses=1]
	%6 = load i8** %5, align 1		; <i8*> [#uses=1]
	%tmp = bitcast i8* %6 to i32*		; <i32*> [#uses=1]
	%lhsv = load i32* %tmp, align 1		; <i32> [#uses=1]
	%7 = icmp eq i32 %lhsv, 6383405		; <i1> [#uses=1]
	br i1 %7, label %bb5, label %bb3

bb3:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 661, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%8 = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %bb4

bb4:		; preds = %bb3, %entry
	%i.0 = phi i32 [ %8, %bb3 ], [ 1, %entry ]		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 661, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%9 = icmp slt i32 %i.0, %argc		; <i1> [#uses=1]
	br i1 %9, label %bb1, label %bb5

bb5:		; preds = %bb4, %bb1, %entry
	%ga_testing.0 = phi i8 [ 0, %entry ], [ 0, %bb4 ], [ 1, %bb1 ]		; <i8> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 672, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdEC1Ej(%"struct.polynomial<double>"* %poly1, i32 524288)
	call void @llvm.dbg.stoppoint(i32 673, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdEC1Ej(%"struct.polynomial<double>"* %poly2, i32 524288)
			to label %invcont unwind label %lpad

invcont:		; preds = %bb5
	call void @llvm.dbg.stoppoint(i32 674, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdEC1Ej(%"struct.polynomial<double>"* %poly3, i32 1048575)
			to label %bb8.thread unwind label %lpad47

bb8.thread:		; preds = %invcont
	store i32 0, i32* %n.0.reg2mem
	call void @llvm.dbg.stoppoint(i32 676, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb7

bb7:		; preds = %bb8, %bb8.thread
	call void @llvm.dbg.stoppoint(i32 678, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%n.0.reload3 = load i32* %n.0.reg2mem		; <i32> [#uses=1]
	%10 = call double* @_ZN10polynomialIdEixEj(%"struct.polynomial<double>"* %poly1, i32 %n.0.reload3) nounwind		; <double*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 68, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%11 = load i32* @_ZZL13random_doublevE4seed, align 4		; <i32> [#uses=1]
	%12 = xor i32 %11, 123459876		; <i32> [#uses=3]
	store i32 %12, i32* @_ZZL13random_doublevE4seed, align 4
	call void @llvm.dbg.stoppoint(i32 69, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%13 = sdiv i32 %12, 127773		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 70, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%14 = mul i32 %13, 127773		; <i32> [#uses=1]
	%15 = sub i32 %12, %14		; <i32> [#uses=1]
	%16 = mul i32 %15, 16807		; <i32> [#uses=1]
	%17 = mul i32 %13, 2836		; <i32> [#uses=1]
	%18 = sub i32 %16, %17		; <i32> [#uses=2]
	store i32 %18, i32* @_ZZL13random_doublevE4seed, align 4
	call void @llvm.dbg.stoppoint(i32 72, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%19 = icmp slt i32 %18, 0		; <i1> [#uses=1]
	br i1 %19, label %bb.i, label %_ZL13random_doublev.exit

bb.i:		; preds = %bb7
	call void @llvm.dbg.stoppoint(i32 73, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%20 = load i32* @_ZZL13random_doublevE4seed, align 4		; <i32> [#uses=1]
	%21 = add i32 %20, 2147483647		; <i32> [#uses=1]
	store i32 %21, i32* @_ZZL13random_doublevE4seed, align 4
	br label %_ZL13random_doublev.exit

_ZL13random_doublev.exit:		; preds = %bb.i, %bb7
	call void @llvm.dbg.stoppoint(i32 75, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%22 = load i32* @_ZZL13random_doublevE4seed, align 4		; <i32> [#uses=2]
	%23 = sitofp i32 %22 to double		; <double> [#uses=1]
	%24 = mul double %23, 0x3E340000002813D9		; <double> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 76, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%25 = xor i32 %22, 123459876		; <i32> [#uses=1]
	store i32 %25, i32* @_ZZL13random_doublevE4seed, align 4
	call void @llvm.dbg.stoppoint(i32 78, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	store double %24, double* %10, align 8
	call void @llvm.dbg.stoppoint(i32 679, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%n.0.reload4 = load i32* %n.0.reg2mem		; <i32> [#uses=1]
	%26 = call double* @_ZN10polynomialIdEixEj(%"struct.polynomial<double>"* %poly2, i32 %n.0.reload4) nounwind		; <double*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 68, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%27 = load i32* @_ZZL13random_doublevE4seed, align 4		; <i32> [#uses=1]
	%28 = xor i32 %27, 123459876		; <i32> [#uses=3]
	store i32 %28, i32* @_ZZL13random_doublevE4seed, align 4
	call void @llvm.dbg.stoppoint(i32 69, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%29 = sdiv i32 %28, 127773		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 70, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%30 = mul i32 %29, 127773		; <i32> [#uses=1]
	%31 = sub i32 %28, %30		; <i32> [#uses=1]
	%32 = mul i32 %31, 16807		; <i32> [#uses=1]
	%33 = mul i32 %29, 2836		; <i32> [#uses=1]
	%34 = sub i32 %32, %33		; <i32> [#uses=2]
	store i32 %34, i32* @_ZZL13random_doublevE4seed, align 4
	call void @llvm.dbg.stoppoint(i32 72, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%35 = icmp slt i32 %34, 0		; <i1> [#uses=1]
	br i1 %35, label %bb.i1, label %bb8

bb.i1:		; preds = %_ZL13random_doublev.exit
	call void @llvm.dbg.stoppoint(i32 73, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%36 = load i32* @_ZZL13random_doublevE4seed, align 4		; <i32> [#uses=1]
	%37 = add i32 %36, 2147483647		; <i32> [#uses=1]
	store i32 %37, i32* @_ZZL13random_doublevE4seed, align 4
	br label %bb8

bb8:		; preds = %bb.i1, %_ZL13random_doublev.exit
	call void @llvm.dbg.stoppoint(i32 75, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%38 = load i32* @_ZZL13random_doublevE4seed, align 4		; <i32> [#uses=2]
	%39 = sitofp i32 %38 to double		; <double> [#uses=1]
	%40 = mul double %39, 0x3E340000002813D9		; <double> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 76, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%41 = xor i32 %38, 123459876		; <i32> [#uses=1]
	store i32 %41, i32* @_ZZL13random_doublevE4seed, align 4
	call void @llvm.dbg.stoppoint(i32 78, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	store double %40, double* %26, align 8
	call void @llvm.dbg.stoppoint(i32 676, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%n.0.reload5 = load i32* %n.0.reg2mem		; <i32> [#uses=1]
	%42 = add i32 %n.0.reload5, 1		; <i32> [#uses=2]
	store i32 %42, i32* %n.0.reg2mem
	call void @llvm.dbg.stoppoint(i32 676, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%43 = icmp sgt i32 %42, 524287		; <i1> [#uses=1]
	br i1 %43, label %bb9, label %bb7

bb9:		; preds = %bb8
	call void @llvm.dbg.stoppoint(i32 687, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZNK10polynomialIdEmlERKS0_(%"struct.polynomial<double>"* noalias sret %0, %"struct.polynomial<double>"* %poly1, %"struct.polynomial<double>"* %poly2)
			to label %invcont10 unwind label %lpad51

invcont10:		; preds = %bb9
	call void @llvm.dbg.stoppoint(i32 687, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%44 = invoke %"struct.polynomial<double>"* @_ZN10polynomialIdEaSERKS0_(%"struct.polynomial<double>"* %poly3, %"struct.polynomial<double>"* %0)
			to label %invcont11 unwind label %lpad55		; <%"struct.polynomial<double>"*> [#uses=0]

invcont11:		; preds = %invcont10
	call void @llvm.dbg.stoppoint(i32 687, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %0)
			to label %bb16 unwind label %lpad51

bb16:		; preds = %invcont11
	call void @llvm.dbg.stoppoint(i32 695, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%toBool = icmp eq i8 %ga_testing.0, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb19, label %bb17

bb17:		; preds = %bb16
	call void @llvm.dbg.stoppoint(i32 696, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%45 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEd(%"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4cout, double 0.000000e+00)
			to label %bb23 unwind label %lpad51		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]

bb19:		; preds = %bb16
	call void @llvm.dbg.stoppoint(i32 698, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%46 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4cout, i8* getelementptr ([32 x i8]* @.str1825, i32 0, i32 0))
			to label %invcont20 unwind label %lpad51		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]

invcont20:		; preds = %bb19
	call void @llvm.dbg.stoppoint(i32 698, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%47 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEd(%"struct.std::basic_ostream<char,std::char_traits<char> >"* %46, double 0.000000e+00)
			to label %invcont21 unwind label %lpad51		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]

invcont21:		; preds = %invcont20
	call void @llvm.dbg.stoppoint(i32 698, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%48 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"struct.std::basic_ostream<char,std::char_traits<char> >"* %47, i8* getelementptr ([3 x i8]* @.str1826, i32 0, i32 0))
			to label %bb23 unwind label %lpad51		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]

bb23:		; preds = %invcont21, %bb17
	call void @llvm.dbg.stoppoint(i32 700, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%49 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSo5flushEv(%"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4cout)
			to label %invcont24 unwind label %lpad51		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]

invcont24:		; preds = %bb23
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %poly3)
			to label %bb31 unwind label %lpad47

bb31:		; preds = %invcont24
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %poly2)
			to label %bb38 unwind label %lpad

bb38:		; preds = %bb31
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %poly1)
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1806 to %0*))
	ret i32 0

lpad:		; preds = %bb31, %bb5
	%eh_ptr = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select46 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad

lpad47:		; preds = %invcont24, %invcont
	%eh_ptr48 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select50 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr48, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad75

lpad51:		; preds = %bb23, %invcont21, %invcont20, %bb19, %bb17, %invcont11, %bb9
	%eh_ptr52 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select54 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr52, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad76

lpad55:		; preds = %invcont10
	%eh_ptr56 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select58 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr56, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 687, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %0)
			to label %ppad76 unwind label %lpad59

lpad59:		; preds = %lpad55
	%eh_ptr60 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select62 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr60, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 687, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad63:		; preds = %ppad76
	%eh_ptr64 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select66 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr64, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad67:		; preds = %ppad75
	%eh_ptr68 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select70 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr68, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad71:		; preds = %ppad
	%eh_ptr72 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select74 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr72, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

ppad:		; preds = %ppad75, %lpad
	%eh_exception.2 = phi i8* [ %eh_ptr, %lpad ], [ %eh_exception.1, %ppad75 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %poly1)
			to label %Unwind unwind label %lpad71

ppad75:		; preds = %ppad76, %lpad47
	%eh_exception.1 = phi i8* [ %eh_ptr48, %lpad47 ], [ %eh_exception.0, %ppad76 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %poly2)
			to label %ppad unwind label %lpad67

ppad76:		; preds = %lpad55, %lpad51
	%eh_exception.0 = phi i8* [ %eh_ptr52, %lpad51 ], [ %eh_ptr56, %lpad55 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 702, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %poly3)
			to label %ppad75 unwind label %lpad63

Unwind:		; preds = %ppad
	call void @_Unwind_Resume(i8* %eh_exception.2)
	unreachable
}

define internal void @_GLOBAL__I_main() {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram657 to %0*))
	call void @llvm.dbg.stoppoint(i32 77, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit636 to %0*))
	call void @_ZNSt8ios_base4InitC1Ev(%"struct.std::allocator<char>"* @_ZStL8__ioinit)
	%0 = call i32 @__cxa_atexit(void (i8*)* @__tcf_0, i8* null, i8* bitcast (i8** @__dso_handle to i8*)) nounwind		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 400, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = load i8* bitcast (i64* @_ZGVN10polynomialIdE4PI2IE to i8*), align 8		; <i8> [#uses=1]
	%2 = icmp eq i8 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb2.i, label %_Z41__static_initialization_and_destruction_0ii.exit

bb2.i:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 400, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	store i8 1, i8* bitcast (i64* @_ZGVN10polynomialIdE4PI2IE to i8*), align 8
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* @_ZN10polynomialIdE4PI2IE, double 0.000000e+00, double 0x401921FB54442D18) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram657 to %0*))
	ret void

_Z41__static_initialization_and_destruction_0ii.exit:		; preds = %entry
	ret void
}

define linkonce void @_ZNSt7complexIdEC1ECd(%"struct.std::complex<double>"* %this, %1 %__z) nounwind {
entry:
	%__z_addr = alloca %1, align 8		; <%1*> [#uses=4]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram227 to %0*))
	%0 = bitcast %1* %__z_addr to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable231 to %0*))
	store %1 %__z, %1* %__z_addr, align 8
	call void @llvm.dbg.stoppoint(i32 1161, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%1 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%2 = getelementptr %1* %__z_addr, i32 0, i32 0		; <double*> [#uses=1]
	%3 = load double* %2, align 8		; <double> [#uses=1]
	store double %3, double* %1, align 4
	%4 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%5 = getelementptr %1* %__z_addr, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 8		; <double> [#uses=1]
	store double %6, double* %4, align 4
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram227 to %0*))
	ret void
}

declare void @llvm.dbg.func.start(%0*) nounwind readnone

declare void @llvm.dbg.declare(%0*, %0*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, %0*) nounwind readnone

declare void @llvm.dbg.region.end(%0*) nounwind readnone

define linkonce %1* @_ZNKSt7complexIdE5__repEv(%"struct.std::complex<double>"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram232 to %0*))
	call void @llvm.dbg.stoppoint(i32 1192, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0		; <%1*> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram232 to %0*))
	ret %1* %0
}

define linkonce double* @_ZNSt7complexIdE4realEv(%"struct.std::complex<double>"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram235 to %0*))
	call void @llvm.dbg.stoppoint(i32 1200, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram235 to %0*))
	ret double* %0
}

define linkonce double* @_ZNKSt7complexIdE4realEv(%"struct.std::complex<double>"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram237 to %0*))
	call void @llvm.dbg.stoppoint(i32 1204, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram237 to %0*))
	ret double* %0
}

define linkonce double* @_ZNKSt7complexIdE4imagEv(%"struct.std::complex<double>"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram239 to %0*))
	call void @llvm.dbg.stoppoint(i32 1212, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram239 to %0*))
	ret double* %0
}

define linkonce void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %this, double %__r, double %__i) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram242 to %0*))
	call void @llvm.dbg.stoppoint(i32 1217, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	store double %__r, double* %0, align 4
	call void @llvm.dbg.stoppoint(i32 1218, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%1 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	store double %__i, double* %1, align 4
	call void @llvm.dbg.stoppoint(i32 1219, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram242 to %0*))
	ret void
}

define linkonce %"struct.std::complex<double>"* @_ZNSt7complexIdEaSEd(%"struct.std::complex<double>"* %this, double %__d) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram248 to %0*))
	call void @llvm.dbg.stoppoint(i32 1224, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	store double %__d, double* %0, align 4
	call void @llvm.dbg.stoppoint(i32 1225, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%1 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	store double 0.000000e+00, double* %1, align 4
	call void @llvm.dbg.stoppoint(i32 1226, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram248 to %0*))
	ret %"struct.std::complex<double>"* %this
}

define linkonce %"struct.std::complex<double>"* @_ZNSt7complexIdEdVEd(%"struct.std::complex<double>"* %this, double %__d) nounwind {
entry:
	%0 = alloca %1, align 8		; <%1*> [#uses=4]
	%1 = alloca %1, align 8		; <%1*> [#uses=4]
	%2 = alloca %1, align 8		; <%1*> [#uses=4]
	%memtmp = alloca %1, align 8		; <%1*> [#uses=4]
	%memtmp1 = alloca %1, align 8		; <%1*> [#uses=4]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram252 to %0*))
	call void @llvm.dbg.stoppoint(i32 1253, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%3 = getelementptr %1* %2, i32 0, i32 0		; <double*> [#uses=1]
	%4 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%5 = load double* %4, align 4		; <double> [#uses=1]
	store double %5, double* %3, align 8
	%6 = getelementptr %1* %2, i32 0, i32 1		; <double*> [#uses=1]
	%7 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%8 = load double* %7, align 4		; <double> [#uses=1]
	store double %8, double* %6, align 8
	%real = getelementptr %1* %1, i32 0, i32 0		; <double*> [#uses=1]
	store double %__d, double* %real, align 8
	%imag = getelementptr %1* %1, i32 0, i32 1		; <double*> [#uses=1]
	store double 0.000000e+00, double* %imag, align 8
	%9 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%10 = getelementptr %1* %2, i32 0, i32 0		; <double*> [#uses=1]
	%11 = load double* %10, align 8		; <double> [#uses=1]
	store double %11, double* %9, align 8
	%12 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%13 = getelementptr %1* %2, i32 0, i32 1		; <double*> [#uses=1]
	%14 = load double* %13, align 8		; <double> [#uses=1]
	store double %14, double* %12, align 8
	%15 = getelementptr %1* %memtmp1, i32 0, i32 0		; <double*> [#uses=1]
	%16 = getelementptr %1* %1, i32 0, i32 0		; <double*> [#uses=1]
	%17 = load double* %16, align 8		; <double> [#uses=1]
	store double %17, double* %15, align 8
	%18 = getelementptr %1* %memtmp1, i32 0, i32 1		; <double*> [#uses=1]
	%19 = getelementptr %1* %1, i32 0, i32 1		; <double*> [#uses=1]
	%20 = load double* %19, align 8		; <double> [#uses=1]
	store double %20, double* %18, align 8
	%real2 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%real3 = load double* %real2, align 8		; <double> [#uses=2]
	%imag4 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%imag5 = load double* %imag4, align 8		; <double> [#uses=2]
	%real6 = getelementptr %1* %memtmp1, i32 0, i32 0		; <double*> [#uses=1]
	%real7 = load double* %real6, align 8		; <double> [#uses=4]
	%imag8 = getelementptr %1* %memtmp1, i32 0, i32 1		; <double*> [#uses=1]
	%imag9 = load double* %imag8, align 8		; <double> [#uses=4]
	%21 = mul double %real3, %real7		; <double> [#uses=1]
	%22 = mul double %imag5, %imag9		; <double> [#uses=1]
	%23 = add double %21, %22		; <double> [#uses=1]
	%24 = mul double %real7, %real7		; <double> [#uses=1]
	%25 = mul double %imag9, %imag9		; <double> [#uses=1]
	%26 = add double %24, %25		; <double> [#uses=2]
	%27 = fdiv double %23, %26		; <double> [#uses=1]
	%28 = mul double %imag5, %real7		; <double> [#uses=1]
	%29 = mul double %real3, %imag9		; <double> [#uses=1]
	%30 = sub double %28, %29		; <double> [#uses=1]
	%31 = fdiv double %30, %26		; <double> [#uses=1]
	%real10 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	store double %27, double* %real10, align 8
	%imag11 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	store double %31, double* %imag11, align 8
	%32 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%33 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	%34 = load double* %33, align 8		; <double> [#uses=1]
	store double %34, double* %32, align 4
	%35 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%36 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	%37 = load double* %36, align 8		; <double> [#uses=1]
	store double %37, double* %35, align 4
	call void @llvm.dbg.stoppoint(i32 1254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram252 to %0*))
	ret %"struct.std::complex<double>"* %this
}

define linkonce double* @_ZN10polynomialIdEixEj(%"struct.polynomial<double>"* %this, i32 %term) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram522 to %0*))
	call void @llvm.dbg.stoppoint(i32 308, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%1 = load double** %0, align 4		; <double*> [#uses=1]
	%2 = getelementptr double* %1, i32 %term		; <double*> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram522 to %0*))
	ret double* %2
}

define linkonce i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram527 to %0*))
	call void @llvm.dbg.stoppoint(i32 113, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram527 to %0*))
	ret i32 %1
}

define linkonce %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %this, i32 %term) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram530 to %0*))
	call void @llvm.dbg.stoppoint(i32 308, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	%1 = load %"struct.std::complex<double>"** %0, align 4		; <%"struct.std::complex<double>"*> [#uses=1]
	%2 = getelementptr %"struct.std::complex<double>"* %1, i32 %term		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram530 to %0*))
	ret %"struct.std::complex<double>"* %2
}

define linkonce %"struct.std::complex<double>"* @_ZNSt7complexIdEmLIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %this, %"struct.std::complex<double>"* %__z) nounwind {
entry:
	%__t = alloca %1, align 8		; <%1*> [#uses=7]
	%0 = alloca %1, align 8		; <%1*> [#uses=4]
	%1 = alloca %1, align 8		; <%1*> [#uses=4]
	%memtmp = alloca %1, align 8		; <%1*> [#uses=4]
	%memtmp3 = alloca %1, align 8		; <%1*> [#uses=4]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram538 to %0*))
	%2 = bitcast %1* %__t to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %2, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable542 to %0*))
	call void @llvm.dbg.stoppoint(i32 1289, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%3 = call double* @_ZNKSt7complexIdE4realEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%4 = load double* %3, align 8		; <double> [#uses=1]
	%5 = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 8		; <double> [#uses=1]
	%real = getelementptr %1* %__t, i32 0, i32 0		; <double*> [#uses=1]
	store double %4, double* %real, align 8
	%imag = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	store double %6, double* %imag, align 8
	call void @llvm.dbg.stoppoint(i32 1290, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%7 = call double* @_ZNKSt7complexIdE4imagEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%8 = load double* %7, align 8		; <double> [#uses=1]
	%imag2 = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	store double %8, double* %imag2, align 8
	call void @llvm.dbg.stoppoint(i32 1291, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%9 = getelementptr %1* %1, i32 0, i32 0		; <double*> [#uses=1]
	%10 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%11 = load double* %10, align 4		; <double> [#uses=1]
	store double %11, double* %9, align 8
	%12 = getelementptr %1* %1, i32 0, i32 1		; <double*> [#uses=1]
	%13 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%14 = load double* %13, align 4		; <double> [#uses=1]
	store double %14, double* %12, align 8
	%15 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%16 = getelementptr %1* %1, i32 0, i32 0		; <double*> [#uses=1]
	%17 = load double* %16, align 8		; <double> [#uses=1]
	store double %17, double* %15, align 8
	%18 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%19 = getelementptr %1* %1, i32 0, i32 1		; <double*> [#uses=1]
	%20 = load double* %19, align 8		; <double> [#uses=1]
	store double %20, double* %18, align 8
	%21 = getelementptr %1* %memtmp3, i32 0, i32 0		; <double*> [#uses=1]
	%22 = getelementptr %1* %__t, i32 0, i32 0		; <double*> [#uses=1]
	%23 = load double* %22, align 8		; <double> [#uses=1]
	store double %23, double* %21, align 8
	%24 = getelementptr %1* %memtmp3, i32 0, i32 1		; <double*> [#uses=1]
	%25 = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	%26 = load double* %25, align 8		; <double> [#uses=1]
	store double %26, double* %24, align 8
	%real4 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%real5 = load double* %real4, align 8		; <double> [#uses=2]
	%imag6 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%imag7 = load double* %imag6, align 8		; <double> [#uses=2]
	%real8 = getelementptr %1* %memtmp3, i32 0, i32 0		; <double*> [#uses=1]
	%real9 = load double* %real8, align 8		; <double> [#uses=2]
	%imag10 = getelementptr %1* %memtmp3, i32 0, i32 1		; <double*> [#uses=1]
	%imag11 = load double* %imag10, align 8		; <double> [#uses=2]
	%27 = mul double %real5, %real9		; <double> [#uses=1]
	%28 = mul double %imag7, %imag11		; <double> [#uses=1]
	%29 = sub double %27, %28		; <double> [#uses=1]
	%30 = mul double %real5, %imag11		; <double> [#uses=1]
	%31 = mul double %real9, %imag7		; <double> [#uses=1]
	%32 = add double %30, %31		; <double> [#uses=1]
	%real12 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	store double %29, double* %real12, align 8
	%imag13 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	store double %32, double* %imag13, align 8
	%33 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%34 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	%35 = load double* %34, align 8		; <double> [#uses=1]
	store double %35, double* %33, align 4
	%36 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%37 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	%38 = load double* %37, align 8		; <double> [#uses=1]
	store double %38, double* %36, align 4
	call void @llvm.dbg.stoppoint(i32 1292, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram538 to %0*))
	ret %"struct.std::complex<double>"* %this
}

define linkonce void @_ZN10polynomialIdE9deep_copyEPKd(%"struct.polynomial<double>"* %this, double* %source) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram543 to %0*))
	call void @llvm.dbg.stoppoint(i32 197, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb1

bb:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 198, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%1 = load double** %0, align 4		; <double*> [#uses=1]
	%2 = getelementptr double* %source, i32 %n.0		; <double*> [#uses=1]
	%3 = load double* %2, align 1		; <double> [#uses=1]
	%4 = getelementptr double* %1, i32 %n.0		; <double*> [#uses=1]
	store double %3, double* %4, align 1
	call void @llvm.dbg.stoppoint(i32 197, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%5 = add i32 %n.0, 1		; <i32> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%n.0 = phi i32 [ 0, %entry ], [ %5, %bb ]		; <i32> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 197, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%6 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%7 = load i32* %6, align 4		; <i32> [#uses=1]
	%8 = icmp ugt i32 %7, %n.0		; <i1> [#uses=1]
	br i1 %8, label %bb, label %return

return:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 198, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram543 to %0*))
	ret void
}

define linkonce i32 @_ZN10polynomialIdE4log2Ej(i32 %n) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram549 to %0*))
	call void @llvm.dbg.stoppoint(i32 407, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb1

bb:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 411, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = add i32 %c.0, 1		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 412, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = shl i32 %x.0, 1		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 414, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%2 = icmp eq i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb1

bb1:		; preds = %bb, %entry
	%c.0 = phi i32 [ 0, %entry ], [ %0, %bb ]		; <i32> [#uses=2]
	%x.0 = phi i32 [ 1, %entry ], [ %1, %bb ]		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 409, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = icmp ult i32 %x.0, %n		; <i1> [#uses=1]
	br i1 %3, label %bb, label %return

return:		; preds = %bb1, %bb
	%c.1 = phi i32 [ %c.0, %bb1 ], [ %0, %bb ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 418, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram549 to %0*))
	ret i32 %c.1
}

define linkonce void @_ZStmlIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %agg.result, %"struct.std::complex<double>"* %__x, %"struct.std::complex<double>"* %__y) nounwind {
entry:
	%__r = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram559 to %0*))
	%0 = bitcast %"struct.std::complex<double>"* %__r to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable564 to %0*))
	call void @llvm.dbg.stoppoint(i32 380, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%1 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%2 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%3 = load double* %2, align 4		; <double> [#uses=1]
	store double %3, double* %1, align 4
	%4 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%5 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 4		; <double> [#uses=1]
	store double %6, double* %4, align 4
	call void @llvm.dbg.stoppoint(i32 381, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%7 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEmLIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %agg.result, %"struct.std::complex<double>"* %__y) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram559 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialISt7complexIdEE9deep_copyEPKS1_(%"struct.polynomial<std::complex<double> >"* %this, %"struct.std::complex<double>"* %source) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram565 to %0*))
	call void @llvm.dbg.stoppoint(i32 197, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb1

bb:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 198, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	%1 = load %"struct.std::complex<double>"** %0, align 4		; <%"struct.std::complex<double>"*> [#uses=2]
	%2 = getelementptr %"struct.std::complex<double>"* %1, i32 %n.0, i32 0, i32 0		; <double*> [#uses=1]
	%3 = getelementptr %"struct.std::complex<double>"* %source, i32 %n.0, i32 0, i32 0		; <double*> [#uses=1]
	%4 = load double* %3, align 1		; <double> [#uses=1]
	store double %4, double* %2, align 1
	%5 = getelementptr %"struct.std::complex<double>"* %1, i32 %n.0, i32 0, i32 1		; <double*> [#uses=1]
	%6 = getelementptr %"struct.std::complex<double>"* %source, i32 %n.0, i32 0, i32 1		; <double*> [#uses=1]
	%7 = load double* %6, align 1		; <double> [#uses=1]
	store double %7, double* %5, align 1
	call void @llvm.dbg.stoppoint(i32 197, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%8 = add i32 %n.0, 1		; <i32> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%n.0 = phi i32 [ 0, %entry ], [ %8, %bb ]		; <i32> [#uses=6]
	call void @llvm.dbg.stoppoint(i32 197, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%9 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%10 = load i32* %9, align 4		; <i32> [#uses=1]
	%11 = icmp ugt i32 %10, %n.0		; <i1> [#uses=1]
	br i1 %11, label %bb, label %return

return:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 198, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram565 to %0*))
	ret void
}

define linkonce i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram569 to %0*))
	call void @llvm.dbg.stoppoint(i32 113, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram569 to %0*))
	ret i32 %1
}

define linkonce void @_ZStngIdESt7complexIT_ERKS2_(%"struct.std::complex<double>"* noalias sret %agg.result, %"struct.std::complex<double>"* %__x) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram576 to %0*))
	call void @llvm.dbg.stoppoint(i32 444, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = call double* @_ZNKSt7complexIdE4imagEv(%"struct.std::complex<double>"* %__x) nounwind		; <double*> [#uses=1]
	%1 = load double* %0, align 8		; <double> [#uses=1]
	%2 = sub double -0.000000e+00, %1		; <double> [#uses=1]
	%3 = call double* @_ZNKSt7complexIdE4realEv(%"struct.std::complex<double>"* %__x) nounwind		; <double*> [#uses=1]
	%4 = load double* %3, align 8		; <double> [#uses=1]
	%5 = sub double -0.000000e+00, %4		; <double> [#uses=1]
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %agg.result, double %5, double %2) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram576 to %0*))
	ret void
}

define linkonce double @_ZNK10polynomialIdE3getEj(%"struct.polynomial<double>"* %this, i32 %term) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram578 to %0*))
	call void @llvm.dbg.stoppoint(i32 302, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%1 = load double** %0, align 4		; <double*> [#uses=1]
	%2 = getelementptr double* %1, i32 %term		; <double*> [#uses=1]
	%3 = load double* %2, align 1		; <double> [#uses=1]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram578 to %0*))
	ret double %3
}

define linkonce i32 @_ZN10polynomialIdE9flip_bitsEjj(i32 %k, i32 %bits) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram581 to %0*))
	call void @llvm.dbg.stoppoint(i32 425, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = add i32 %bits, -1		; <i32> [#uses=1]
	%1 = shl i32 1, %0		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 427, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb3

bb:		; preds = %bb3
	call void @llvm.dbg.stoppoint(i32 431, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%2 = and i32 %rm.0, %k		; <i32> [#uses=1]
	%3 = icmp ne i32 %2, 0		; <i1> [#uses=1]
	%4 = select i1 %3, i32 %lm.0, i32 0		; <i32> [#uses=1]
	%.r.1 = or i32 %r.1, %4		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 434, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%5 = lshr i32 %lm.0, 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 435, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%6 = shl i32 %rm.0, 1		; <i32> [#uses=1]
	br label %bb3

bb3:		; preds = %bb, %entry
	%r.1 = phi i32 [ 0, %entry ], [ %.r.1, %bb ]		; <i32> [#uses=2]
	%rm.0 = phi i32 [ 1, %entry ], [ %6, %bb ]		; <i32> [#uses=2]
	%lm.0 = phi i32 [ %1, %entry ], [ %5, %bb ]		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 429, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%7 = icmp eq i32 %lm.0, 0		; <i1> [#uses=1]
	br i1 %7, label %return, label %bb

return:		; preds = %bb3
	call void @llvm.dbg.stoppoint(i32 438, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram581 to %0*))
	ret i32 %r.1
}

define linkonce %"struct.std::complex<double>"* @_ZNSt7complexIdEdVIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %this, %"struct.std::complex<double>"* %__z) nounwind {
entry:
	%__t = alloca %1, align 8		; <%1*> [#uses=7]
	%0 = alloca %1, align 8		; <%1*> [#uses=4]
	%1 = alloca %1, align 8		; <%1*> [#uses=4]
	%memtmp = alloca %1, align 8		; <%1*> [#uses=4]
	%memtmp3 = alloca %1, align 8		; <%1*> [#uses=4]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram593 to %0*))
	%2 = bitcast %1* %__t to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %2, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable596 to %0*))
	call void @llvm.dbg.stoppoint(i32 1300, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%3 = call double* @_ZNKSt7complexIdE4realEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%4 = load double* %3, align 8		; <double> [#uses=1]
	%5 = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 8		; <double> [#uses=1]
	%real = getelementptr %1* %__t, i32 0, i32 0		; <double*> [#uses=1]
	store double %4, double* %real, align 8
	%imag = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	store double %6, double* %imag, align 8
	call void @llvm.dbg.stoppoint(i32 1301, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%7 = call double* @_ZNKSt7complexIdE4imagEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%8 = load double* %7, align 8		; <double> [#uses=1]
	%imag2 = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	store double %8, double* %imag2, align 8
	call void @llvm.dbg.stoppoint(i32 1302, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%9 = getelementptr %1* %1, i32 0, i32 0		; <double*> [#uses=1]
	%10 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%11 = load double* %10, align 4		; <double> [#uses=1]
	store double %11, double* %9, align 8
	%12 = getelementptr %1* %1, i32 0, i32 1		; <double*> [#uses=1]
	%13 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%14 = load double* %13, align 4		; <double> [#uses=1]
	store double %14, double* %12, align 8
	%15 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%16 = getelementptr %1* %1, i32 0, i32 0		; <double*> [#uses=1]
	%17 = load double* %16, align 8		; <double> [#uses=1]
	store double %17, double* %15, align 8
	%18 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%19 = getelementptr %1* %1, i32 0, i32 1		; <double*> [#uses=1]
	%20 = load double* %19, align 8		; <double> [#uses=1]
	store double %20, double* %18, align 8
	%21 = getelementptr %1* %memtmp3, i32 0, i32 0		; <double*> [#uses=1]
	%22 = getelementptr %1* %__t, i32 0, i32 0		; <double*> [#uses=1]
	%23 = load double* %22, align 8		; <double> [#uses=1]
	store double %23, double* %21, align 8
	%24 = getelementptr %1* %memtmp3, i32 0, i32 1		; <double*> [#uses=1]
	%25 = getelementptr %1* %__t, i32 0, i32 1		; <double*> [#uses=1]
	%26 = load double* %25, align 8		; <double> [#uses=1]
	store double %26, double* %24, align 8
	%real4 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%real5 = load double* %real4, align 8		; <double> [#uses=2]
	%imag6 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%imag7 = load double* %imag6, align 8		; <double> [#uses=2]
	%real8 = getelementptr %1* %memtmp3, i32 0, i32 0		; <double*> [#uses=1]
	%real9 = load double* %real8, align 8		; <double> [#uses=4]
	%imag10 = getelementptr %1* %memtmp3, i32 0, i32 1		; <double*> [#uses=1]
	%imag11 = load double* %imag10, align 8		; <double> [#uses=4]
	%27 = mul double %real5, %real9		; <double> [#uses=1]
	%28 = mul double %imag7, %imag11		; <double> [#uses=1]
	%29 = add double %27, %28		; <double> [#uses=1]
	%30 = mul double %real9, %real9		; <double> [#uses=1]
	%31 = mul double %imag11, %imag11		; <double> [#uses=1]
	%32 = add double %30, %31		; <double> [#uses=2]
	%33 = fdiv double %29, %32		; <double> [#uses=1]
	%34 = mul double %imag7, %real9		; <double> [#uses=1]
	%35 = mul double %real5, %imag11		; <double> [#uses=1]
	%36 = sub double %34, %35		; <double> [#uses=1]
	%37 = fdiv double %36, %32		; <double> [#uses=1]
	%real12 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	store double %33, double* %real12, align 8
	%imag13 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	store double %37, double* %imag13, align 8
	%38 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%39 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	%40 = load double* %39, align 8		; <double> [#uses=1]
	store double %40, double* %38, align 4
	%41 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%42 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	%43 = load double* %42, align 8		; <double> [#uses=1]
	store double %43, double* %41, align 4
	call void @llvm.dbg.stoppoint(i32 1303, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram593 to %0*))
	ret %"struct.std::complex<double>"* %this
}

define linkonce void @_ZStdvIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %agg.result, %"struct.std::complex<double>"* %__x, %"struct.std::complex<double>"* %__y) {
entry:
	%__r = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram599 to %0*))
	%0 = bitcast %"struct.std::complex<double>"* %__r to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable602 to %0*))
	call void @llvm.dbg.stoppoint(i32 410, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%1 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%2 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%3 = load double* %2, align 4		; <double> [#uses=1]
	store double %3, double* %1, align 4
	%4 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%5 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 4		; <double> [#uses=1]
	store double %6, double* %4, align 4
	call void @llvm.dbg.stoppoint(i32 411, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%7 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEdVIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %agg.result, %"struct.std::complex<double>"* %__y) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram599 to %0*))
	ret void
}

define linkonce %"struct.std::complex<double>"* @_ZNSt7complexIdEpLIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %this, %"struct.std::complex<double>"* %__z) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram605 to %0*))
	call void @llvm.dbg.stoppoint(i32 1270, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%1 = load double* %0, align 4		; <double> [#uses=1]
	%2 = call double* @_ZNKSt7complexIdE4realEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%3 = load double* %2, align 8		; <double> [#uses=1]
	%4 = add double %1, %3		; <double> [#uses=1]
	%5 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	store double %4, double* %5, align 4
	call void @llvm.dbg.stoppoint(i32 1271, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%6 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%7 = load double* %6, align 4		; <double> [#uses=1]
	%8 = call double* @_ZNKSt7complexIdE4imagEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%9 = load double* %8, align 8		; <double> [#uses=1]
	%10 = add double %7, %9		; <double> [#uses=1]
	%11 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	store double %10, double* %11, align 4
	call void @llvm.dbg.stoppoint(i32 1272, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram605 to %0*))
	ret %"struct.std::complex<double>"* %this
}

define linkonce void @_ZStplIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %agg.result, %"struct.std::complex<double>"* %__x, %"struct.std::complex<double>"* %__y) {
entry:
	%__r = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram610 to %0*))
	%0 = bitcast %"struct.std::complex<double>"* %__r to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable613 to %0*))
	call void @llvm.dbg.stoppoint(i32 320, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%1 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%2 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%3 = load double* %2, align 4		; <double> [#uses=1]
	store double %3, double* %1, align 4
	%4 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%5 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 4		; <double> [#uses=1]
	store double %6, double* %4, align 4
	call void @llvm.dbg.stoppoint(i32 321, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%7 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEpLIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %agg.result, %"struct.std::complex<double>"* %__y) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram610 to %0*))
	ret void
}

define linkonce %"struct.std::complex<double>"* @_ZNSt7complexIdEmIIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %this, %"struct.std::complex<double>"* %__z) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram616 to %0*))
	call void @llvm.dbg.stoppoint(i32 1279, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%0 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%1 = load double* %0, align 4		; <double> [#uses=1]
	%2 = call double* @_ZNKSt7complexIdE4realEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%3 = load double* %2, align 8		; <double> [#uses=1]
	%4 = sub double %1, %3		; <double> [#uses=1]
	%5 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	store double %4, double* %5, align 4
	call void @llvm.dbg.stoppoint(i32 1280, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%6 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%7 = load double* %6, align 4		; <double> [#uses=1]
	%8 = call double* @_ZNKSt7complexIdE4imagEv(%"struct.std::complex<double>"* %__z) nounwind		; <double*> [#uses=1]
	%9 = load double* %8, align 8		; <double> [#uses=1]
	%10 = sub double %7, %9		; <double> [#uses=1]
	%11 = getelementptr %"struct.std::complex<double>"* %this, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	store double %10, double* %11, align 4
	call void @llvm.dbg.stoppoint(i32 1281, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram616 to %0*))
	ret %"struct.std::complex<double>"* %this
}

define linkonce void @_ZStmiIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %agg.result, %"struct.std::complex<double>"* %__x, %"struct.std::complex<double>"* %__y) {
entry:
	%__r = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram620 to %0*))
	%0 = bitcast %"struct.std::complex<double>"* %__r to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable623 to %0*))
	call void @llvm.dbg.stoppoint(i32 350, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%1 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%2 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%3 = load double* %2, align 4		; <double> [#uses=1]
	store double %3, double* %1, align 4
	%4 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%5 = getelementptr %"struct.std::complex<double>"* %__x, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 4		; <double> [#uses=1]
	store double %6, double* %4, align 4
	call void @llvm.dbg.stoppoint(i32 351, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%7 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEmIIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %agg.result, %"struct.std::complex<double>"* %__y) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram620 to %0*))
	ret void
}

define linkonce void @_ZNK10polynomialISt7complexIdEE3getEj(%"struct.std::complex<double>"* noalias sret %agg.result, %"struct.polynomial<std::complex<double> >"* %this, i32 %term) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram624 to %0*))
	call void @llvm.dbg.stoppoint(i32 302, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	%1 = load %"struct.std::complex<double>"** %0, align 4		; <%"struct.std::complex<double>"*> [#uses=2]
	%2 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%3 = getelementptr %"struct.std::complex<double>"* %1, i32 %term, i32 0, i32 0		; <double*> [#uses=1]
	%4 = load double* %3, align 1		; <double> [#uses=1]
	store double %4, double* %2, align 1
	%5 = getelementptr %"struct.std::complex<double>"* %agg.result, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%6 = getelementptr %"struct.std::complex<double>"* %1, i32 %term, i32 0, i32 1		; <double*> [#uses=1]
	%7 = load double* %6, align 1		; <double> [#uses=1]
	store double %7, double* %5, align 1
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram624 to %0*))
	ret void
}

declare void @_ZNSt8ios_base4InitC1Ev(%"struct.std::allocator<char>"*)

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) nounwind

define internal void @__tcf_0(i8* nocapture %unnamed_arg) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram662 to %0*))
	call void @llvm.dbg.stoppoint(i32 77, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit636 to %0*))
	call void @_ZNSt8ios_base4InitD1Ev(%"struct.std::allocator<char>"* @_ZStL8__ioinit)
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram662 to %0*))
	ret void
}

declare void @_ZNSt8ios_base4InitD1Ev(%"struct.std::allocator<char>"*)

define linkonce void @_ZN10polynomialIdE7releaseEv(%"struct.polynomial<double>"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram665 to %0*))
	call void @llvm.dbg.stoppoint(i32 190, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%1 = load double** %0, align 4		; <double*> [#uses=1]
	%2 = icmp eq double* %1, null		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb

bb:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 190, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%4 = load double** %3, align 4		; <double*> [#uses=1]
	%5 = bitcast double* %4 to i8*		; <i8*> [#uses=1]
	call void @_ZdaPv(i8* %5) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram665 to %0*))
	ret void

return:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 190, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	ret void
}

declare void @_ZdaPv(i8*) nounwind

define linkonce void @_ZN10polynomialIdED0Ev(%"struct.polynomial<double>"* %this) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram668 to %0*))
	call void @llvm.dbg.stoppoint(i32 255, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @_ZTV10polynomialIdE, i32 0, i32 2), i32 (...)*** %0, align 4
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE7releaseEv(%"struct.polynomial<double>"* %this) nounwind
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = bitcast %"struct.polynomial<double>"* %this to i8*		; <i8*> [#uses=1]
	call void @_ZdlPv(i8* %1) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram668 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %this) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram691 to %0*))
	call void @llvm.dbg.stoppoint(i32 255, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @_ZTV10polynomialIdE, i32 0, i32 2), i32 (...)*** %0, align 4
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE7releaseEv(%"struct.polynomial<double>"* %this) nounwind
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	ret void
}

declare void @_ZdlPv(i8*) nounwind

define linkonce void @_ZN10polynomialISt7complexIdEE7releaseEv(%"struct.polynomial<std::complex<double> >"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram693 to %0*))
	call void @llvm.dbg.stoppoint(i32 190, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	%1 = load %"struct.std::complex<double>"** %0, align 4		; <%"struct.std::complex<double>"*> [#uses=1]
	%2 = icmp eq %"struct.std::complex<double>"* %1, null		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb

bb:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 190, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	%4 = load %"struct.std::complex<double>"** %3, align 4		; <%"struct.std::complex<double>"*> [#uses=1]
	%5 = bitcast %"struct.std::complex<double>"* %4 to i8*		; <i8*> [#uses=1]
	call void @_ZdaPv(i8* %5) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram693 to %0*))
	ret void

return:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 190, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	ret void
}

define linkonce void @_ZN10polynomialISt7complexIdEED0Ev(%"struct.polynomial<std::complex<double> >"* %this) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram696 to %0*))
	call void @llvm.dbg.stoppoint(i32 255, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @_ZTV10polynomialISt7complexIdEE, i32 0, i32 2), i32 (...)*** %0, align 4
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialISt7complexIdEE7releaseEv(%"struct.polynomial<std::complex<double> >"* %this) nounwind
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = bitcast %"struct.polynomial<std::complex<double> >"* %this to i8*		; <i8*> [#uses=1]
	call void @_ZdlPv(i8* %1) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram696 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %this) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram708 to %0*))
	call void @llvm.dbg.stoppoint(i32 255, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @_ZTV10polynomialISt7complexIdEE, i32 0, i32 2), i32 (...)*** %0, align 4
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialISt7complexIdEE7releaseEv(%"struct.polynomial<std::complex<double> >"* %this) nounwind
	call void @llvm.dbg.stoppoint(i32 254, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	ret void
}

define linkonce void @_ZN10polynomialIdE7acquireEv(%"struct.polynomial<double>"* %this) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram710 to %0*))
	call void @llvm.dbg.stoppoint(i32 183, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = shl i32 %1, 3		; <i32> [#uses=1]
	%3 = call i8* @_Znaj(i32 %2)		; <i8*> [#uses=1]
	%4 = bitcast i8* %3 to double*		; <double*> [#uses=1]
	%5 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	store double* %4, double** %5, align 4
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram710 to %0*))
	ret void
}

declare i8* @_Znaj(i32)

define linkonce void @_ZN10polynomialIdEC1Ej(%"struct.polynomial<double>"* %this, i32 %degree) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram713 to %0*))
	call void @llvm.dbg.stoppoint(i32 213, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @_ZTV10polynomialIdE, i32 0, i32 2), i32 (...)*** %0, align 4
	%1 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	store double* null, double** %1, align 4
	%2 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %degree, i32* %2, align 4
	call void @llvm.dbg.stoppoint(i32 215, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE7acquireEv(%"struct.polynomial<double>"* %this)
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram713 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialIdEC1ERKS0_(%"struct.polynomial<double>"* %this, %"struct.polynomial<double>"* %source) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram717 to %0*))
	call void @llvm.dbg.stoppoint(i32 244, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @_ZTV10polynomialIdE, i32 0, i32 2), i32 (...)*** %0, align 4
	%1 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	store double* null, double** %1, align 4
	%2 = getelementptr %"struct.polynomial<double>"* %source, i32 0, i32 2		; <i32*> [#uses=1]
	%3 = load i32* %2, align 4		; <i32> [#uses=1]
	%4 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %3, i32* %4, align 4
	call void @llvm.dbg.stoppoint(i32 246, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE7acquireEv(%"struct.polynomial<double>"* %this)
	call void @llvm.dbg.stoppoint(i32 247, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%5 = getelementptr %"struct.polynomial<double>"* %source, i32 0, i32 1		; <double**> [#uses=1]
	%6 = load double** %5, align 4		; <double*> [#uses=1]
	call void @_ZN10polynomialIdE9deep_copyEPKd(%"struct.polynomial<double>"* %this, double* %6) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram717 to %0*))
	ret void
}

define linkonce %"struct.polynomial<double>"* @_ZN10polynomialIdE7stretchEj(%"struct.polynomial<double>"* %this, i32 %degrees) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram720 to %0*))
	call void @llvm.dbg.stoppoint(i32 278, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = icmp eq i32 %degrees, 0		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

bb:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 280, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%2 = load double** %1, align 4		; <double*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 281, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%4 = load i32* %3, align 4		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 283, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%5 = add i32 %4, %degrees		; <i32> [#uses=1]
	%6 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %5, i32* %6, align 4
	call void @llvm.dbg.stoppoint(i32 284, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE7acquireEv(%"struct.polynomial<double>"* %this)
	call void @llvm.dbg.stoppoint(i32 286, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb2

bb1:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 289, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%7 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%8 = load double** %7, align 4		; <double*> [#uses=1]
	%9 = getelementptr double* %2, i32 %n.0		; <double*> [#uses=1]
	%10 = load double* %9, align 1		; <double> [#uses=1]
	%11 = getelementptr double* %8, i32 %n.0		; <double*> [#uses=1]
	store double %10, double* %11, align 1
	call void @llvm.dbg.stoppoint(i32 288, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%12 = add i32 %n.0, 1		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb1, %bb
	%n.0 = phi i32 [ 0, %bb ], [ %12, %bb1 ]		; <i32> [#uses=5]
	call void @llvm.dbg.stoppoint(i32 288, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%13 = icmp ult i32 %n.0, %4		; <i1> [#uses=1]
	br i1 %13, label %bb1, label %bb5

bb4:		; preds = %bb5
	call void @llvm.dbg.stoppoint(i32 292, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%14 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 1		; <double**> [#uses=1]
	%15 = load double** %14, align 4		; <double*> [#uses=1]
	%16 = getelementptr double* %15, i32 %n.1		; <double*> [#uses=1]
	store double 0.000000e+00, double* %16, align 1
	call void @llvm.dbg.stoppoint(i32 291, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%17 = add i32 %n.1, 1		; <i32> [#uses=1]
	br label %bb5

bb5:		; preds = %bb4, %bb2
	%n.1 = phi i32 [ %17, %bb4 ], [ %n.0, %bb2 ]		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 291, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%18 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%19 = load i32* %18, align 4		; <i32> [#uses=1]
	%20 = icmp ugt i32 %19, %n.1		; <i1> [#uses=1]
	br i1 %20, label %bb4, label %return

return:		; preds = %bb5, %entry
	call void @llvm.dbg.stoppoint(i32 295, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram720 to %0*))
	ret %"struct.polynomial<double>"* %this
}

define linkonce %"struct.polynomial<double>"* @_ZN10polynomialIdEaSERKS0_(%"struct.polynomial<double>"* %this, %"struct.polynomial<double>"* %source) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram729 to %0*))
	call void @llvm.dbg.stoppoint(i32 261, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = getelementptr %"struct.polynomial<double>"* %source, i32 0, i32 2		; <i32*> [#uses=1]
	%3 = load i32* %2, align 4		; <i32> [#uses=1]
	%4 = icmp eq i32 %1, %3		; <i1> [#uses=1]
	br i1 %4, label %bb1, label %bb

bb:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 263, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE7releaseEv(%"struct.polynomial<double>"* %this) nounwind
	call void @llvm.dbg.stoppoint(i32 265, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%5 = getelementptr %"struct.polynomial<double>"* %source, i32 0, i32 2		; <i32*> [#uses=1]
	%6 = load i32* %5, align 4		; <i32> [#uses=1]
	%7 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %6, i32* %7, align 4
	call void @llvm.dbg.stoppoint(i32 266, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE7acquireEv(%"struct.polynomial<double>"* %this)
	br label %bb1

bb1:		; preds = %bb, %entry
	call void @llvm.dbg.stoppoint(i32 269, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%8 = getelementptr %"struct.polynomial<double>"* %source, i32 0, i32 1		; <double**> [#uses=1]
	%9 = load double** %8, align 4		; <double*> [#uses=1]
	call void @_ZN10polynomialIdE9deep_copyEPKd(%"struct.polynomial<double>"* %this, double* %9) nounwind
	call void @llvm.dbg.stoppoint(i32 271, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram729 to %0*))
	ret %"struct.polynomial<double>"* %this
}

define linkonce void @_ZN10polynomialISt7complexIdEE7acquireEv(%"struct.polynomial<std::complex<double> >"* %this) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram732 to %0*))
	call void @llvm.dbg.stoppoint(i32 183, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=2]
	%2 = shl i32 %1, 4		; <i32> [#uses=1]
	%3 = call i8* @_Znaj(i32 %2)		; <i8*> [#uses=1]
	%4 = bitcast i8* %3 to %"struct.std::complex<double>"*		; <%"struct.std::complex<double>"*> [#uses=2]
	br label %bb1

bb:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 183, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %.0, double 0.000000e+00, double 0.000000e+00) nounwind
	%5 = getelementptr %"struct.std::complex<double>"* %.0, i32 1		; <%"struct.std::complex<double>"*> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%.01.in = phi i32 [ %1, %entry ], [ %.01, %bb ]		; <i32> [#uses=1]
	%.0 = phi %"struct.std::complex<double>"* [ %4, %entry ], [ %5, %bb ]		; <%"struct.std::complex<double>"*> [#uses=2]
	%.01 = add i32 %.01.in, -1		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 183, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%6 = icmp eq i32 %.01, -1		; <i1> [#uses=1]
	br i1 %6, label %bb2, label %bb

bb2:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 183, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%7 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	store %"struct.std::complex<double>"* %4, %"struct.std::complex<double>"** %7, align 4
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram732 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialISt7complexIdEEC1Ej(%"struct.polynomial<std::complex<double> >"* %this, i32 %degree) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram735 to %0*))
	call void @llvm.dbg.stoppoint(i32 213, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @_ZTV10polynomialISt7complexIdEE, i32 0, i32 2), i32 (...)*** %0, align 4
	%1 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	store %"struct.std::complex<double>"* null, %"struct.std::complex<double>"** %1, align 4
	%2 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %degree, i32* %2, align 4
	call void @llvm.dbg.stoppoint(i32 215, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialISt7complexIdEE7acquireEv(%"struct.polynomial<std::complex<double> >"* %this)
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram735 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialIdE11bit_reverseERKS0_(%"struct.polynomial<std::complex<double> >"* noalias sret %agg.result, %"struct.polynomial<double>"* %poly) {
entry:
	%result = alloca %"struct.polynomial<std::complex<double> >", align 8		; <%"struct.polynomial<std::complex<double> >"*> [#uses=1]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram738 to %0*))
	%0 = bitcast %"struct.polynomial<std::complex<double> >"* %result to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable744 to %0*))
	call void @llvm.dbg.stoppoint(i32 471, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %poly) nounwind		; <i32> [#uses=1]
	%2 = call i32 @_ZN10polynomialIdE4log2Ej(i32 %1) nounwind		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 473, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %poly) nounwind		; <i32> [#uses=1]
	call void @_ZN10polynomialISt7complexIdEEC1Ej(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %3)
	call void @llvm.dbg.stoppoint(i32 475, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb6

bb:		; preds = %bb6
	call void @llvm.dbg.stoppoint(i32 476, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%4 = call double @_ZNK10polynomialIdE3getEj(%"struct.polynomial<double>"* %poly, i32 %n.0) nounwind		; <double> [#uses=1]
	%5 = call i32 @_ZN10polynomialIdE9flip_bitsEjj(i32 %n.0, i32 %2) nounwind		; <i32> [#uses=1]
	%6 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %5) nounwind		; <%"struct.std::complex<double>"*> [#uses=1]
	%7 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEaSEd(%"struct.std::complex<double>"* %6, double %4) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 475, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%8 = add i32 %n.0, 1		; <i32> [#uses=1]
	br label %bb6

bb6:		; preds = %bb, %entry
	%n.0 = phi i32 [ 0, %entry ], [ %8, %bb ]		; <i32> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 475, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%9 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %poly) nounwind		; <i32> [#uses=1]
	%10 = icmp ugt i32 %9, %n.0		; <i1> [#uses=1]
	br i1 %10, label %bb, label %return

return:		; preds = %bb6
	call void @llvm.dbg.stoppoint(i32 475, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram738 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialIdE11bit_reverseERKS_ISt7complexIdEE(%"struct.polynomial<std::complex<double> >"* noalias sret %agg.result, %"struct.polynomial<std::complex<double> >"* %poly) {
entry:
	%result = alloca %"struct.polynomial<std::complex<double> >", align 8		; <%"struct.polynomial<std::complex<double> >"*> [#uses=1]
	%memtmp7 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram747 to %0*))
	%0 = bitcast %"struct.polynomial<std::complex<double> >"* %result to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %0, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable751 to %0*))
	call void @llvm.dbg.stoppoint(i32 485, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = call i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %poly) nounwind		; <i32> [#uses=1]
	%2 = call i32 @_ZN10polynomialIdE4log2Ej(i32 %1) nounwind		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 487, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = call i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %poly) nounwind		; <i32> [#uses=1]
	call void @_ZN10polynomialISt7complexIdEEC1Ej(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %3)
	call void @llvm.dbg.stoppoint(i32 489, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb8

bb:		; preds = %bb8
	call void @llvm.dbg.stoppoint(i32 490, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%4 = call i32 @_ZN10polynomialIdE9flip_bitsEjj(i32 %n.0, i32 %2) nounwind		; <i32> [#uses=1]
	%5 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %4) nounwind		; <%"struct.std::complex<double>"*> [#uses=2]
	call void @_ZNK10polynomialISt7complexIdEE3getEj(%"struct.std::complex<double>"* noalias sret %memtmp7, %"struct.polynomial<std::complex<double> >"* %poly, i32 %n.0) nounwind
	%6 = getelementptr %"struct.std::complex<double>"* %5, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%7 = getelementptr %"struct.std::complex<double>"* %memtmp7, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%8 = load double* %7, align 8		; <double> [#uses=1]
	store double %8, double* %6, align 4
	%9 = getelementptr %"struct.std::complex<double>"* %5, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%10 = getelementptr %"struct.std::complex<double>"* %memtmp7, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%11 = load double* %10, align 8		; <double> [#uses=1]
	store double %11, double* %9, align 4
	call void @llvm.dbg.stoppoint(i32 489, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%12 = add i32 %n.0, 1		; <i32> [#uses=1]
	br label %bb8

bb8:		; preds = %bb, %entry
	%n.0 = phi i32 [ 0, %entry ], [ %12, %bb ]		; <i32> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 489, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%13 = call i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %poly) nounwind		; <i32> [#uses=1]
	%14 = icmp ugt i32 %13, %n.0		; <i1> [#uses=1]
	br i1 %14, label %bb, label %return

return:		; preds = %bb8
	call void @llvm.dbg.stoppoint(i32 489, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram747 to %0*))
	ret void
}

define linkonce %"struct.polynomial<std::complex<double> >"* @_ZN10polynomialISt7complexIdEEaSERKS2_(%"struct.polynomial<std::complex<double> >"* %this, %"struct.polynomial<std::complex<double> >"* %source) {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram753 to %0*))
	call void @llvm.dbg.stoppoint(i32 261, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = getelementptr %"struct.polynomial<std::complex<double> >"* %source, i32 0, i32 2		; <i32*> [#uses=1]
	%3 = load i32* %2, align 4		; <i32> [#uses=1]
	%4 = icmp eq i32 %1, %3		; <i1> [#uses=1]
	br i1 %4, label %bb1, label %bb

bb:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 263, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialISt7complexIdEE7releaseEv(%"struct.polynomial<std::complex<double> >"* %this) nounwind
	call void @llvm.dbg.stoppoint(i32 265, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%5 = getelementptr %"struct.polynomial<std::complex<double> >"* %source, i32 0, i32 2		; <i32*> [#uses=1]
	%6 = load i32* %5, align 4		; <i32> [#uses=1]
	%7 = getelementptr %"struct.polynomial<std::complex<double> >"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %6, i32* %7, align 4
	call void @llvm.dbg.stoppoint(i32 266, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialISt7complexIdEE7acquireEv(%"struct.polynomial<std::complex<double> >"* %this)
	br label %bb1

bb1:		; preds = %bb, %entry
	call void @llvm.dbg.stoppoint(i32 269, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%8 = getelementptr %"struct.polynomial<std::complex<double> >"* %source, i32 0, i32 1		; <%"struct.std::complex<double>"**> [#uses=1]
	%9 = load %"struct.std::complex<double>"** %8, align 4		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @_ZN10polynomialISt7complexIdEE9deep_copyEPKS1_(%"struct.polynomial<std::complex<double> >"* %this, %"struct.std::complex<double>"* %9) nounwind
	call void @llvm.dbg.stoppoint(i32 271, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram753 to %0*))
	ret %"struct.polynomial<std::complex<double> >"* %this
}

define linkonce i32 @_ZN10polynomialIdE11stretch_fftEv(%"struct.polynomial<double>"* %this) {
entry:
	%0 = alloca %"struct.std::allocator<char>", align 8		; <%"struct.std::allocator<char>"*> [#uses=4]
	%1 = alloca %"struct.std::string", align 8		; <%"struct.std::string"*> [#uses=4]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram756 to %0*))
	call void @llvm.dbg.stoppoint(i32 445, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb

bb:		; preds = %bb1, %entry
	%n.0 = phi i32 [ 1, %entry ], [ %5, %bb1 ]		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 449, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%2 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%3 = load i32* %2, align 4		; <i32> [#uses=1]
	%4 = icmp ugt i32 %3, %n.0		; <i1> [#uses=1]
	%5 = shl i32 %n.0, 1		; <i32> [#uses=4]
	br i1 %4, label %bb1, label %bb17

bb1:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 454, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%6 = icmp eq i32 %5, 0		; <i1> [#uses=1]
	br i1 %6, label %bb2, label %bb

bb2:		; preds = %bb1
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZNSaIcEC1Ev(%"struct.std::allocator<char>"* %0) nounwind
	invoke void @_ZNSsC1EPKcRKSaIcE(%"struct.std::string"* %1, i8* getelementptr ([35 x i8]* @.str759, i32 0, i32 0), %"struct.std::allocator<char>"* %0)
			to label %invcont unwind label %lpad

invcont:		; preds = %bb2
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%7 = call i8* @__cxa_allocate_exception(i32 8) nounwind		; <i8*> [#uses=3]
	%8 = bitcast i8* %7 to %"struct.std::overflow_error"*		; <%"struct.std::overflow_error"*> [#uses=1]
	invoke void @_ZNSt14overflow_errorC1ERKSs(%"struct.std::overflow_error"* %8, %"struct.std::string"* %1)
			to label %invcont3 unwind label %lpad23

invcont3:		; preds = %invcont
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZNSsD1Ev(%"struct.std::string"* %1)
			to label %bb11 unwind label %lpad31

bb11:		; preds = %invcont3
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZNSaIcED1Ev(%"struct.std::allocator<char>"* %0) nounwind
	call void @__cxa_throw(i8* %7, i8* bitcast (%struct.__si_class_type_info_pseudo* @_ZTISt14overflow_error to i8*), void (i8*)* bitcast (void (%"struct.std::overflow_error"*)* @_ZNSt14overflow_errorD1Ev to void (i8*)*)) noreturn
	unreachable

bb17:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 459, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%9 = getelementptr %"struct.polynomial<double>"* %this, i32 0, i32 2		; <i32*> [#uses=1]
	%10 = load i32* %9, align 4		; <i32> [#uses=2]
	%11 = sub i32 %5, %10		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 461, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%12 = icmp eq i32 %5, %10		; <i1> [#uses=1]
	br i1 %12, label %return, label %bb18

bb18:		; preds = %bb17
	call void @llvm.dbg.stoppoint(i32 462, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%13 = call %"struct.polynomial<double>"* @_ZN10polynomialIdE7stretchEj(%"struct.polynomial<double>"* %this, i32 %11)		; <%"struct.polynomial<double>"*> [#uses=0]
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram756 to %0*))
	ret i32 %11

return:		; preds = %bb17
	call void @llvm.dbg.stoppoint(i32 464, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	ret i32 %11

lpad:		; preds = %bb2
	%eh_ptr = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select22 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad

lpad23:		; preds = %invcont
	%eh_ptr24 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select26 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr24, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @__cxa_free_exception(i8* %7) nounwind
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZNSsD1Ev(%"struct.std::string"* %1)
			to label %ppad unwind label %lpad27

lpad27:		; preds = %lpad23
	%eh_ptr28 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select30 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr28, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad31:		; preds = %invcont3
	%eh_ptr32 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select34 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr32, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

ppad:		; preds = %lpad23, %lpad
	%eh_exception.0 = phi i8* [ %eh_ptr, %lpad ], [ %eh_ptr24, %lpad23 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 455, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZNSaIcED1Ev(%"struct.std::allocator<char>"* %0) nounwind
	call void @_Unwind_Resume(i8* %eh_exception.0)
	unreachable
}

declare void @_ZNSaIcEC1Ev(%"struct.std::allocator<char>"*) nounwind

declare void @_ZNSsC1EPKcRKSaIcE(%"struct.std::string"*, i8*, %"struct.std::allocator<char>"*)

declare i8* @__cxa_allocate_exception(i32) nounwind

declare void @_ZNSt14overflow_errorC1ERKSs(%"struct.std::overflow_error"*, %"struct.std::string"*)

declare void @_ZNSsD1Ev(%"struct.std::string"*)

declare i8* @llvm.eh.exception() nounwind

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind

declare void @__cxa_free_exception(i8*) nounwind

declare void @_ZSt9terminatev() noreturn nounwind

declare void @_ZNSaIcED1Ev(%"struct.std::allocator<char>"*) nounwind

declare void @__cxa_throw(i8*, i8*, void (i8*)*) noreturn

define linkonce void @_ZNSt14overflow_errorD1Ev(%"struct.std::overflow_error"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1727 to %0*))
	call void @llvm.dbg.stoppoint(i32 134, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*))
	%0 = getelementptr %"struct.std::overflow_error"* %this, i32 0, i32 0, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([5 x i32 (...)*]* @_ZTVSt14overflow_error, i32 0, i32 2), i32 (...)*** %0, align 4
	%1 = getelementptr %"struct.std::overflow_error"* %this, i32 0, i32 0		; <%"struct.std::runtime_error"*> [#uses=1]
	call void @_ZNSt13runtime_errorD2Ev(%"struct.std::runtime_error"* %1) nounwind
	call void @llvm.dbg.stoppoint(i32 134, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*))
	ret void
}

declare i32 @__gxx_personality_v0(...)

declare void @_Unwind_Resume(i8*)

define linkonce void @_ZNSt14overflow_errorD0Ev(%"struct.std::overflow_error"* %this) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1736 to %0*))
	call void @llvm.dbg.stoppoint(i32 134, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*))
	%0 = getelementptr %"struct.std::overflow_error"* %this, i32 0, i32 0, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	store i32 (...)** getelementptr ([5 x i32 (...)*]* @_ZTVSt14overflow_error, i32 0, i32 2), i32 (...)*** %0, align 4
	%1 = getelementptr %"struct.std::overflow_error"* %this, i32 0, i32 0		; <%"struct.std::runtime_error"*> [#uses=1]
	call void @_ZNSt13runtime_errorD2Ev(%"struct.std::runtime_error"* %1) nounwind
	call void @llvm.dbg.stoppoint(i32 134, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit770 to %0*))
	%2 = bitcast %"struct.std::overflow_error"* %this to i8*		; <i8*> [#uses=1]
	call void @_ZdlPv(i8* %2) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1736 to %0*))
	ret void
}

declare i8* @_ZNKSt13runtime_error4whatEv(%"struct.std::runtime_error"*) nounwind

declare void @_ZNSt13runtime_errorD2Ev(%"struct.std::runtime_error"*) nounwind

define linkonce void @_ZSt13__complex_expCd(%1* noalias sret %agg.result, %1 %__z) nounwind {
entry:
	%0 = alloca %1, align 8		; <%1*> [#uses=4]
	%memtmp = alloca %1, align 8		; <%1*> [#uses=3]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1742 to %0*))
	call void @llvm.dbg.stoppoint(i32 730, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	call void @cexp(%1* noalias sret %memtmp, %1 %__z) nounwind
	%1 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	%2 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%3 = load double* %2, align 8		; <double> [#uses=1]
	store double %3, double* %1, align 8
	%4 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	%5 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%6 = load double* %5, align 8		; <double> [#uses=1]
	store double %6, double* %4, align 8
	%7 = getelementptr %1* %agg.result, i32 0, i32 0		; <double*> [#uses=1]
	%8 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	%9 = load double* %8, align 8		; <double> [#uses=1]
	store double %9, double* %7, align 8
	%10 = getelementptr %1* %agg.result, i32 0, i32 1		; <double*> [#uses=1]
	%11 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	%12 = load double* %11, align 8		; <double> [#uses=1]
	store double %12, double* %10, align 8
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1742 to %0*))
	ret void
}

declare void @cexp(%1* noalias sret, %1) nounwind

define linkonce void @_ZSt3expIdESt7complexIT_ERKS2_(%"struct.std::complex<double>"* noalias sret %agg.result, %"struct.std::complex<double>"* %__z) nounwind {
entry:
	%0 = alloca %1, align 8		; <%1*> [#uses=3]
	%1 = alloca %1, align 8		; <%1*> [#uses=3]
	%memtmp = alloca %1, align 8		; <%1*> [#uses=3]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1746 to %0*))
	call void @llvm.dbg.stoppoint(i32 738, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit5 to %0*))
	%2 = call %1* @_ZNKSt7complexIdE5__repEv(%"struct.std::complex<double>"* %__z) nounwind		; <%1*> [#uses=2]
	%3 = getelementptr %1* %1, i32 0, i32 0		; <double*> [#uses=1]
	%4 = getelementptr %1* %2, i32 0, i32 0		; <double*> [#uses=1]
	%5 = load double* %4, align 8		; <double> [#uses=1]
	store double %5, double* %3, align 8
	%6 = getelementptr %1* %1, i32 0, i32 1		; <double*> [#uses=1]
	%7 = getelementptr %1* %2, i32 0, i32 1		; <double*> [#uses=1]
	%8 = load double* %7, align 8		; <double> [#uses=1]
	store double %8, double* %6, align 8
	%9 = load %1* %1, align 8		; <%1> [#uses=1]
	call void @_ZSt13__complex_expCd(%1* noalias sret %memtmp, %1 %9) nounwind
	%10 = getelementptr %1* %0, i32 0, i32 0		; <double*> [#uses=1]
	%11 = getelementptr %1* %memtmp, i32 0, i32 0		; <double*> [#uses=1]
	%12 = load double* %11, align 8		; <double> [#uses=1]
	store double %12, double* %10, align 8
	%13 = getelementptr %1* %0, i32 0, i32 1		; <double*> [#uses=1]
	%14 = getelementptr %1* %memtmp, i32 0, i32 1		; <double*> [#uses=1]
	%15 = load double* %14, align 8		; <double> [#uses=1]
	store double %15, double* %13, align 8
	%16 = load %1* %0, align 8		; <%1> [#uses=1]
	call void @_ZNSt7complexIdEC1ECd(%"struct.std::complex<double>"* %agg.result, %1 %16) nounwind
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1746 to %0*))
	ret void
}

define linkonce void @_ZN10polynomialIdE3fftERKS0_(%"struct.polynomial<std::complex<double> >"* noalias sret %agg.result, %"struct.polynomial<double>"* %poly) {
entry:
	%result = alloca %"struct.polynomial<std::complex<double> >", align 8		; <%"struct.polynomial<std::complex<double> >"*> [#uses=1]
	%u = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=6]
	%t = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=6]
	%w = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=5]
	%wm = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=5]
	%0 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=2]
	%1 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=2]
	%memtmp20 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	%memtmp23 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	%memtmp24 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	%memtmp26 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1748 to %0*))
	%2 = bitcast %"struct.polynomial<std::complex<double> >"* %result to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %2, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1751 to %0*))
	%3 = bitcast %"struct.std::complex<double>"* %u to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %3, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1753 to %0*))
	%4 = bitcast %"struct.std::complex<double>"* %t to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %4, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1755 to %0*))
	%5 = bitcast %"struct.std::complex<double>"* %w to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %5, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1757 to %0*))
	%6 = bitcast %"struct.std::complex<double>"* %wm to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %6, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1759 to %0*))
	call void @llvm.dbg.stoppoint(i32 499, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%7 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %poly) nounwind		; <i32> [#uses=1]
	%8 = call i32 @_ZN10polynomialIdE4log2Ej(i32 %7) nounwind		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 501, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %wm, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %w, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %t, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %u, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @llvm.dbg.stoppoint(i32 503, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE11bit_reverseERKS0_(%"struct.polynomial<std::complex<double> >"* noalias sret %agg.result, %"struct.polynomial<double>"* %poly)
	call void @llvm.dbg.stoppoint(i32 508, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb32

bb:		; preds = %bb32
	call void @llvm.dbg.stoppoint(i32 510, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%9 = uitofp i32 %m.0 to double		; <double> [#uses=1]
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %0, double %9, double 0.000000e+00) nounwind
	invoke void @_ZStdvIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %1, %"struct.std::complex<double>"* @_ZN10polynomialIdE4PI2IE, %"struct.std::complex<double>"* %0)
			to label %invcont unwind label %lpad

invcont:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 510, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt3expIdESt7complexIT_ERKS2_(%"struct.std::complex<double>"* noalias sret %memtmp20, %"struct.std::complex<double>"* %1) nounwind
	%10 = getelementptr %"struct.std::complex<double>"* %wm, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%11 = getelementptr %"struct.std::complex<double>"* %memtmp20, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%12 = load double* %11, align 8		; <double> [#uses=1]
	store double %12, double* %10, align 8
	%13 = getelementptr %"struct.std::complex<double>"* %wm, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%14 = getelementptr %"struct.std::complex<double>"* %memtmp20, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%15 = load double* %14, align 8		; <double> [#uses=1]
	store double %15, double* %13, align 8
	call void @llvm.dbg.stoppoint(i32 511, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%16 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEaSEd(%"struct.std::complex<double>"* %w, double 1.000000e+00) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 513, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb30

bb22:		; preds = %bb28
	call void @llvm.dbg.stoppoint(i32 517, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%17 = add i32 %k.0, %m2.0		; <i32> [#uses=1]
	%18 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %17) nounwind		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @_ZStmlIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %memtmp23, %"struct.std::complex<double>"* %w, %"struct.std::complex<double>"* %18) nounwind
	%19 = getelementptr %"struct.std::complex<double>"* %t, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%20 = getelementptr %"struct.std::complex<double>"* %memtmp23, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%21 = load double* %20, align 8		; <double> [#uses=1]
	store double %21, double* %19, align 8
	%22 = getelementptr %"struct.std::complex<double>"* %t, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%23 = getelementptr %"struct.std::complex<double>"* %memtmp23, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%24 = load double* %23, align 8		; <double> [#uses=1]
	store double %24, double* %22, align 8
	call void @llvm.dbg.stoppoint(i32 518, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%25 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %k.0) nounwind		; <%"struct.std::complex<double>"*> [#uses=2]
	%26 = getelementptr %"struct.std::complex<double>"* %u, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%27 = getelementptr %"struct.std::complex<double>"* %25, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%28 = load double* %27, align 4		; <double> [#uses=1]
	store double %28, double* %26, align 8
	%29 = getelementptr %"struct.std::complex<double>"* %u, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%30 = getelementptr %"struct.std::complex<double>"* %25, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%31 = load double* %30, align 4		; <double> [#uses=1]
	store double %31, double* %29, align 8
	call void @llvm.dbg.stoppoint(i32 519, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%32 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %k.0) nounwind		; <%"struct.std::complex<double>"*> [#uses=2]
	invoke void @_ZStplIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %memtmp24, %"struct.std::complex<double>"* %u, %"struct.std::complex<double>"* %t)
			to label %invcont25 unwind label %lpad

invcont25:		; preds = %bb22
	%33 = getelementptr %"struct.std::complex<double>"* %32, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%34 = getelementptr %"struct.std::complex<double>"* %memtmp24, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%35 = load double* %34, align 8		; <double> [#uses=1]
	store double %35, double* %33, align 4
	%36 = getelementptr %"struct.std::complex<double>"* %32, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%37 = getelementptr %"struct.std::complex<double>"* %memtmp24, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%38 = load double* %37, align 8		; <double> [#uses=1]
	store double %38, double* %36, align 4
	call void @llvm.dbg.stoppoint(i32 520, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%39 = add i32 %k.0, %m2.0		; <i32> [#uses=1]
	%40 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %39) nounwind		; <%"struct.std::complex<double>"*> [#uses=2]
	invoke void @_ZStmiIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %memtmp26, %"struct.std::complex<double>"* %u, %"struct.std::complex<double>"* %t)
			to label %invcont27 unwind label %lpad

invcont27:		; preds = %invcont25
	%41 = getelementptr %"struct.std::complex<double>"* %40, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%42 = getelementptr %"struct.std::complex<double>"* %memtmp26, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%43 = load double* %42, align 8		; <double> [#uses=1]
	store double %43, double* %41, align 4
	%44 = getelementptr %"struct.std::complex<double>"* %40, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%45 = getelementptr %"struct.std::complex<double>"* %memtmp26, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%46 = load double* %45, align 8		; <double> [#uses=1]
	store double %46, double* %44, align 4
	call void @llvm.dbg.stoppoint(i32 515, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%47 = add i32 %k.0, %m.0		; <i32> [#uses=1]
	br label %bb28

bb28:		; preds = %bb30, %invcont27
	%k.0 = phi i32 [ %47, %invcont27 ], [ %j.0, %bb30 ]		; <i32> [#uses=6]
	call void @llvm.dbg.stoppoint(i32 515, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%48 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %poly) nounwind		; <i32> [#uses=1]
	%49 = add i32 %48, -1		; <i32> [#uses=1]
	%50 = icmp ult i32 %49, %k.0		; <i1> [#uses=1]
	br i1 %50, label %bb29, label %bb22

bb29:		; preds = %bb28
	call void @llvm.dbg.stoppoint(i32 523, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%51 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEmLIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %w, %"struct.std::complex<double>"* %wm) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 513, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%52 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %bb30

bb30:		; preds = %bb29, %invcont
	%j.0 = phi i32 [ 0, %invcont ], [ %52, %bb29 ]		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 513, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%53 = add i32 %m2.0, -1		; <i32> [#uses=1]
	%54 = icmp ult i32 %53, %j.0		; <i1> [#uses=1]
	br i1 %54, label %bb31, label %bb28

bb31:		; preds = %bb30
	call void @llvm.dbg.stoppoint(i32 526, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%55 = shl i32 %m.0, 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 527, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%56 = shl i32 %m2.0, 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 508, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%57 = add i32 %s.0, 1		; <i32> [#uses=1]
	br label %bb32

bb32:		; preds = %bb31, %entry
	%m.0 = phi i32 [ 2, %entry ], [ %55, %bb31 ]		; <i32> [#uses=3]
	%m2.0 = phi i32 [ 1, %entry ], [ %56, %bb31 ]		; <i32> [#uses=4]
	%s.0 = phi i32 [ 0, %entry ], [ %57, %bb31 ]		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 508, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%58 = icmp ult i32 %s.0, %8		; <i1> [#uses=1]
	br i1 %58, label %bb, label %return

return:		; preds = %bb32
	call void @llvm.dbg.stoppoint(i32 530, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1748 to %0*))
	ret void

lpad:		; preds = %invcont25, %bb22, %bb
	%eh_ptr = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select40 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 530, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %agg.result)
			to label %Unwind unwind label %lpad41

lpad41:		; preds = %lpad
	%eh_ptr42 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select44 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr42, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 530, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

Unwind:		; preds = %lpad
	call void @_Unwind_Resume(i8* %eh_ptr)
	unreachable
}

define linkonce void @_ZN10polynomialIdE11inverse_fftERKS_ISt7complexIdEE(%"struct.polynomial<std::complex<double> >"* noalias sret %agg.result, %"struct.polynomial<std::complex<double> >"* %poly) {
entry:
	%result = alloca %"struct.polynomial<std::complex<double> >", align 8		; <%"struct.polynomial<std::complex<double> >"*> [#uses=1]
	%u = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=6]
	%t = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=6]
	%w = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=5]
	%wm = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=5]
	%0 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=2]
	%1 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=2]
	%2 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=2]
	%memtmp22 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	%memtmp25 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	%memtmp26 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	%memtmp28 = alloca %"struct.std::complex<double>", align 8		; <%"struct.std::complex<double>"*> [#uses=3]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1771 to %0*))
	%3 = bitcast %"struct.polynomial<std::complex<double> >"* %result to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %3, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1774 to %0*))
	%4 = bitcast %"struct.std::complex<double>"* %u to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %4, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1775 to %0*))
	%5 = bitcast %"struct.std::complex<double>"* %t to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %5, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1776 to %0*))
	%6 = bitcast %"struct.std::complex<double>"* %w to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %6, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1777 to %0*))
	%7 = bitcast %"struct.std::complex<double>"* %wm to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %7, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1778 to %0*))
	call void @llvm.dbg.stoppoint(i32 537, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%8 = call i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %poly) nounwind		; <i32> [#uses=1]
	%9 = call i32 @_ZN10polynomialIdE4log2Ej(i32 %8) nounwind		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 539, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %wm, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %w, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %t, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %u, double 0.000000e+00, double 0.000000e+00) nounwind
	call void @llvm.dbg.stoppoint(i32 541, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdE11bit_reverseERKS_ISt7complexIdEE(%"struct.polynomial<std::complex<double> >"* noalias sret %agg.result, %"struct.polynomial<std::complex<double> >"* %poly)
	call void @llvm.dbg.stoppoint(i32 546, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb34

bb:		; preds = %bb34
	call void @llvm.dbg.stoppoint(i32 548, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%10 = uitofp i32 %m.0 to double		; <double> [#uses=1]
	call void @_ZNSt7complexIdEC1Edd(%"struct.std::complex<double>"* %1, double %10, double 0.000000e+00) nounwind
	call void @_ZStngIdESt7complexIT_ERKS2_(%"struct.std::complex<double>"* noalias sret %0, %"struct.std::complex<double>"* @_ZN10polynomialIdE4PI2IE) nounwind
	invoke void @_ZStdvIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %2, %"struct.std::complex<double>"* %0, %"struct.std::complex<double>"* %1)
			to label %invcont unwind label %lpad

invcont:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 548, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt3expIdESt7complexIT_ERKS2_(%"struct.std::complex<double>"* noalias sret %memtmp22, %"struct.std::complex<double>"* %2) nounwind
	%11 = getelementptr %"struct.std::complex<double>"* %wm, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%12 = getelementptr %"struct.std::complex<double>"* %memtmp22, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%13 = load double* %12, align 8		; <double> [#uses=1]
	store double %13, double* %11, align 8
	%14 = getelementptr %"struct.std::complex<double>"* %wm, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%15 = getelementptr %"struct.std::complex<double>"* %memtmp22, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%16 = load double* %15, align 8		; <double> [#uses=1]
	store double %16, double* %14, align 8
	call void @llvm.dbg.stoppoint(i32 549, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%17 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEaSEd(%"struct.std::complex<double>"* %w, double 1.000000e+00) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 551, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb32

bb24:		; preds = %bb30
	call void @llvm.dbg.stoppoint(i32 555, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%18 = add i32 %k.0, %m2.0		; <i32> [#uses=1]
	%19 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %18) nounwind		; <%"struct.std::complex<double>"*> [#uses=1]
	call void @_ZStmlIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %memtmp25, %"struct.std::complex<double>"* %w, %"struct.std::complex<double>"* %19) nounwind
	%20 = getelementptr %"struct.std::complex<double>"* %t, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%21 = getelementptr %"struct.std::complex<double>"* %memtmp25, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%22 = load double* %21, align 8		; <double> [#uses=1]
	store double %22, double* %20, align 8
	%23 = getelementptr %"struct.std::complex<double>"* %t, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%24 = getelementptr %"struct.std::complex<double>"* %memtmp25, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%25 = load double* %24, align 8		; <double> [#uses=1]
	store double %25, double* %23, align 8
	call void @llvm.dbg.stoppoint(i32 556, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%26 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %k.0) nounwind		; <%"struct.std::complex<double>"*> [#uses=2]
	%27 = getelementptr %"struct.std::complex<double>"* %u, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%28 = getelementptr %"struct.std::complex<double>"* %26, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%29 = load double* %28, align 4		; <double> [#uses=1]
	store double %29, double* %27, align 8
	%30 = getelementptr %"struct.std::complex<double>"* %u, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%31 = getelementptr %"struct.std::complex<double>"* %26, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%32 = load double* %31, align 4		; <double> [#uses=1]
	store double %32, double* %30, align 8
	call void @llvm.dbg.stoppoint(i32 557, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%33 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %k.0) nounwind		; <%"struct.std::complex<double>"*> [#uses=2]
	invoke void @_ZStplIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %memtmp26, %"struct.std::complex<double>"* %u, %"struct.std::complex<double>"* %t)
			to label %invcont27 unwind label %lpad

invcont27:		; preds = %bb24
	%34 = getelementptr %"struct.std::complex<double>"* %33, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%35 = getelementptr %"struct.std::complex<double>"* %memtmp26, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%36 = load double* %35, align 8		; <double> [#uses=1]
	store double %36, double* %34, align 4
	%37 = getelementptr %"struct.std::complex<double>"* %33, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%38 = getelementptr %"struct.std::complex<double>"* %memtmp26, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%39 = load double* %38, align 8		; <double> [#uses=1]
	store double %39, double* %37, align 4
	call void @llvm.dbg.stoppoint(i32 558, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%40 = add i32 %k.0, %m2.0		; <i32> [#uses=1]
	%41 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %40) nounwind		; <%"struct.std::complex<double>"*> [#uses=2]
	invoke void @_ZStmiIdESt7complexIT_ERKS2_S4_(%"struct.std::complex<double>"* noalias sret %memtmp28, %"struct.std::complex<double>"* %u, %"struct.std::complex<double>"* %t)
			to label %invcont29 unwind label %lpad

invcont29:		; preds = %invcont27
	%42 = getelementptr %"struct.std::complex<double>"* %41, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%43 = getelementptr %"struct.std::complex<double>"* %memtmp28, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%44 = load double* %43, align 8		; <double> [#uses=1]
	store double %44, double* %42, align 4
	%45 = getelementptr %"struct.std::complex<double>"* %41, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%46 = getelementptr %"struct.std::complex<double>"* %memtmp28, i32 0, i32 0, i32 1		; <double*> [#uses=1]
	%47 = load double* %46, align 8		; <double> [#uses=1]
	store double %47, double* %45, align 4
	call void @llvm.dbg.stoppoint(i32 553, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%48 = add i32 %k.0, %m.0		; <i32> [#uses=1]
	br label %bb30

bb30:		; preds = %bb32, %invcont29
	%k.0 = phi i32 [ %48, %invcont29 ], [ %j.0, %bb32 ]		; <i32> [#uses=6]
	call void @llvm.dbg.stoppoint(i32 553, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%49 = call i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %poly) nounwind		; <i32> [#uses=1]
	%50 = add i32 %49, -1		; <i32> [#uses=1]
	%51 = icmp ult i32 %50, %k.0		; <i1> [#uses=1]
	br i1 %51, label %bb31, label %bb24

bb31:		; preds = %bb30
	call void @llvm.dbg.stoppoint(i32 561, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%52 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEmLIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %w, %"struct.std::complex<double>"* %wm) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 551, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%53 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %bb32

bb32:		; preds = %bb31, %invcont
	%j.0 = phi i32 [ 0, %invcont ], [ %53, %bb31 ]		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 551, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%54 = add i32 %m2.0, -1		; <i32> [#uses=1]
	%55 = icmp ult i32 %54, %j.0		; <i1> [#uses=1]
	br i1 %55, label %bb33, label %bb30

bb33:		; preds = %bb32
	call void @llvm.dbg.stoppoint(i32 564, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%56 = shl i32 %m.0, 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 565, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%57 = shl i32 %m2.0, 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 546, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%58 = add i32 %s.0, 1		; <i32> [#uses=1]
	br label %bb34

bb34:		; preds = %bb33, %entry
	%m.0 = phi i32 [ 2, %entry ], [ %56, %bb33 ]		; <i32> [#uses=3]
	%m2.0 = phi i32 [ 1, %entry ], [ %57, %bb33 ]		; <i32> [#uses=4]
	%s.0 = phi i32 [ 0, %entry ], [ %58, %bb33 ]		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 546, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%59 = icmp ult i32 %s.0, %9		; <i1> [#uses=1]
	br i1 %59, label %bb, label %bb37

bb36:		; preds = %bb37
	call void @llvm.dbg.stoppoint(i32 569, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%60 = call i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %poly) nounwind		; <i32> [#uses=1]
	%61 = uitofp i32 %60 to double		; <double> [#uses=1]
	%62 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %agg.result, i32 %j.1) nounwind		; <%"struct.std::complex<double>"*> [#uses=1]
	%63 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEdVEd(%"struct.std::complex<double>"* %62, double %61) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 568, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%64 = add i32 %j.1, 1		; <i32> [#uses=1]
	br label %bb37

bb37:		; preds = %bb36, %bb34
	%j.1 = phi i32 [ %64, %bb36 ], [ 0, %bb34 ]		; <i32> [#uses=3]
	call void @llvm.dbg.stoppoint(i32 568, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%65 = call i32 @_ZNK10polynomialISt7complexIdEE6degreeEv(%"struct.polynomial<std::complex<double> >"* %poly) nounwind		; <i32> [#uses=1]
	%66 = icmp ugt i32 %65, %j.1		; <i1> [#uses=1]
	br i1 %66, label %bb36, label %return

return:		; preds = %bb37
	call void @llvm.dbg.stoppoint(i32 571, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1771 to %0*))
	ret void

lpad:		; preds = %invcont27, %bb24, %bb
	%eh_ptr = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select46 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 571, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %agg.result)
			to label %Unwind unwind label %lpad47

lpad47:		; preds = %lpad
	%eh_ptr48 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select50 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr48, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 571, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

Unwind:		; preds = %lpad
	call void @_Unwind_Resume(i8* %eh_ptr)
	unreachable
}

define linkonce void @_ZNK10polynomialIdEmlERKS0_(%"struct.polynomial<double>"* noalias sret %agg.result, %"struct.polynomial<double>"* %this, %"struct.polynomial<double>"* %poly) {
entry:
	%result = alloca %"struct.polynomial<double>", align 8		; <%"struct.polynomial<double>"*> [#uses=1]
	%dft2 = alloca %"struct.polynomial<std::complex<double> >", align 8		; <%"struct.polynomial<std::complex<double> >"*> [#uses=7]
	%dft1 = alloca %"struct.polynomial<std::complex<double> >", align 8		; <%"struct.polynomial<std::complex<double> >"*> [#uses=6]
	%a2 = alloca %"struct.polynomial<double>", align 8		; <%"struct.polynomial<double>"*> [#uses=8]
	%a1 = alloca %"struct.polynomial<double>", align 8		; <%"struct.polynomial<double>"*> [#uses=9]
	%0 = alloca %"struct.polynomial<std::complex<double> >", align 8		; <%"struct.polynomial<std::complex<double> >"*> [#uses=4]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1785 to %0*))
	%1 = bitcast %"struct.polynomial<double>"* %result to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %1, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1791 to %0*))
	%2 = bitcast %"struct.polynomial<std::complex<double> >"* %dft2 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %2, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1795 to %0*))
	%3 = bitcast %"struct.polynomial<std::complex<double> >"* %dft1 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %3, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1797 to %0*))
	%4 = bitcast %"struct.polynomial<double>"* %a2 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %4, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1799 to %0*))
	%5 = bitcast %"struct.polynomial<double>"* %a1 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %5, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable1801 to %0*))
	call void @llvm.dbg.stoppoint(i32 590, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdEC1ERKS0_(%"struct.polynomial<double>"* %a1, %"struct.polynomial<double>"* %this)
	call void @llvm.dbg.stoppoint(i32 591, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdEC1ERKS0_(%"struct.polynomial<double>"* %a2, %"struct.polynomial<double>"* %poly)
			to label %invcont unwind label %lpad

invcont:		; preds = %entry
	call void @llvm.dbg.stoppoint(i32 594, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%6 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %a1) nounwind		; <i32> [#uses=1]
	%7 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %a2) nounwind		; <i32> [#uses=1]
	%8 = icmp ugt i32 %6, %7		; <i1> [#uses=1]
	br i1 %8, label %bb, label %bb26

bb:		; preds = %invcont
	call void @llvm.dbg.stoppoint(i32 595, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%9 = invoke i32 @_ZN10polynomialIdE11stretch_fftEv(%"struct.polynomial<double>"* %a1)
			to label %invcont24 unwind label %lpad76		; <i32> [#uses=1]

invcont24:		; preds = %bb
	call void @llvm.dbg.stoppoint(i32 595, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%10 = invoke %"struct.polynomial<double>"* @_ZN10polynomialIdE7stretchEj(%"struct.polynomial<double>"* %a2, i32 %9)
			to label %bb29 unwind label %lpad76		; <%"struct.polynomial<double>"*> [#uses=0]

bb26:		; preds = %invcont
	call void @llvm.dbg.stoppoint(i32 597, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%11 = invoke i32 @_ZN10polynomialIdE11stretch_fftEv(%"struct.polynomial<double>"* %a2)
			to label %invcont27 unwind label %lpad76		; <i32> [#uses=1]

invcont27:		; preds = %bb26
	call void @llvm.dbg.stoppoint(i32 597, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%12 = invoke %"struct.polynomial<double>"* @_ZN10polynomialIdE7stretchEj(%"struct.polynomial<double>"* %a1, i32 %11)
			to label %bb29 unwind label %lpad76		; <%"struct.polynomial<double>"*> [#uses=0]

bb29:		; preds = %invcont27, %invcont24
	call void @llvm.dbg.stoppoint(i32 600, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdE3fftERKS0_(%"struct.polynomial<std::complex<double> >"* noalias sret %dft1, %"struct.polynomial<double>"* %a1)
			to label %invcont30 unwind label %lpad76

invcont30:		; preds = %bb29
	call void @llvm.dbg.stoppoint(i32 601, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdE3fftERKS0_(%"struct.polynomial<std::complex<double> >"* noalias sret %dft2, %"struct.polynomial<double>"* %a2)
			to label %invcont31 unwind label %lpad80

invcont31:		; preds = %invcont30
	call void @llvm.dbg.stoppoint(i32 604, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%13 = call i32 @_ZNK10polynomialIdE6degreeEv(%"struct.polynomial<double>"* %a1) nounwind		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 606, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br label %bb33

bb32:		; preds = %bb33
	call void @llvm.dbg.stoppoint(i32 607, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%14 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %dft2, i32 %k15.0) nounwind		; <%"struct.std::complex<double>"*> [#uses=1]
	%15 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %dft1, i32 %k15.0) nounwind		; <%"struct.std::complex<double>"*> [#uses=1]
	%16 = call %"struct.std::complex<double>"* @_ZNSt7complexIdEmLIdEERS0_RKS_IT_E(%"struct.std::complex<double>"* %15, %"struct.std::complex<double>"* %14) nounwind		; <%"struct.std::complex<double>"*> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 606, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%17 = add i32 %k15.0, 1		; <i32> [#uses=1]
	br label %bb33

bb33:		; preds = %bb32, %invcont31
	%k15.0 = phi i32 [ 0, %invcont31 ], [ %17, %bb32 ]		; <i32> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 606, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%18 = icmp ult i32 %k15.0, %13		; <i1> [#uses=1]
	br i1 %18, label %bb32, label %bb34

bb34:		; preds = %bb33
	call void @llvm.dbg.stoppoint(i32 610, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdE11inverse_fftERKS_ISt7complexIdEE(%"struct.polynomial<std::complex<double> >"* noalias sret %0, %"struct.polynomial<std::complex<double> >"* %dft1)
			to label %invcont35 unwind label %lpad84

invcont35:		; preds = %bb34
	call void @llvm.dbg.stoppoint(i32 610, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%19 = invoke %"struct.polynomial<std::complex<double> >"* @_ZN10polynomialISt7complexIdEEaSERKS2_(%"struct.polynomial<std::complex<double> >"* %dft2, %"struct.polynomial<std::complex<double> >"* %0)
			to label %invcont36 unwind label %lpad88		; <%"struct.polynomial<std::complex<double> >"*> [#uses=0]

invcont36:		; preds = %invcont35
	call void @llvm.dbg.stoppoint(i32 610, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %0)
			to label %bb43 unwind label %lpad84

bb43:		; preds = %invcont36
	call void @llvm.dbg.stoppoint(i32 613, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%20 = add i32 %13, -1		; <i32> [#uses=2]
	call void @llvm.dbg.stoppoint(i32 614, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdEC1Ej(%"struct.polynomial<double>"* %agg.result, i32 %20)
			to label %bb46 unwind label %lpad84

bb45:		; preds = %bb46
	call void @llvm.dbg.stoppoint(i32 617, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%21 = call double* @_ZN10polynomialIdEixEj(%"struct.polynomial<double>"* %agg.result, i32 %k.0) nounwind		; <double*> [#uses=1]
	%22 = call %"struct.std::complex<double>"* @_ZN10polynomialISt7complexIdEEixEj(%"struct.polynomial<std::complex<double> >"* %dft2, i32 %k.0) nounwind		; <%"struct.std::complex<double>"*> [#uses=1]
	%23 = call double* @_ZNSt7complexIdE4realEv(%"struct.std::complex<double>"* %22) nounwind		; <double*> [#uses=1]
	%24 = load double* %23, align 8		; <double> [#uses=1]
	store double %24, double* %21, align 8
	call void @llvm.dbg.stoppoint(i32 616, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%25 = add i32 %k.0, 1		; <i32> [#uses=1]
	br label %bb46

bb46:		; preds = %bb45, %bb43
	%k.0 = phi i32 [ %25, %bb45 ], [ 0, %bb43 ]		; <i32> [#uses=4]
	call void @llvm.dbg.stoppoint(i32 616, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%26 = icmp ult i32 %k.0, %20		; <i1> [#uses=1]
	br i1 %26, label %bb45, label %bb47

bb47:		; preds = %bb46
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %dft2)
			to label %bb54 unwind label %lpad80

bb54:		; preds = %bb47
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %dft1)
			to label %bb61 unwind label %lpad76

bb61:		; preds = %bb54
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %a2)
			to label %bb68 unwind label %lpad

bb68:		; preds = %bb61
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %a1)
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram1785 to %0*))
	ret void

lpad:		; preds = %bb61, %entry
	%eh_ptr = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select75 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad

lpad76:		; preds = %bb54, %bb29, %invcont27, %bb26, %invcont24, %bb
	%eh_ptr77 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select79 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr77, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad112

lpad80:		; preds = %bb47, %invcont30
	%eh_ptr81 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select83 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr81, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad113

lpad84:		; preds = %bb43, %invcont36, %bb34
	%eh_ptr85 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select87 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr85, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	br label %ppad114

lpad88:		; preds = %invcont35
	%eh_ptr89 = call i8* @llvm.eh.exception()		; <i8*> [#uses=2]
	%eh_select91 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr89, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 610, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %0)
			to label %ppad114 unwind label %lpad92

lpad92:		; preds = %lpad88
	%eh_ptr93 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select95 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr93, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 610, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad96:		; preds = %ppad114
	%eh_ptr97 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select99 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr97, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad100:		; preds = %ppad113
	%eh_ptr101 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select103 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr101, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad104:		; preds = %ppad112
	%eh_ptr105 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select107 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr105, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

lpad108:		; preds = %ppad
	%eh_ptr109 = call i8* @llvm.eh.exception()		; <i8*> [#uses=1]
	%eh_select111 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr109, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1)		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @_ZSt9terminatev() noreturn nounwind
	unreachable

ppad:		; preds = %ppad112, %lpad
	%eh_exception.3 = phi i8* [ %eh_ptr, %lpad ], [ %eh_exception.2, %ppad112 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %a1)
			to label %Unwind unwind label %lpad108

ppad112:		; preds = %ppad113, %lpad76
	%eh_exception.2 = phi i8* [ %eh_ptr77, %lpad76 ], [ %eh_exception.1, %ppad113 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialIdED1Ev(%"struct.polynomial<double>"* %a2)
			to label %ppad unwind label %lpad104

ppad113:		; preds = %ppad114, %lpad80
	%eh_exception.1 = phi i8* [ %eh_ptr81, %lpad80 ], [ %eh_exception.0, %ppad114 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %dft1)
			to label %ppad112 unwind label %lpad100

ppad114:		; preds = %lpad88, %lpad84
	%eh_exception.0 = phi i8* [ %eh_ptr85, %lpad84 ], [ %eh_ptr89, %lpad88 ]		; <i8*> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 620, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	invoke void @_ZN10polynomialISt7complexIdEED1Ev(%"struct.polynomial<std::complex<double> >"* %dft2)
			to label %ppad113 unwind label %lpad96

Unwind:		; preds = %ppad
	call void @_Unwind_Resume(i8* %eh_exception.3)
	unreachable
}

declare i32 @strcmp(i8* nocapture, i8* nocapture) nounwind readonly

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEd(%"struct.std::basic_ostream<char,std::char_traits<char> >"*, double)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8*)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSo5flushEv(%"struct.std::basic_ostream<char,std::char_traits<char> >"*)

declare extern_weak i32 @pthread_once(i32*, void ()*)

declare extern_weak i8* @pthread_getspecific(i32)

declare extern_weak i32 @pthread_setspecific(i32, i8*)

declare extern_weak i32 @pthread_create(i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)

declare extern_weak i32 @pthread_cancel(i32)

declare extern_weak i32 @pthread_mutex_lock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_trylock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_unlock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_init(%struct.pthread_mutex_t*, %struct..0._50*)

declare extern_weak i32 @pthread_key_create(i32*, void (i8*)*)

declare extern_weak i32 @pthread_key_delete(i32)

declare extern_weak i32 @pthread_mutexattr_init(%struct..0._50*)

declare extern_weak i32 @pthread_mutexattr_settype(%struct..0._50*, i32)

declare extern_weak i32 @pthread_mutexattr_destroy(%struct..0._50*)

declare i32 @memcmp(i8* nocapture, i8* nocapture, i32) nounwind readonly
