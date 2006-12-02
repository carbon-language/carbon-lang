; RUN: llvm-upgrade < %s | llvm-as | opt -inline -prune-eh -disable-output
; PR993
target endian = little
target pointersize = 32
target triple = "i386-unknown-openbsd3.9"
deplibs = [ "stdc++", "c", "crtend" ]
	"struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" = type { sbyte* }
	"struct.__gnu_cxx::char_producer<char>" = type { int (...)** }
	%struct.__sFILE = type { ubyte*, int, int, short, short, %struct.__sbuf, int, sbyte*, int (sbyte*)*, int (sbyte*, sbyte*, int)*, long (sbyte*, long, int)*, int (sbyte*, sbyte*, int)*, %struct.__sbuf, ubyte*, int, [3 x ubyte], [1 x ubyte], %struct.__sbuf, int, long }
	%struct.__sbuf = type { ubyte*, int }
	"struct.std::__basic_file<char>" = type { %struct.__sFILE*, bool }
	"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>" = type { "struct.std::locale::facet" }
	"struct.std::bad_alloc" = type { "struct.__gnu_cxx::char_producer<char>" }
	"struct.std::basic_filebuf<char,std::char_traits<char> >" = type { "struct.std::basic_streambuf<char,std::char_traits<char> >", int, "struct.std::__basic_file<char>", uint, %union.__mbstate_t, %union.__mbstate_t, sbyte*, uint, bool, bool, bool, bool, sbyte, sbyte*, sbyte*, bool, "struct.std::codecvt<char,char,__mbstate_t>"*, sbyte*, int, sbyte*, sbyte* }
	"struct.std::basic_ios<char,std::char_traits<char> >" = type { "struct.std::ios_base", "struct.std::basic_ostream<char,std::char_traits<char> >"*, sbyte, bool, "struct.std::basic_streambuf<char,std::char_traits<char> >"*, "struct.std::ctype<char>"*, "struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"*, "struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"* }
	"struct.std::basic_iostream<char,std::char_traits<char> >" = type { "struct.std::locale::facet", "struct.__gnu_cxx::char_producer<char>", "struct.std::basic_ios<char,std::char_traits<char> >" }
	"struct.std::basic_ofstream<char,std::char_traits<char> >" = type { "struct.__gnu_cxx::char_producer<char>", "struct.std::basic_filebuf<char,std::char_traits<char> >", "struct.std::basic_ios<char,std::char_traits<char> >" }
	"struct.std::basic_ostream<char,std::char_traits<char> >" = type { int (...)**, "struct.std::basic_ios<char,std::char_traits<char> >" }
	"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { int (...)**, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, "struct.std::locale" }
	"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" = type { "struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" }
	"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep" = type { "struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" }
	"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" = type { uint, uint, int }
	"struct.std::codecvt<char,char,__mbstate_t>" = type { "struct.std::__codecvt_abstract_base<char,char,__mbstate_t>", int* }
	"struct.std::ctype<char>" = type { "struct.std::__codecvt_abstract_base<char,char,__mbstate_t>", int*, bool, int*, int*, uint* }
	"struct.std::domain_error" = type { "struct.std::logic_error" }
	"struct.std::ios_base" = type { int (...)**, int, int, uint, uint, uint, "struct.std::ios_base::_Callback_list"*, "struct.std::ios_base::_Words", [8 x "struct.std::ios_base::_Words"], int, "struct.std::ios_base::_Words"*, "struct.std::locale" }
	"struct.std::ios_base::_Callback_list" = type { "struct.std::ios_base::_Callback_list"*, void (uint, "struct.std::ios_base"*, int)*, int, int }
	"struct.std::ios_base::_Words" = type { sbyte*, int }
	"struct.std::locale" = type { "struct.std::locale::_Impl"* }
	"struct.std::locale::_Impl" = type { int, "struct.std::locale::facet"**, uint, "struct.std::locale::facet"**, sbyte** }
	"struct.std::locale::facet" = type { int (...)**, int }
	"struct.std::logic_error" = type { "struct.__gnu_cxx::char_producer<char>", "struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" }
	%union.__mbstate_t = type { long, [120 x ubyte] }
%.str_1 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_9 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]

implementation   ; Functions:

void %main() {
entry:
	call fastcc void %_ZNSt14basic_ofstreamIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode( )
	ret void
}

fastcc void %_ZNSt14basic_ofstreamIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode() {
entry:
	%tmp.6 = seteq "struct.std::basic_filebuf<char,std::char_traits<char> >"* null, null		; <bool> [#uses=1]
	br bool %tmp.6, label %then, label %UnifiedReturnBlock

then:		; preds = %entry
	tail call fastcc void %_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate( )
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

fastcc void %_ZN10__cxxabiv111__terminateEPFvvE() {
entry:
	unreachable
}

void %_ZNSdD0Ev() {
entry:
	unreachable
}

void %_ZThn8_NSdD1Ev() {
entry:
	ret void
}

void %_ZNSt13basic_filebufIcSt11char_traitsIcEED0Ev() {
entry:
	ret void
}

void %_ZNSt13basic_filebufIcSt11char_traitsIcEE9pbackfailEi() {
entry:
	unreachable
}

fastcc void %_ZNSoD2Ev() {
entry:
	unreachable
}

fastcc void %_ZNSt9basic_iosIcSt11char_traitsIcEED2Ev() {
entry:
	unreachable
}

fastcc void %_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate() {
entry:
	tail call fastcc void %_ZSt19__throw_ios_failurePKc( )
	ret void
}

declare fastcc void %_ZNSaIcED1Ev()

fastcc void %_ZNSsC1EPKcRKSaIcE() {
entry:
	tail call fastcc void %_ZNSs16_S_construct_auxIPKcEEPcT_S3_RKSaIcE12__false_type( )
	unreachable
}

fastcc void %_ZSt14__convert_to_vIyEvPKcRT_RSt12_Ios_IostateRKPii() {
entry:
	ret void
}

fastcc void %_ZNSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEEC1Ej() {
entry:
	ret void
}

fastcc void %_ZSt19__throw_ios_failurePKc() {
entry:
	call fastcc void %_ZNSsC1EPKcRKSaIcE( )
	unwind
}

void %_GLOBAL__D__ZSt23lexicographical_compareIPKaS1_EbT_S2_T0_S3_() {
entry:
	ret void
}

void %_ZNSt9bad_allocD1Ev() {
entry:
	unreachable
}

fastcc void %_ZSt19__throw_logic_errorPKc() {
entry:
	invoke fastcc void %_ZNSt11logic_errorC1ERKSs( )
			to label %try_exit.0 unwind label %try_catch.0

try_catch.0:		; preds = %entry
	unreachable

try_exit.0:		; preds = %entry
	unwind
}

fastcc void %_ZNSt11logic_errorC1ERKSs() {
entry:
	call fastcc void %_ZNSsC1ERKSs( )
	ret void
}

void %_ZNSt12domain_errorD1Ev() {
entry:
	unreachable
}

fastcc void %_ZSt20__throw_length_errorPKc() {
entry:
	call fastcc void %_ZNSt12length_errorC1ERKSs( )
	unwind
}

fastcc void %_ZNSt12length_errorC1ERKSs() {
entry:
	invoke fastcc void %_ZNSsC1ERKSs( )
			to label %_ZNSt11logic_errorC2ERKSs.exit unwind label %invoke_catch.i

invoke_catch.i:		; preds = %entry
	unwind

_ZNSt11logic_errorC2ERKSs.exit:		; preds = %entry
	ret void
}

fastcc void %_ZNSs4_Rep9_S_createEjRKSaIcE() {
entry:
	call fastcc void %_ZSt20__throw_length_errorPKc( )
	unreachable
}

fastcc void %_ZNSs12_S_constructIN9__gnu_cxx17__normal_iteratorIPcSsEEEES2_T_S4_RKSaIcESt20forward_iterator_tag() {
entry:
	unreachable
}

fastcc void %_ZNSs16_S_construct_auxIPKcEEPcT_S3_RKSaIcE12__false_type() {
entry:
	br bool false, label %then.1.i, label %endif.1.i

then.1.i:		; preds = %entry
	call fastcc void %_ZSt19__throw_logic_errorPKc( )
	br label %endif.1.i

endif.1.i:		; preds = %then.1.i, %entry
	call fastcc void %_ZNSs4_Rep9_S_createEjRKSaIcE( )
	unreachable
}

fastcc void %_ZNSsC1ERKSs() {
entry:
	call fastcc void %_ZNSs4_Rep7_M_grabERKSaIcES2_( )
	invoke fastcc void %_ZNSaIcEC1ERKS_( )
			to label %invoke_cont.1 unwind label %invoke_catch.1

invoke_catch.1:		; preds = %entry
	call fastcc void %_ZNSaIcED1Ev( )
	unwind

invoke_cont.1:		; preds = %entry
	call fastcc void %_ZNSaIcEC2ERKS_( )
	ret void
}

fastcc void %_ZNSaIcEC1ERKS_() {
entry:
	ret void
}

fastcc void %_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_jc() {
entry:
	ret void
}

fastcc void %_ZNSs4_Rep7_M_grabERKSaIcES2_() {
entry:
	br bool false, label %else.i, label %cond_true

cond_true:		; preds = %entry
	ret void

else.i:		; preds = %entry
	tail call fastcc void %_ZNSs4_Rep9_S_createEjRKSaIcE( )
	unreachable
}

fastcc void %_ZNSaIcEC2ERKS_() {
entry:
	ret void
}

fastcc void %_ZN9__gnu_cxx12__pool_allocILb1ELi0EE8allocateEj() {
entry:
	ret void
}

fastcc void %_ZN9__gnu_cxx12__pool_allocILb1ELi0EE9_S_refillEj() {
entry:
	unreachable
}
