; RUN: llvm-as < %s | opt -inline -prune-eh -disable-output

 	"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>" = type { "struct.std::locale::facet" }
	"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >" = type { int (...)**, int*, int*, int*, int*, int*, int*, "struct.std::locale" }
	"struct.std::ios_base" = type { int (...)**, int, int, uint, uint, uint, "struct.std::ios_base::_Callback_list"*, "struct.std::ios_base::_Words", [8 x "struct.std::ios_base::_Words"], int, "struct.std::ios_base::_Words"*, "struct.std::locale" }
	"struct.std::ios_base::_Callback_list" = type { "struct.std::ios_base::_Callback_list"*, void (uint, "struct.std::ios_base"*, int)*, int, int }
	"struct.std::ios_base::_Words" = type { sbyte*, int }
	"struct.std::locale" = type { "struct.std::locale::_Impl"* }
	"struct.std::locale::_Impl" = type { int, "struct.std::locale::facet"**, uint, "struct.std::locale::facet"**, sbyte** }
	"struct.std::locale::facet" = type { int (...)**, int }
	"struct.std::ostreambuf_iterator<wchar_t,std::char_traits<wchar_t> >" = type { "struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >"*, int }

implementation   ; Functions:

void %_ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE6do_putES3_RSt8ios_basewl("struct.std::ostreambuf_iterator<wchar_t,std::char_traits<wchar_t> >"* %agg.result, "struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"* %this, "struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >"* %__s.0__, int %__s.1__, "struct.std::ios_base"* %__io, int %__fill, int %__v) {
entry:
	tail call fastcc void %_ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE13_M_insert_intIlEES3_S3_RSt8ios_basewT_( )
	ret void
}

fastcc void %_ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE13_M_insert_intIlEES3_S3_RSt8ios_basewT_() {
entry:
	%tmp.38 = shl uint 0, ubyte 3		; <uint> [#uses=1]
	%tmp.39 = alloca sbyte, uint %tmp.38		; <sbyte*> [#uses=0]
	ret void
}
