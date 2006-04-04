; RUN: llvm-as < %s | llc -march=alpha

target endian = little
target pointersize = 64
target triple = "alphaev67-unknown-linux-gnu"
	%llvm.dbg.compile_unit.type = type { uint, {  }*, uint, uint, sbyte*, sbyte*, sbyte* }
	%struct._Callback_list = type { %struct._Callback_list*, void (uint, %struct.ios_base*, int)*, int, int }
	%struct._Impl = type { int, %struct.facet**, ulong, %struct.facet**, sbyte** }
	%struct._Words = type { sbyte*, long }
	"struct.__codecvt_abstract_base<char,char,__mbstate_t>" = type { %struct.facet }
	"struct.basic_streambuf<char,std::char_traits<char> >" = type { int (...)**, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct.locale }
	%struct.facet = type { int (...)**, int }
	%struct.ios_base = type { int (...)**, long, long, uint, uint, uint, %struct._Callback_list*, %struct._Words, [8 x %struct._Words], int, %struct._Words*, %struct.locale }
	%struct.locale = type { %struct._Impl* }
	"struct.ostreambuf_iterator<char,std::char_traits<char> >" = type { "struct.basic_streambuf<char,std::char_traits<char> >"*, bool }
%llvm.dbg.compile_unit1047 = external global %llvm.dbg.compile_unit.type		; <%llvm.dbg.compile_unit.type*> [#uses=1]

implementation   ; Functions:

void %_ZNKSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE15_M_insert_floatIdEES3_S3_RSt8ios_baseccT_() {
entry:
	%tmp234 = seteq sbyte 0, 0		; <bool> [#uses=1]
	br bool %tmp234, label %cond_next243, label %cond_true235

cond_true235:		; preds = %entry
	ret void

cond_next243:		; preds = %entry
	%tmp428 = load long* null		; <long> [#uses=1]
	%tmp428 = cast long %tmp428 to uint		; <uint> [#uses=1]
	%tmp429 = alloca sbyte, uint %tmp428		; <sbyte*> [#uses=0]
	call void %llvm.dbg.stoppoint( uint 1146, uint 0, {  }* cast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit1047 to {  }*) )
	unreachable
}

declare void %llvm.dbg.stoppoint(uint, uint, {  }*)
