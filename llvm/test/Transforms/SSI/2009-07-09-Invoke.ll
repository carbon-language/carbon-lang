; RUN: opt < %s -ssi-everything -disable-output
; PR4511

	%"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >" = type { %"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >::_Vector_impl" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" = type { i8* }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" = type { i32, i32, i32 }
	%"struct.std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >" = type { %"struct.std::_Vector_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >" }

declare void @_Unwind_Resume(i8*)

declare fastcc %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* @_ZSt24__uninitialized_copy_auxIPSsS0_ET0_T_S2_S1_St12__false_type(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*)

define fastcc void @_ZNSt6vectorISsSaISsEE9push_backERKSs(%"struct.std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >"* nocapture %this, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* nocapture %__x) {
entry:
	br i1 undef, label %_ZNSt12_Vector_baseISsSaISsEE11_M_allocateEj.exit.i, label %bb

bb:		; preds = %entry
	ret void

_ZNSt12_Vector_baseISsSaISsEE11_M_allocateEj.exit.i:		; preds = %entry
	%0 = invoke fastcc %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* @_ZSt24__uninitialized_copy_auxIPSsS0_ET0_T_S2_S1_St12__false_type(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* undef, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* undef, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* undef)
			to label %invcont14.i unwind label %ppad81.i		; <%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*> [#uses=3]

invcont14.i:		; preds = %_ZNSt12_Vector_baseISsSaISsEE11_M_allocateEj.exit.i
	%1 = icmp eq %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %0, null		; <i1> [#uses=1]
	br i1 %1, label %bb19.i, label %bb.i17.i

bb.i17.i:		; preds = %invcont14.i
	%2 = invoke fastcc i8* @_ZNSs4_Rep8_M_cloneERKSaIcEj(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep"* undef, i32 0)
			to label %bb2.i25.i unwind label %ppad.i.i.i23.i		; <i8*> [#uses=0]

ppad.i.i.i23.i:		; preds = %bb.i17.i
	invoke void @_Unwind_Resume(i8* undef)
			to label %.noexc.i24.i unwind label %lpad.i29.i

.noexc.i24.i:		; preds = %ppad.i.i.i23.i
	unreachable

bb2.i25.i:		; preds = %bb.i17.i
	unreachable

lpad.i29.i:		; preds = %ppad.i.i.i23.i
	invoke void @_Unwind_Resume(i8* undef)
			to label %.noexc.i9 unwind label %ppad81.i

.noexc.i9:		; preds = %lpad.i29.i
	unreachable

bb19.i:		; preds = %invcont14.i
	%3 = getelementptr %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %0, i32 1		; <%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*> [#uses=2]
	%4 = invoke fastcc %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* @_ZSt24__uninitialized_copy_auxIPSsS0_ET0_T_S2_S1_St12__false_type(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* undef, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* undef, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %3)
			to label %invcont20.i unwind label %ppad81.i		; <%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*> [#uses=0]

invcont20.i:		; preds = %bb19.i
	unreachable

invcont32.i:		; preds = %ppad81.i
	unreachable

ppad81.i:		; preds = %bb19.i, %lpad.i29.i, %_ZNSt12_Vector_baseISsSaISsEE11_M_allocateEj.exit.i
	%__new_finish.0.i = phi %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* [ %0, %lpad.i29.i ], [ undef, %_ZNSt12_Vector_baseISsSaISsEE11_M_allocateEj.exit.i ], [ %3, %bb19.i ]		; <%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*> [#uses=0]
	br i1 undef, label %invcont32.i, label %bb.i.i.i.i

bb.i.i.i.i:		; preds = %bb.i.i.i.i, %ppad81.i
	br label %bb.i.i.i.i
}

declare fastcc i8* @_ZNSs4_Rep8_M_cloneERKSaIcEj(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep"* nocapture, i32)
