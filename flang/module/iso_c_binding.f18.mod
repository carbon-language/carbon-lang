!mod$ v1 sum:9822adcde8c57677
module iso_c_binding
type::c_ptr
integer(8)::address
end type
type::c_funptr
integer(8)::address
end type
type(c_ptr),parameter::c_null_ptr=c_ptr(address=0_8)
type(c_funptr),parameter::c_null_funptr=c_funptr(address=0_8)
integer(4),parameter::c_int8_t=1_4
integer(4),parameter::c_int16_t=2_4
integer(4),parameter::c_int32_t=4_4
integer(4),parameter::c_int64_t=8_4
integer(4),parameter::c_int128_t=16_4
integer(4),parameter::c_int=4_4
integer(4),parameter::c_short=2_4
integer(4),parameter::c_long=8_4
integer(4),parameter::c_long_long=8_4
integer(4),parameter::c_signed_char=1_4
integer(4),parameter::c_size_t=8_4
integer(4),parameter::c_intmax_t=16_4
integer(4),parameter::c_intptr_t=8_4
integer(4),parameter::c_ptrdiff_t=8_4
integer(4),parameter::c_int_least8_t=1_4
integer(4),parameter::c_int_fast8_t=1_4
integer(4),parameter::c_int_least16_t=2_4
integer(4),parameter::c_int_fast16_t=2_4
integer(4),parameter::c_int_least32_t=4_4
integer(4),parameter::c_int_fast32_t=4_4
integer(4),parameter::c_int_least64_t=8_4
integer(4),parameter::c_int_fast64_t=8_4
integer(4),parameter::c_int_least128_t=16_4
integer(4),parameter::c_int_fast128_t=16_4
integer(4),parameter::c_float=4_4
integer(4),parameter::c_double=8_4
integer(4),parameter::c_long_double=10_4
integer(4),parameter::c_float_complex=4_4
integer(4),parameter::c_double_complex=8_4
integer(4),parameter::c_long_double_complex=10_4
integer(4),parameter::c_bool=1_4
integer(4),parameter::c_char=1_4
contains
function c_associated(c_ptr_1,c_ptr_2)
logical(4)::c_associated
type(c_ptr),intent(in)::c_ptr_1
type(c_ptr),intent(in),optional::c_ptr_2
end
subroutine c_f_pointer(cptr,fptr,shape)
type(c_ptr),intent(in)::cptr
type(*),intent(out),pointer::fptr(..)
integer(4),intent(in),optional::shape(1_8:)
end
end
