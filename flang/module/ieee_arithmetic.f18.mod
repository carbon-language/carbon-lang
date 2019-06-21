!mod$ v1 sum:377fafa870889c4f
module ieee_arithmetic
type::ieee_class_type
integer(1),private::which=0_4
end type
type(ieee_class_type),parameter::ieee_signaling_nan=ieee_class_type(which=1_1)
type(ieee_class_type),parameter::ieee_quiet_nan=ieee_class_type(which=2_1)
type(ieee_class_type),parameter::ieee_negative_inf=ieee_class_type(which=3_1)
type(ieee_class_type),parameter::ieee_negative_normal=ieee_class_type(which=4_1)
type(ieee_class_type),parameter::ieee_negative_denormal=ieee_class_type(which=5_1)
type(ieee_class_type),parameter::ieee_negative_zero=ieee_class_type(which=6_1)
type(ieee_class_type),parameter::ieee_positive_zero=ieee_class_type(which=7_1)
type(ieee_class_type),parameter::ieee_positive_subnormal=ieee_class_type(which=8_1)
type(ieee_class_type),parameter::ieee_positive_normal=ieee_class_type(which=9_1)
type(ieee_class_type),parameter::ieee_positive_inf=ieee_class_type(which=10_1)
type(ieee_class_type),parameter::ieee_other_value=ieee_class_type(which=11_1)
type(ieee_class_type),parameter::ieee_negative_subnormal=ieee_class_type(which=5_1)
type(ieee_class_type),parameter::ieee_positive_denormal=ieee_class_type(which=5_1)
type::ieee_round_type
integer(1),private::mode=0_4
end type
type(ieee_round_type),parameter::ieee_nearest=ieee_round_type(mode=1_1)
type(ieee_round_type),parameter::ieee_to_zero=ieee_round_type(mode=2_1)
type(ieee_round_type),parameter::ieee_up=ieee_round_type(mode=3_1)
type(ieee_round_type),parameter::ieee_down=ieee_round_type(mode=4_1)
type(ieee_round_type),parameter::ieee_away=ieee_round_type(mode=5_1)
type(ieee_round_type),parameter::ieee_other=ieee_round_type(mode=6_1)
generic::operator(==)=>class_eq,round_eq
generic::operator(/=)=>class_ne,round_ne
generic::ieee_class=>ieee_class_a2,ieee_class_a3,ieee_class_a4,ieee_class_a8,ieee_class_a10,ieee_class_a16
generic::ieee_copy_sign=>ieee_copy_sign_a2,ieee_copy_sign_a3,ieee_copy_sign_a4,ieee_copy_sign_a8,ieee_copy_sign_a10,ieee_copy_sign_a16
contains
elemental function class_eq(x,y)
logical(4)::class_eq
type(ieee_class_type),intent(in)::x
type(ieee_class_type),intent(in)::y
end
elemental function class_ne(x,y)
logical(4)::class_ne
type(ieee_class_type),intent(in)::x
type(ieee_class_type),intent(in)::y
end
elemental function round_eq(x,y)
logical(4)::round_eq
type(ieee_round_type),intent(in)::x
type(ieee_round_type),intent(in)::y
end
elemental function round_ne(x,y)
logical(4)::round_ne
type(ieee_round_type),intent(in)::x
type(ieee_round_type),intent(in)::y
end
elemental private function classify(expo,maxexpo,negative,significandnz,quietbit)
type(ieee_class_type)::classify
integer(4),intent(in)::expo
integer(4),intent(in)::maxexpo
logical(4),intent(in)::negative
logical(4),intent(in)::significandnz
logical(4),intent(in)::quietbit
end
elemental function ieee_class_a2(x)
type(ieee_class_type)::ieee_class_a2
real(2),intent(in)::x
end
elemental function ieee_class_a3(x)
type(ieee_class_type)::ieee_class_a3
real(3),intent(in)::x
end
elemental function ieee_class_a4(x)
type(ieee_class_type)::ieee_class_a4
real(4),intent(in)::x
end
elemental function ieee_class_a8(x)
type(ieee_class_type)::ieee_class_a8
real(8),intent(in)::x
end
elemental function ieee_class_a10(x)
type(ieee_class_type)::ieee_class_a10
real(10),intent(in)::x
end
elemental function ieee_class_a16(x)
type(ieee_class_type)::ieee_class_a16
real(16),intent(in)::x
end
elemental function ieee_copy_sign_a2(x,y)
real(2)::ieee_copy_sign_a2
real(2),intent(in)::x
real(2),intent(in)::y
end
elemental function ieee_copy_sign_a3(x,y)
real(3)::ieee_copy_sign_a3
real(3),intent(in)::x
real(3),intent(in)::y
end
elemental function ieee_copy_sign_a4(x,y)
real(4)::ieee_copy_sign_a4
real(4),intent(in)::x
real(4),intent(in)::y
end
elemental function ieee_copy_sign_a8(x,y)
real(8)::ieee_copy_sign_a8
real(8),intent(in)::x
real(8),intent(in)::y
end
elemental function ieee_copy_sign_a10(x,y)
real(10)::ieee_copy_sign_a10
real(10),intent(in)::x
real(10),intent(in)::y
end
elemental function ieee_copy_sign_a16(x,y)
real(16)::ieee_copy_sign_a16
real(16),intent(in)::x
real(16),intent(in)::y
end
end
