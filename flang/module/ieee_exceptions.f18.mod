!mod$ v1 sum:1e5a37c09db771dd
module ieee_exceptions
type::ieee_flag_type
integer(1),private::flag=0_4
end type
type(ieee_flag_type),parameter::ieee_invalid=ieee_flag_type(flag=1_1)
type(ieee_flag_type),parameter::ieee_overflow=ieee_flag_type(flag=2_1)
type(ieee_flag_type),parameter::ieee_divide_by_zero=ieee_flag_type(flag=4_1)
type(ieee_flag_type),parameter::ieee_underflow=ieee_flag_type(flag=8_1)
type(ieee_flag_type),parameter::ieee_inexact=ieee_flag_type(flag=16_1)
type(ieee_flag_type),parameter::ieee_usual(1_8:)=[ieee_flag_type::ieee_flag_type(flag=2_1),ieee_flag_type(flag=4_1),ieee_flag_type(flag=1_1)]
type(ieee_flag_type),parameter::ieee_all(1_8:)=[ieee_flag_type::ieee_flag_type(flag=2_1),ieee_flag_type(flag=4_1),ieee_flag_type(flag=1_1),ieee_flag_type(flag=8_1),ieee_flag_type(flag=16_1)]
type::ieee_modes_type
end type
type::ieee_status_type
end type
contains
subroutine ieee_get_modes(modes)
type(ieee_modes_type),intent(out)::modes
end
subroutine ieee_set_modes(modes)
type(ieee_modes_type),intent(in)::modes
end
subroutine ieee_get_status(status)
type(ieee_status_type),intent(out)::status
end
subroutine ieee_set_status(status)
type(ieee_status_type),intent(in)::status
end
end
