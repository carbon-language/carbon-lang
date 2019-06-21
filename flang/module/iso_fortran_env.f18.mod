!mod$ v1 sum:0bfd96183f25666a
module iso_fortran_env
integer(4),parameter::atomic_int_kind=8_4
integer(4),parameter::atomic_logical_kind=8_4
integer(4),parameter::character_kinds(1_8:)=[Integer(4)::1_4,2_4,4_4]
integer(4),parameter::int8=1_4
integer(4),parameter::int16=2_4
integer(4),parameter::int32=4_4
integer(4),parameter::int64=8_4
integer(4),parameter::int128=16_4
integer(4),parameter::integer_kinds(1_8:)=[Integer(4)::1_4,2_4,4_4,8_4,16_4]
integer(4),parameter::logical8=1_4
integer(4),parameter::logical16=2_4
integer(4),parameter::logical32=4_4
integer(4),parameter::logical64=8_4
integer(4),parameter::logical_kinds(1_8:)=[Integer(4)::1_4,2_4,4_4,8_4]
integer(4),parameter::real16=2_4
integer(4),parameter::real32=4_4
integer(4),parameter::real64=8_4
integer(4),parameter::real80=10_4
integer(4),parameter::real128=16_4
integer(4),parameter::real_kinds(1_8:)=[Integer(4)::2_4,3_4,4_4,8_4,10_4,16_4]
integer(4),parameter::current_team=-1_4
integer(4),parameter::initial_team=-2_4
integer(4),parameter::parent_team=-3_4
integer(4),parameter::input_unit=5_4
integer(4),parameter::output_unit=6_4
integer(4),parameter::iostat_end=-1_4
integer(4),parameter::iostat_eor=-2_4
integer(4),parameter::iostat_inquire_internal_unit=-1_4
integer(4),parameter::character_storage_size=8_4
integer(4),parameter::file_storage_size=8_4
integer(4),parameter::numeric_storage_size=32_4
integer(4),parameter::stat_failed_image=-1_4
integer(4),parameter::stat_locked=2_4
integer(4),parameter::stat_locked_other_image=3_4
integer(4),parameter::stat_stopped_image=4_4
integer(4),parameter::stat_unlocked=5_4
integer(4),parameter::stat_unlocked_failed_image=6_4
type::event_type
integer(8),private::count=0_4
end type
type::lock_type
integer(8),private::count=0_4
end type
type::team_type
integer(8),private::id=0_4
end type
contains
function compiler_options()
character(80_4,1)::compiler_options
end
function compiler_version()
character(80_4,1)::compiler_version
end
end
