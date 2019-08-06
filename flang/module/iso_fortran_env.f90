! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! See Fortran 2018, clause 16.10.2
! TODO: These are placeholder values so that some tests can be run.

module iso_fortran_env

  integer, parameter :: atomic_int_kind = 8
  integer, parameter :: atomic_logical_kind = 8

  integer, parameter :: character_kinds(*) = [1, 2, 4]
  integer, parameter :: int8 = 1, int16 = 2, int32 = 4, int64 = 8, int128 = 16
  integer, parameter :: integer_kinds(*) = [int8, int16, int32, int64, int128]
  integer, parameter :: &
    logical8 = 1, logical16 = 2, logical32 = 4, logical64 = 8
  integer, parameter :: logical_kinds(*) = &
    [logical8, logical16, logical32, logical64]
  integer, parameter :: &
    real16 = 2, real32 = 4, real64 = 8, real80 = 10, real128 = 16
  integer, parameter :: real_kinds(*) = &
    [real16, 3, real32, real64, real80, real128]

  integer, parameter :: current_team = -1, initial_team = -2, parent_team = -3

  integer, parameter :: input_unit = 5, output_unit = 6
  integer, parameter :: iostat_end = -1, iostat_eor = -2
  integer, parameter :: iostat_inquire_internal_unit = -1

  integer, parameter :: character_storage_size = 8
  integer, parameter :: file_storage_size = 8
  integer, parameter :: numeric_storage_size = 32

  integer, parameter :: stat_failed_image = -1
  integer, parameter :: stat_locked = 2
  integer, parameter :: stat_locked_other_image = 3
  integer, parameter :: stat_stopped_image = 4
  integer, parameter :: stat_unlocked = 5
  integer, parameter :: stat_unlocked_failed_image = 6

  type :: event_type
    private
    integer(kind=atomic_int_kind) :: count = 0
  end type event_type

  type :: lock_type
    private
    integer(kind=atomic_int_kind) :: count = 0
  end type lock_type

  type :: team_type
    private
    integer(kind=int64) :: id = 0
  end type team_type

 contains

  character(len=80) function compiler_options()
    compiler_options = 'COMPILER_OPTIONS() not yet implemented'
  end function compiler_options

  character(len=80) function compiler_version()
    compiler_version = 'f18 in development'
  end function compiler_version
end module iso_fortran_env

