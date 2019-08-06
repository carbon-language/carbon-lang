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

! See Fortran 2018, clause 17
module ieee_exceptions

  type :: ieee_flag_type ! Fortran 2018, 17.2 & 17.3
    private
    integer(kind=1) :: flag = 0
  end type ieee_flag_type

  type(ieee_flag_type), parameter :: &
    ieee_invalid = ieee_flag_type(1), &
    ieee_overflow = ieee_flag_type(2), &
    ieee_divide_by_zero = ieee_flag_type(4), &
    ieee_underflow = ieee_flag_type(8), &
    ieee_inexact = ieee_flag_type(16)

  type(ieee_flag_type), parameter :: &
    ieee_usual(*) = [ &
      ieee_overflow, ieee_divide_by_zero, ieee_invalid ], &
    ieee_all(*) = [ &
      ieee_usual, ieee_underflow, ieee_inexact ]

  type :: ieee_modes_type ! Fortran 2018, 17.7
    private
  end type ieee_modes_type

  type :: ieee_status_type ! Fortran 2018, 17.7
    private
  end type ieee_status_type

 contains
  subroutine ieee_get_modes(modes)
    type(ieee_modes_type), intent(out) :: modes
  end subroutine ieee_get_modes

  subroutine ieee_set_modes(modes)
    type(ieee_modes_type), intent(in) :: modes
  end subroutine ieee_set_modes

  subroutine ieee_get_status(status)
    type(ieee_status_type), intent(out) :: status
  end subroutine ieee_get_status

  subroutine ieee_set_status(status)
    type(ieee_status_type), intent(in) :: status
  end subroutine ieee_set_status

  ! TODO: other interfaces (see Table 17.3)

end module ieee_exceptions
