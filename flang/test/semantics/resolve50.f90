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

! Test coarray association in CHANGE TEAM statement

subroutine s1
  use iso_fortran_env
  type(team_type) :: t
  complex :: x[*]
  real :: y[*]
  real :: z
  ! OK
  change team(t, x[*] => y)
  end team
  ! C1116
  !ERROR: Selector in coarray association must name a coarray
  change team(t, x[*] => 1)
  end team
  !ERROR: Selector in coarray association must name a coarray
  change team(t, x[*] => z)
  end team
end

subroutine s2
  use iso_fortran_env
  type(team_type) :: t
  real :: y[10,*], y2[*], x[*]
  ! C1113
  !ERROR: The codimensions of 'x' have already been declared
  change team(t, x[10,*] => y, x[*] => y2)
  end team
end
