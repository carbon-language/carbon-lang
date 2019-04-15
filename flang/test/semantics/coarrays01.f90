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

! Test selector and team-value in CHANGE TEAM statement

! Temporary, until we have real iso_fortran_env
module iso_fortran_env
  type :: team_type
  end type
end

! OK
subroutine s1
  use iso_fortran_env, only: team_type
  type(team_type) :: t
  real :: y[10,*]
  change team(t, x[10,*] => y)
  end team
  form team(1, t)
end

subroutine s2
  use iso_fortran_env
  type(team_type) :: t
  real :: y[10,*], y2[*], x[*]
  ! C1113
  !ERROR: Selector 'y' was already used as a selector or coarray in this statement
  change team(t, x[10,*] => y, x2[*] => y)
  end team
  !ERROR: Selector 'x' was already used as a selector or coarray in this statement
  change team(t, x[10,*] => y, x2[*] => x)
  end team
  !ERROR: Coarray 'y' was already used as a selector or coarray in this statement
  change team(t, x[10,*] => y, y[*] => y2)
  end team
end

subroutine s3
  type :: team_type
  end type
  type :: foo
  end type
  type(team_type) :: t1
  type(foo) :: t2
  real :: y[10,*]
  ! C1114
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  change team(t1, x[10,*] => y)
  end team
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  change team(t2, x[10,*] => y)
  end team
  !ERROR: Team variable 't1' must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  form team(1, t1)
  !ERROR: Team variable 't2' must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  form team(2, t2)
end
