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

! Test 15.5.2.6 constraints and restrictions for ALLOCATABLE
! dummy arguments.

module m

  real, allocatable :: cov[:], com[:,:]

 contains

  subroutine s01(x)
    real, allocatable :: x
  end subroutine
  subroutine s02(x)
    real, allocatable :: x[:]
  end subroutine
  subroutine s03(x)
    real, allocatable :: x[:,:]
  end subroutine
  subroutine s04(x)
    real, allocatable, intent(in) :: x
  end subroutine
  subroutine s05(x)
    real, allocatable, intent(out) :: x
  end subroutine
  subroutine s06(x)
    real, allocatable, intent(in out) :: x
  end subroutine
  function allofunc()
    real, allocatable :: allofunc
  end function

  subroutine test(x)
    real :: scalar
    real, allocatable, intent(in) :: x
    ! ERROR: ALLOCATABLE dummy argument must be associated with an ALLOCATABLE effective argument
    call s01(scalar)
    ! ERROR: ALLOCATABLE dummy argument must be associated with an ALLOCATABLE effective argument
    call s01(1.)
    ! ERROR: ALLOCATABLE dummy argument must be associated with an ALLOCATABLE effective argument
    call s01(allofunc()) ! subtle: ALLOCATABLE function result isn't
    call s02(cov) ! ok
    call s03(com) ! ok
    ! ERROR: Dummy argument has corank 1, but effective argument has corank 2
    call s02(com)
    ! ERROR: Dummy argument has corank 2, but effective argument has corank 1
    call s03(cov)
    call s04(cov[1]) ! ok
    ! ERROR: Coindexed ALLOCATABLE effective argument must be associated with INTENT(IN) dummy argument
    call s01(cov[1])
    ! ERROR: Effective argument associated with INTENT(OUT) dummy is not definable.
    call s05(x)
    ! ERROR: Effective argument associated with INTENT(IN OUT) dummy is not definable.
    call s06(x)
  end subroutine
end module
