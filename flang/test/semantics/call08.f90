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

! Test 15.5.2.8 coarray dummy arguments

module m

  real :: c1[*]
  real, volatile :: c2[*]
  real, pointer :: c3(:)[*]
  real, pointer, contiguous :: c4(:)[*]

 contains

  subroutine s01(x)
    real :: x[*]
  end subroutine
  subroutine s02(x)
    real, volatile :: x[*]
  end subroutine
  subroutine s03(x)
    real, contiguous :: x(:)[*]
  end subroutine
  subroutine s04(x)
    real :: x(*)[*]
  end subroutine

  subroutine test(x)
    real :: scalar
    real :: x(:)[*]
    call s01(c1) ! ok
    call s02(c2) ! ok
    call s03(c4) ! ok
    call s04(c4) ! ok
    ! ERROR: Effective argument associated with a coarray dummy argument must be a coarray
    call s01(scalar)
    ! ERROR: VOLATILE coarray cannot be associated with non-VOLATILE dummy argument
    call s01(c2)
    ! ERROR: non-VOLATILE coarray cannot be associated with VOLATILE dummy argument
    call s02(c1)
    ! ERROR: Effective argument associated with a CONTIGUOUS coarray dummy argument must be simply contiguous
    call s03(c3)
    ! ERROR: Effective argument associated with a CONTIGUOUS coarray dummy argument must be simply contiguous
    call s03(x)
    ! ERROR: Effective argument associated with a CONTIGUOUS coarray dummy argument must be simply contiguous
    call s04(c3)
    ! ERROR: Effective argument associated with a CONTIGUOUS coarray dummy argument must be simply contiguous
    call s04(x)
  end subroutine
end module
