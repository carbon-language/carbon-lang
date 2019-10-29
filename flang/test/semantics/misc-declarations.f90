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

! Miscellaneous constraint and requirement checking on declarations:
! - 8.5.6.2 & 8.5.6.3 constraints on coarrays
! - 8.5.19 constraints on the VOLATILE attribute

module m
  !ERROR: ALLOCATABLE coarray must have a deferred coshape
  real, allocatable :: mustBeDeferred[*]  ! C827
  !ERROR: Non-ALLOCATABLE coarray must have an explicit coshape
  real :: mustBeExplicit[:]  ! C828
  type :: hasCoarray
    real :: coarray[*]
  end type
  real :: coarray[*]
  type(hasCoarray) :: coarrayComponent
 contains
  !ERROR: VOLATILE attribute may not apply to an INTENT(IN) argument
  subroutine C866(x)
    intent(in) :: x
    volatile :: x
    !ERROR: VOLATILE attribute may apply only to a variable
    volatile :: notData
    external :: notData
  end subroutine
  subroutine C867
    !ERROR: VOLATILE attribute may not apply to a coarray accessed by USE or host association
    volatile :: coarray
    !ERROR: VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association
    volatile :: coarrayComponent
  end subroutine
  subroutine C868(coarray,coarrayComponent)
    real, volatile :: coarray[*]
    type(hasCoarray) :: coarrayComponent
    block
      !ERROR: VOLATILE attribute may not apply to a coarray accessed by USE or host association
      volatile :: coarray
      !ERROR: VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association
      volatile :: coarrayComponent
    end block
  end subroutine
end module
