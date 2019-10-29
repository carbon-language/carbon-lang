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

! Test 8.5.18 constraints on the VALUE attribute

module m
  type :: hasCoarray
    real :: coarray[*]
  end type
 contains
  !ERROR: VALUE attribute may apply only to a dummy data object
  subroutine C863(notData,assumedSize,coarray,coarrayComponent)
    external :: notData
    !ERROR: VALUE attribute may apply only to a dummy argument
    real, value :: notADummy
    value :: notData
    !ERROR: VALUE attribute may not apply to an assumed-size array
    real, value :: assumedSize(10,*)
    !ERROR: VALUE attribute may not apply to a coarray
    real, value :: coarray[*]
    !ERROR: VALUE attribute may not apply to a type with a coarray ultimate component
    type(hasCoarray), value :: coarrayComponent
  end subroutine
  subroutine C864(allocatable, inout, out, pointer, volatile)
    !ERROR: VALUE attribute may not apply to an ALLOCATABLE
    real, value, allocatable :: allocatable
    !ERROR: VALUE attribute may not apply to an INTENT(IN OUT) argument
    real, value, intent(in out) :: inout
    !ERROR: VALUE attribute may not apply to an INTENT(OUT) argument
    real, value, intent(out) :: out
    !ERROR: VALUE attribute may not apply to a POINTER
    real, value, pointer :: pointer
    !ERROR: VALUE attribute may not apply to a VOLATILE
    real, value, volatile :: volatile
  end subroutine
  subroutine C865(optional) bind(c)
    !ERROR: VALUE attribute may not apply to an OPTIONAL in a BIND(C) procedure
    real, value, optional :: optional
  end subroutine
end module
