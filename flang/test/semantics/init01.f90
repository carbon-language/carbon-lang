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

! Object pointer initializer error tests

subroutine test(j)
  integer, intent(in) :: j
  real, allocatable, target, save :: x1
  real, codimension[:], target, save :: x2
  real, save :: x3
  real, target :: x4
  real, target, save :: x5(10)
!ERROR: An initial data target may not be a reference to an ALLOCATABLE 'x1'
  real, pointer :: p1 => x1
!ERROR: An initial data target may not be a reference to a coarray 'x2'
  real, pointer :: p2 => x2
!ERROR: An initial data target may not be a reference to an object 'x3' that lacks the TARGET attribute
  real, pointer :: p3 => x3
!ERROR: An initial data target may not be a reference to an object 'x4' that lacks the SAVE attribute
  real, pointer :: p4 => x4
!ERROR: Pointer 'p5' cannot be initialized with a reference to a designator with non-constant subscripts
  real, pointer :: p5 => x5(j)
!ERROR: Pointer 'p6' of rank 0 cannot be initialized with a target of different rank (1)
  real, pointer :: p6 => x5

!TODO: type incompatibility, non-deferred type parameter values, contiguity

end subroutine test
