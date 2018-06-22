! Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

subroutine s1
  type :: t
    integer :: i
    integer :: s1
    integer :: t
  end type
  !ERROR: 't' is already declared in this scoping unit
  integer :: t
  integer :: i, j
  type(t) :: x
  !ERROR: Derived type 't2' not found
  type(t2) :: y
  !ERROR: 'z' is not an object of derived type; it is implicitly typed
  i = z%i
  !ERROR: 's1' is not an object of derived type
  i = s1%i
  !ERROR: 'j' is not an object of derived type
  i = j%i
  !ERROR: Component 'j' not found in derived type 't'
  i = x%j
  i = x%i  !OK
end subroutine

subroutine s2
  type :: t1
    integer :: i
  end type
  type :: t2
    type(t1) :: x
  end type
  type(t2) :: y
  integer :: i
  !ERROR: Component 'j' not found in derived type 't1'
  k = y%x%j
  k = y%x%i !OK
end subroutine
