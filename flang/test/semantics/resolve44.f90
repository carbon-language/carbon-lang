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

! Error tests for recursive use of derived types.

program main
  type :: recursive1
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(recursive1) :: bad1
    type(recursive1), pointer :: ok1
    type(recursive1), allocatable :: ok2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    class(recursive1) :: bad2
    class(recursive1), pointer :: ok3
    class(recursive1), allocatable :: ok4
  end type recursive1
  type :: recursive2(kind,len)
    integer, kind :: kind
    integer, len :: len
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(recursive2(kind,len)) :: bad1
    type(recursive2(kind,len)), pointer :: ok1
    type(recursive2(kind,len)), allocatable :: ok2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    class(recursive2(kind,len)) :: bad2
    class(recursive2(kind,len)), pointer :: ok3
    class(recursive2(kind,len)), allocatable :: ok4
  end type recursive2
  type :: recursive3(kind,len)
    integer, kind :: kind = 1
    integer, len :: len = 2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(recursive3) :: bad1
    type(recursive3), pointer :: ok1
    type(recursive3), allocatable :: ok2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    class(recursive3) :: bad2
    class(recursive3), pointer :: ok3
    class(recursive3), allocatable :: ok4
  end type recursive3
  !ERROR: Derived type 'recursive4' cannot extend itself
  type, extends(recursive4) :: recursive4
  end type recursive4
end program main
