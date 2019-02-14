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

! Error tests for structure constructors.
! Errors caught by name resolution are tested elsewhere; these are the
! errors meant to be caught by expression semantic analysis, as well as
! acceptable use cases.
! Type parameters are used here to make the parses unambiguous.

module module1
  type :: type1(j)
    integer, kind :: j
    integer :: n = 1
  end type type1
  type, extends(type1) :: type2(k)
    integer, kind :: k
    integer :: m
  end type type2
  type, abstract :: abstract(j)
    integer, kind :: j
    integer :: n
  end type abstract
  type :: privaten(j)
    integer, kind :: j
    integer, private :: n
  end type privaten
 contains
  subroutine type1arg(x)
    type(type1(0)), intent(in) :: x
  end subroutine type1arg
  subroutine type2arg(x)
    type(type2(0,0)), intent(in) :: x
  end subroutine type2arg
  subroutine abstractarg(x)
    type(abstract(0)), intent(in) :: x
  end subroutine abstractarg
  subroutine errors
    call type1arg(type1(0)())
    call type1arg(type1(0)(1))
    call type1arg(type1(0)(n=1))
    !ERROR: Type parameter 'j' cannot be a component of this structure constructor
    call type1arg(type1(0)(j=1))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type1arg(type1(0)(1,n=2))
    !ERROR: Value in structure constructor lacks a component name
    call type1arg(type1(0)(n=1,2))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type1arg(type1(0)(n=1,n=2))
    call type2arg(type2(0,0)(n=1,m=2))
    call type2arg(type2(0,0)(m=2))
    !ERROR: Structure constructor lacks a value for component 'm'
    call type2arg(type2(0,0)())
    call type2arg(type2(0,0)(type1=type1(0)(n=1),m=2))
    call type2arg(type2(0,0)(type1=type1(0)(),m=2))
    !ERROR: Component 'type1' conflicts with another component earlier in this structure constructor
    call type2arg(type2(0,0)(n=1,type1=type1(0)(n=2),m=3))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type2arg(type2(0,0)(type1=type1(0)(n=1),n=2,m=3))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type2arg(type2(0,0)(type1=type1(0)(1),n=2,m=3))
    !ERROR: Type parameter 'j' cannot be a component of this structure constructor
    call type2arg(type2(0,0)(j=1, &
    !ERROR: Type parameter 'k' cannot be a component of this structure constructor
      k=2,m=3))
  end subroutine errors
end module module1
