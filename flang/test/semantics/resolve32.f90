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

module m2
  public s2, s4
  private s3
contains
  subroutine s2
  end
  subroutine s3
  end
  subroutine s4
  end
end module

module m
  use m2
  external bar
  interface
    subroutine foo
    end subroutine
  end interface
  integer :: i
  type t1
    integer :: c
  contains
    !ERROR: Procedure 'missing' not found
    procedure, nopass :: a => missing
    procedure, nopass :: b => s, s2
    !ERROR: 'c' is not a module procedure or external procedure with explicit interface
    procedure, nopass :: c
    !ERROR: DEFERRED is only allowed when an interface-name is provided
    procedure, nopass, deferred :: d => s
    !Note: s3 not found because it's not accessible -- should we issue a message
    !to that effect?
    !ERROR: Procedure 's3' not found
    procedure, nopass :: s3
    procedure, nopass :: foo
    !ERROR: 'bar' is not a module procedure or external procedure with explicit interface
    procedure, nopass :: bar
    !ERROR: 'i' is not a module procedure or external procedure with explicit interface
    procedure, nopass :: i
    !ERROR: Type parameter, component, or procedure binding 'b' already defined in this type
    procedure, nopass :: b => s4
    procedure(foo), nopass, deferred :: e
    procedure(s), nopass, deferred :: f
    !ERROR: Type parameter, component, or procedure binding 'f' already defined in this type
    procedure(foo), nopass, deferred :: f
    !ERROR: DEFERRED is required when an interface-name is provided
    procedure(foo), nopass :: g
    !ERROR: The interface of 'h' ('bar') is not an abstract interface or a procedure with an explicit interface
    procedure(bar), nopass, deferred :: h
  end type
  type t2
    integer :: i
  contains
    procedure, nopass :: b => s
    final :: f
    !ERROR: Type parameter, component, or procedure binding 'i' already defined in this type
    final :: i
  end type
  type t3
  contains
    private
    procedure, nopass :: b => s
    procedure, nopass, public :: f
  end type
contains
  subroutine s
  end
  subroutine f(x)
    type(t2) :: x
  end
end module
