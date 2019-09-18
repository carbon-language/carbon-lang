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

! Test specification expressions

module m
  type :: t(n)
    integer, len :: n = 1
    character(len=n) :: c
  end type
  interface
    integer function foo()
    end function
    pure integer function hasProcArg(p)
      procedure(cos) :: p
    end function
    real function realfunc(x)
      real, intent(in) :: x
    end function
  end interface
  integer :: coarray[*]
 contains
  pure integer function modulefunc(n)
    integer, value :: n
    modulefunc = n
  end function
  subroutine test(out, optional)
    !ERROR: The expression (foo()) cannot be used as a specification expression (reference to impure function 'foo')
    type(t(foo())) :: x1
    integer :: local
    !ERROR: The expression (local) cannot be used as a specification expression (reference to local entity 'local')
    type(t(local)) :: x2
    !ERROR: Cannot call function 'internal' in this context
    type(t(internal(0))) :: x3
    integer, intent(out) :: out
    !ERROR: The expression (out) cannot be used as a specification expression (reference to INTENT(OUT) dummy argument 'out')
    type(t(out)) :: x4
    integer, intent(in), optional :: optional
    !ERROR: The expression (optional) cannot be used as a specification expression (reference to OPTIONAL dummy argument 'optional')
    type(t(optional)) :: x5
    !ERROR: The expression (hasprocarg(realfunc)) cannot be used as a specification expression (dummy procedure argument)
    type(t(hasProcArg(realfunc))) :: x6
    !ERROR: The expression (coarray[1_8]) cannot be used as a specification expression (coindexed reference)
    type(t(coarray[1])) :: x7
    type(t(kind(foo()))) :: x101 ! ok
    type(t(modulefunc(0))) :: x102 ! ok
   contains
    pure integer function internal(n)
      integer, value :: n
      internal = n
    end function
  end subroutine
end module
