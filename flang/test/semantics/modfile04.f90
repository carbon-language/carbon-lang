! Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

! modfile with subprograms

module m1
  type :: t
  end type
contains

  pure subroutine s(x, y) bind(c)
    logical x
    intent(inout) y
    intent(in) x
  end subroutine

  real function f1() result(x)
    x = 1.0
  end function

  function f2(y)
    complex y
    f2 = 2.0
  end function

end

module m2
contains
  type(t) function f3(x)
    use m1
    integer, parameter :: a = 2
    type t2(b)
      integer, kind :: b = a
      integer :: y
    end type
    type(t2) :: x
  end
  function f4() result(x)
    implicit complex(x)
  end
end

!Expect: m1.mod
!module m1
!type::t
!end type
!contains
!pure subroutine s(x,y) bind(c)
!logical(4),intent(in)::x
!real(4),intent(inout)::y
!end
!function f1() result(x)
!real(4)::x
!end
!function f2(y)
!complex(4)::y
!real(4)::f2
!end
!end

!Expect: m2.mod
!module m2
!contains
!function f3(x)
! use m1,only:t
! type::t2(b)
!  integer(4),kind::b=2_4
!  integer(4)::y
! end type
! type(t2(b=2_4))::x
! type(t)::f3
!end
!function f4() result(x)
!complex(4)::x
!end
!end
