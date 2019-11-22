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

! Forward references to derived types (error cases)

!ERROR: The derived type 'undef' was forward-referenced but not defined
type(undef) function f1()
  call sub(f1)
end function

!ERROR: The derived type 'undef' was forward-referenced but not defined
type(undef) function f2() result(r)
  call sub(r)
end function

!ERROR: The derived type 'undefpdt' was forward-referenced but not defined
type(undefpdt(1)) function f3()
  call sub(f3)
end function

!ERROR: The derived type 'undefpdt' was forward-referenced but not defined
type(undefpdt(1)) function f4() result(r)
  call sub(f4)
end function

!ERROR: 'bad' is not the name of a parameter for derived type 'pdt'
type(pdt(bad=1)) function f5()
  type :: pdt(good)
    integer, kind :: good = kind(0)
    integer(kind=good) :: n
  end type
end function

subroutine s1(q1)
  !ERROR: The derived type 'undef' was forward-referenced but not defined
  implicit type(undef)(q)
end subroutine

subroutine s2(q1)
  !ERROR: The derived type 'undefpdt' was forward-referenced but not defined
  implicit type(undefpdt(1))(q)
end subroutine

subroutine s3
  type :: t1
    !ERROR: Derived type 'undef' not found
    type(undef) :: x
  end type
end subroutine

subroutine s4
  type :: t1
    !ERROR: Derived type 'undefpdt' not found
    type(undefpdt(1)) :: x
  end type
end subroutine

subroutine s5(x)
  !ERROR: Derived type 'undef' not found
  type(undef) :: x
end subroutine

subroutine s6(x)
  !ERROR: Derived type 'undefpdt' not found
  type(undefpdt(1)) :: x
end subroutine

subroutine s7(x)
  !ERROR: Derived type 'undef' not found
  type, extends(undef) :: t
  end type
end subroutine
