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

! Resolve generic based on number of arguments
subroutine s1
  interface f
    real function f1(x)
      optional :: x
    end
    real function f2(x, y)
    end
  end interface
  z = f(1.0)
  z = f(1.0, 2.0)
  !ERROR: No specific procedure of generic 'f' matches the actual arguments
  z = f(1.0, 2.0, 3.0)
end

! Elemental and non-element function both match: non-elemental one should be used
subroutine s2
  interface f
    logical elemental function f1(x)
      intent(in) :: x
    end
    real function f2(x)
      real :: x(10)
    end
  end interface
  real :: x, y(10), z
  logical :: a
  a = f(1.0)
  a = f(y)  !TODO: this should resolve to f2 -- should get error here
end

! Resolve named operator
subroutine s3
  interface operator(.foo.)
    pure integer(8) function f_real(x, y)
      real, intent(in) :: x, y
    end
    pure integer(8) function f_integer(x, y)
      integer, intent(in) :: x, y
    end
  end interface
  logical :: a, b, c
  x = y .foo. z  ! OK: f_real
  i = j .foo. k  ! OK: f_integer
  !ERROR: No specific procedure of generic operator '.foo.' matches the actual arguments
  a = b .foo. c
end

! Generic resolves successfully but error analyzing call
module m4
  real, protected :: x
  real :: y
  interface s
    subroutine s1(x)
      real, intent(out) :: x
    end
    subroutine s2(x, y)
      real :: x, y
    end
  end interface
end
subroutine s4a
  use m4
  real :: z
  !OK
  call s(z)
end
subroutine s4b
  use m4
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' must be definable
  call s(x)
end
pure subroutine s4c
  use m4
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' must be definable
  call s(y)
end
