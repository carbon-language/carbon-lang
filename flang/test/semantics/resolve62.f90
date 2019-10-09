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
