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

! Tests -fget-symbols-sources finding all symbols in file.

module mm1
 private :: f
contains
 pure subroutine s (x, y) bind(c)
  intent(in) :: x
  intent(inout) :: y
 contains
  pure subroutine ss
  end subroutine
 end subroutine
 recursive pure function f() result(x)
  real, allocatable :: x
  x = 1.0
 end function
end module

! RUN: ${F18} -fget-symbols-sources -fparse-only -fdebug-semantics %s 2>&1 | ${FileCheck} %s
! CHECK-ONCE:mm1:.*getsymbols01.f90, 17, 8-11
! CHECK-ONCE:f:.*getsymbols01.f90, 27, 26-27
! CHECK-ONCE:s:.*getsymbols01.f90, 20, 18-19
! CHECK-ONCE:ss:.*getsymbols01.f90, 24, 19-21
! CHECK-ONCE:x:.*getsymbols01.f90, 20, 21-22
! CHECK-ONCE:y:.*getsymbols01.f90, 20, 24-25
! CHECK-ONCE:x:.*getsymbols01.f90, 28, 24-25
