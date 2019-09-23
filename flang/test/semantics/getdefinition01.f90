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

! Tests -fget-definition returning source position of symbol definition.

module m1
 private :: f
contains
 pure subroutine s (x, yyy) bind(c)
  intent(in) :: x
  intent(inout) :: yyy
 contains
  pure subroutine ss
  end subroutine
 end subroutine
 recursive pure function f() result(x)
  real, allocatable :: x
  x = 1.0
 end function
end module

! RUN: echo %t 1>&2;
! RUN: ${F18} -fget-definition 21 17 18 -fparse-only -fdebug-semantics %s > %t;
! RUN: ${F18} -fget-definition 22 20 23 -fparse-only -fdebug-semantics %s >> %t;
! RUN: ${F18} -fget-definition 29 3 4 -fparse-only -fdebug-semantics %s >> %t;
! RUN: ${F18} -fget-definition -fparse-only -fdebug-semantics %s >> %t 2>&1;
! RUN: cat %t | ${FileCheck} %s
! CHECK:x:.*getdefinition01.f90, 20, 21-22
! CHECK:yyy:.*getdefinition01.f90, 20, 24-27
! CHECK:x:.*getdefinition01.f90, 28, 24-25
! CHECK:Invalid argument to -fget-definitions
