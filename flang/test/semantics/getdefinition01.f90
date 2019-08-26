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

!DEF: /m Module
module m
 !DEF: /m/f PRIVATE, PURE, RECURSIVE Subprogram REAL(4)
 private :: f
contains
 !DEF: /m/s BIND(C), PUBLIC, PURE Subprogram
 !DEF: /m/s/x INTENT(IN) (implicit) ObjectEntity REAL(4)
 !DEF: /m/s/y INTENT(INOUT) (implicit) ObjectEntity REAL(4)
 pure subroutine s (x, yyy) bind(c)
  !REF: /m/s/x
  intent(in) :: x
  !REF: /m/s/y
  intent(inout) :: yyy
 contains
  !DEF: /m/s/ss PURE Subprogram
  pure subroutine ss
  end subroutine
 end subroutine
 !REF: /m/f
 !DEF: /m/f/x ALLOCATABLE ObjectEntity REAL(4)
 recursive pure function f() result(x)
  !REF: /m/f/x
  real, allocatable :: x
  !REF: /m/f/x
  x = 1.0
 end function
end module

! RUN: echo %t 1>&2;
! RUN: ${F18} -fget-definition 27 17 18 -fparse-only -fdebug-semantics %s > %t;
! RUN: ${F18} -fget-definition 29 20 23 -fparse-only -fdebug-semantics %s >> %t;
! RUN: ${F18} -fget-definition 41 3 4 -fparse-only -fdebug-semantics %s >> %t;
! RUN: ${F18} -fget-definition -fparse-only -fdebug-semantics %s >> %t 2>&1;
! RUN: cat %t | ${FileCheck} %s
! CHECK:x:.*getdefinition01.f90, 25, 21-22
! CHECK:yyy:.*getdefinition01.f90, 25, 24-27
! CHECK:x:.*getdefinition01.f90, 39, 24-25
! CHECK:Invalid argument to -fget-definitions
