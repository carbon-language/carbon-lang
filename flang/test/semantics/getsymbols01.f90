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

!DEF: /m Module
module m
 !DEF: /m/f PRIVATE, PURE, RECURSIVE Subprogram REAL(4)
 private :: f
contains
 !DEF: /m/s BIND(C), PUBLIC, PURE Subprogram
 !DEF: /m/s/x INTENT(IN) (implicit) ObjectEntity REAL(4)
 !DEF: /m/s/y INTENT(INOUT) (implicit) ObjectEntity REAL(4)
 pure subroutine s (x, y) bind(c)
  !REF: /m/s/x
  intent(in) :: x
  !REF: /m/s/y
  intent(inout) :: y
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

! RUN: ${F18} -fget-symbols-sources -fparse-only -fdebug-semantics %s 2>&1 | ${FileCheck} %s
! CHECK-ONCE:m:.*getsymbols01.f90, 18, 8-9
! CHECK-ONCE:f:.*getsymbols01.f90, 37, 26-27
! CHECK-ONCE:s:.*getsymbols01.f90, 25, 18-19
! CHECK-ONCE:ss:.*getsymbols01.f90, 32, 19-21
! CHECK-ONCE:x:.*getsymbols01.f90, 25, 21-22
! CHECK-ONCE:y:.*getsymbols01.f90, 25, 24-25
! CHECK-ONCE:x:.*getsymbols01.f90, 39, 24-25
