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

! Test that intent-stmt and subprogram prefix and suffix are resolved.

!DEF: /m Module
module m
 !DEF: /m/f PRIVATE, PURE, RECURSIVE Subprogram
 private :: f
contains
 !DEF: /m/s BIND(C), PUBLIC, PURE Subprogram
 !DEF: /m/s/x INTENT(IN) (implicit) ObjectEntity REAL(4)
 !DEF: /m/s/y INTENT(INOUT) (implicit) ObjectEntity REAL(4)
 pure subroutine s (x, y) bind(c)
  intent(in) :: x
  intent(inout) :: y
 contains
  !DEF: /m/s/ss PURE Subprogram
  pure subroutine ss
  end subroutine
 end subroutine
 !DEF: /m/f PRIVATE, PURE, RECURSIVE Subprogram
 !DEF: /m/f/x ALLOCATABLE ObjectEntity REAL(4)
 recursive pure function f() result(x)
  real, allocatable :: x
  !REF: /m/f/x
  x = 1.0
 end function
end module
