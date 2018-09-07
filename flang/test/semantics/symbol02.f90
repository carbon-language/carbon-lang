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

! Test host association in module subroutine and internal subroutine.

!DEF: /m Module
module m
 !DEF: /m/t PUBLIC DerivedType
 type :: t
 end type
 !DEF: /m/x PUBLIC ObjectEntity TYPE(t)
 !REF: /m/t
 type(t) :: x
contains
 !DEF: /m/s PUBLIC Subprogram
 subroutine s
  !DEF: /m/s/y ObjectEntity TYPE(t)
  !REF: /m/t
  type(t) :: y
  !REF: /m/s/y
  !REF: /m/x
  y = x
  !REF: /m/s/s
  call s
 contains
  !DEF: /m/s/s2 Subprogram
  subroutine s2
   !REF: /m/x
   !REF: /m/s/y
   !REF: /m/t
   !REF: /m/s/s
   import, only: x, y, t, s
   !DEF: /m/s/s2/z ObjectEntity TYPE(t)
   !REF: /m/t
   type(t) :: z
   !REF: /m/s/s2/z
   !REF: /m/x
   z = x
   !REF: /m/s/s2/z
   !REF: /m/s/y
   z = y
   !REF: /m/s/s
   call s
  end subroutine
 end subroutine
end module
