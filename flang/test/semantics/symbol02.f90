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

! Test host association in module subroutine and internal subroutine.

!DEF: /m Module
module m
 !DEF: /m/t PUBLIC DerivedType
 type :: t
 end type
 !REF: /m/t
 !DEF: /m/x PUBLIC ObjectEntity TYPE(t)
 type(t) :: x
 interface
  !DEF: /m/s3 MODULE, PUBLIC Subprogram
  !DEF: /m/s3/y ObjectEntity TYPE(t)
  module subroutine s3(y)
   !REF: /m/t
   !REF: /m/s3/y
   type(t) :: y
  end subroutine
 end interface
contains
 !DEF: /m/s PUBLIC Subprogram
 subroutine s
  !REF: /m/t
  !DEF: /m/s/y ObjectEntity TYPE(t)
  type(t) :: y
  !REF: /m/s/y
  !REF: /m/x
  y = x
  !DEF: /m/s/s PUBLIC Subprogram
  call s
 contains
  !DEF: /m/s/s2 Subprogram
  subroutine s2
   !REF: /m/x
   !REF: /m/s/y
   !REF: /m/t
   !REF: /m/s/s
   import, only: x, y, t, s
   !REF: /m/t
   !DEF: /m/s/s2/z ObjectEntity TYPE(t)
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
