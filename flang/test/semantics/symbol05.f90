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

! Explicit and implicit entities in blocks

!DEF: /s1 Subprogram
subroutine s1
 !DEF: /s1/x Entity INTEGER
 integer x
 block
  !DEF: /s1/Block1/y Entity INTEGER
  integer y
  !REF: /s1/x
  x = 1
  !REF: /s1/Block1/y
  y = 2.0
 end block
 block
  !DEF: /s1/Block2/y Entity REAL
  real y
  !REF: /s1/Block2/y
  y = 3.0
 end block
end subroutine

!DEF: /s2 Subprogram
subroutine s2
 implicit integer(w-x)
 block
  !DEF: /s2/x (implicit) ObjectEntity INTEGER
  x = 1
  !DEF: /s2/y (implicit) ObjectEntity REAL
  y = 2
 end block
contains
 !DEF: /s2/s Subprogram
 subroutine s
  !REF: /s2/x
  x = 1
  !DEF: /s2/s/w (implicit) ObjectEntity INTEGER
  w = 1
 end subroutine
end subroutine

!DEF: /s3 Subprogram
subroutine s3
 block
  !DEF: /s3/Block1/t DerivedType
  type :: t
   !DEF: /s3/i (implicit) ObjectEntity INTEGER
   !DEF: /s3/Block1/t/x ObjectEntity REAL
   real :: x(10) = [(i, i=1,10)]
  end type
 end block
end subroutine

!DEF: /s4 Subprogram
subroutine s4
 implicit integer(x)
 interface
  !DEF: /s4/s EXTERNAL Subprogram
  !DEF: /s4/s/x (implicit) ObjectEntity REAL
  !DEF: /s4/s/y (implicit) ObjectEntity INTEGER
  subroutine s (x, y)
   implicit integer(y)
  end subroutine
 end interface
end subroutine

!DEF: /s5 Subprogram
subroutine s5
 block
  !DEF: /s5/Block1/x (implicit) ObjectEntity REAL
  dimension :: x(2)
  block
   !DEF: /s5/Block1/Block1/x (implicit) ObjectEntity REAL
   dimension :: x(3)
  end block
 end block
 !DEF: /s5/x (implicit) ObjectEntity REAL
 x = 1.0
end subroutine
