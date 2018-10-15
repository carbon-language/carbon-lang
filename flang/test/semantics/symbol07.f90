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

!DEF: /main MainProgram
program main
 implicit complex(z)
 !DEF: /main/t DerivedType
 type :: t
  !DEF: /main/t/re ObjectEntity REAL(4)
  real :: re
  !DEF: /main/t/im ObjectEntity REAL(4)
  real :: im
 end type
 !DEF: /main/z1 ObjectEntity COMPLEX(4)
 complex z1
 !REF: /main/t
 !DEF: /main/w ObjectEntity TYPE(t)
 type(t) :: w
 !DEF: /main/x ObjectEntity REAL(4)
 !DEF: /main/y ObjectEntity REAL(4)
 real x, y
 !REF: /main/x
 !REF: /main/z1
 x = z1%re
 !REF: /main/y
 !REF: /main/z1
 y = z1%im
 !DEF: /main/z2 (implicit) ObjectEntity COMPLEX(4)
 !REF: /main/x
 z2%re = x
 !REF: /main/z2
 !REF: /main/y
 z2%im = y
 !REF: /main/x
 !REF: /main/w
 !REF: /main/t/re
 x = w%re
 !REF: /main/y
 !REF: /main/w
 !REF: /main/t/im
 y = w%im
end program
