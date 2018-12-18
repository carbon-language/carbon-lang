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

!DEF: /m1 Module
module m1
contains
 !DEF: /m1/foo_complex PUBLIC Subprogram
 !DEF: /m1/foo_complex/z ObjectEntity COMPLEX(4)
 subroutine foo_complex (z)
  !REF: /m1/foo_complex/z
  complex z
 end subroutine
end module
!DEF: /m2 Module
module m2
 !REF: /m1
 use :: m1
 !DEF: /m2/foo PUBLIC Generic
 interface foo
  !DEF: /m2/foo_int PUBLIC Subprogram
  module procedure :: foo_int
  !DEF: /m2/foo_real EXTERNAL, PUBLIC Subprogram
  procedure :: foo_real
  !DEF: /m2/foo_complex PUBLIC Use
  procedure :: foo_complex
 end interface
 interface
  !REF: /m2/foo_real
  !DEF: /m2/foo_real/r ObjectEntity REAL(4)
  subroutine foo_real (r)
   !REF: /m2/foo_real/r
   real r
  end subroutine
 end interface
contains
 !REF: /m2/foo_int
 !DEF: /m2/foo_int/i ObjectEntity INTEGER(4)
 subroutine foo_int (i)
  !REF: /m2/foo_int/i
  integer i
 end subroutine
end module
