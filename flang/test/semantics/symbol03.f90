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

! Test host association in internal subroutine of main program.

!DEF: /main MainProgram
program main
 !DEF: /main/x Entity INTEGER
 integer x
 !REF: /main/s
 call s
contains
 !DEF: /main/s Subprogram
 subroutine s
  !DEF: /main/s/y (implicit) ObjectEntity REAL
  !REF: /main/x
  y = x
 end subroutine
end program
