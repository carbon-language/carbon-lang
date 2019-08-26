! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

! Statement functions

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/f Subprogram INTEGER(4)
 !DEF: /p1/i ObjectEntity INTEGER(4)
 !DEF: /p1/j ObjectEntity INTEGER(4)
 integer f, i, j
 !REF: /p1/f
 !REF: /p1/i
 !DEF: /p1/f/i ObjectEntity INTEGER(4)
 f(i) = i + 1
 !REF: /p1/j
 !REF: /p1/f
 j = f(2)
end program
