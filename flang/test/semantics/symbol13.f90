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

! Old-style "*length" specifiers (R723)

!DEF: /f1 Subprogram CHARACTER(1_8,1)
!DEF: /f1/x1 INTENT(IN) ObjectEntity CHARACTER(2_4,1)
!DEF: /f1/x2 INTENT(IN) ObjectEntity CHARACTER(3_8,1)
character*1 function f1(x1, x2)
 !DEF: /f1/n PARAMETER ObjectEntity INTEGER(4)
 integer, parameter :: n = 2
 !REF: /f1/n
 !REF: /f1/x1
 !REF: /f1/x2
 !DEF: /len INTRINSIC ProcEntity
 character*(n), intent(in) :: x1, x2*(len(x1)+1)
 !DEF: /f1/t DerivedType
 type :: t
  !REF: /len
  !REF: /f1/x2
  !DEF: /f1/t/c1 ObjectEntity CHARACTER(4_8,1)
  !DEF: /f1/t/c2 ObjectEntity CHARACTER(6_8,1)
  character*(len(x2)+1) :: c1, c2*6
 end type t
end function f1
