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

! SELECTED_INT_KIND and SELECTED_REAL_KIND

module m1
  ! INTEGER(KIND=1)  handles  0 <= P < 3
  ! INTEGER(KIND=2)  handles  3 <= P < 5
  ! INTEGER(KIND=4)  handles  5 <= P < 10
  ! INTEGER(KIND=8)  handles 10 <= P < 19
  ! INTEGER(KIND=16) handles 19 <= P < 38
  integer, parameter :: intpvals(:) = [0, 2, 3, 4, 5, 9, 10, 18, 19, 38, 39]
  integer, parameter :: intpkinds(:) = &
    [(selected_int_kind(intpvals(j)),j=1,size(intpvals))]
  logical, parameter :: ipcheck = &
    all([1, 1, 2, 2, 4, 4, 8, 8, 16, 16, -1] == intpkinds)
  ! REAL(KIND=2)  handles  0 <= P < 4  (if available)
  ! REAL(KIND=4)  handles  4 <= P < 7
  ! REAL(KIND=8)  handles  7 <= P < 16
  ! REAL(KIND=10) handles 16 <= P < 19 (if available; ifort is KIND=16)
  ! REAL(KIND=16) handles 19 <= P < 34 (32 with Power double/double)
  integer, parameter :: realpvals(:) = [0, 3, 4, 6, 7, 15, 16, 18, 19, 33, 34]
  integer, parameter :: realpkinds(:) = &
    [(selected_real_kind(realpvals(j),0),j=1,size(realpvals))]
  logical, parameter :: realpcheck = &
    all([2, 2, 4, 4, 8, 8, 10, 10, 16, 16, -1] == realpkinds)
  ! REAL(KIND=2)  handles  0 <= R < 5 (if available)
  ! REAL(KIND=3)  handles  5 <= R < 38 (if available, same range as KIND=4)
  ! REAL(KIND=4)  handles  5 <= R < 38 (if no KIND=3)
  ! REAL(KIND=8)  handles 38 <= R < 308
  ! REAL(KIND=10) handles 308 <= R < 4932 (if available; ifort is KIND=16)
  ! REAL(KIND=16) handles 4932 <= R < 9864 (except Power double/double)
  integer, parameter :: realrvals(:) = &
    [0, 4, 5, 37, 38, 307, 308, 4931, 4932, 9863, 9864]
  integer, parameter :: realrkinds(:) = &
    [(selected_real_kind(0,realrvals(j)),j=1,size(realrvals))]
  logical, parameter :: realrcheck = &
    all([2, 2, 3, 3, 8, 8, 10, 10, 16, 16, -2] == realrkinds)
end module m1
!Expect: m1.mod
!module m1
!integer(4),parameter::intpvals(1_8:)=[Integer(4)::0_4,2_4,3_4,4_4,5_4,9_4,10_4,18_4,19_4,38_4,39_4]
!integer(4),parameter::intpkinds(1_8:)=[Integer(4)::1_4,1_4,2_4,2_4,4_4,4_4,8_4,8_4,16_4,16_4,-1_4]
!logical(4),parameter::ipcheck=.true._4
!integer(4),parameter::realpvals(1_8:)=[Integer(4)::0_4,3_4,4_4,6_4,7_4,15_4,16_4,18_4,19_4,33_4,34_4]
!integer(4),parameter::realpkinds(1_8:)=[Integer(4)::2_4,2_4,4_4,4_4,8_4,8_4,10_4,10_4,16_4,16_4,-1_4]
!logical(4),parameter::realpcheck=.true._4
!integer(4),parameter::realrvals(1_8:)=[Integer(4)::0_4,4_4,5_4,37_4,38_4,307_4,308_4,4931_4,4932_4,9863_4,9864_4]
!integer(4),parameter::realrkinds(1_8:)=[Integer(4)::2_4,2_4,3_4,3_4,8_4,8_4,10_4,10_4,16_4,16_4,-2_4]
!logical(4),parameter::realrcheck=.true._4
!end
