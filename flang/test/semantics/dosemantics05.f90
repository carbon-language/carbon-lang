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

! C1127 -- The DEFAULT (NONE) locality-spec shall not appear more than once in a
! given concurrent-locality.
PROGRAM dosemantics05
  IMPLICIT NONE
  INTEGER :: ivar

! This one works
  DO CONCURRENT (ivar = 1:10) DEFAULT (NONE)
    PRINT *, "ivar is: ", ivar
  END DO

!ERROR: only one DEFAULT(NONE) may appear
  DO CONCURRENT (ivar = 1:10) DEFAULT (NONE) DEFAULT (NONE)
    PRINT *, "ivar is: ", ivar
  END DO

END PROGRAM dosemantics05
