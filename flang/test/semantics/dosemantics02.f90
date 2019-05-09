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

! C1121 -- any procedure referenced in a concurrent header must be pure

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s

SUBROUTINE do_concurrent_c1121(i,n)
  IMPLICIT NONE
  INTEGER :: i, n, flag
!ERROR: concurrent-header mask expression cannot reference an impure procedure
  DO CONCURRENT (i = 1:n, random() < 3)
    flag = 3
  END DO

  CONTAINS
    IMPURE FUNCTION random() RESULT(i)
      INTEGER :: i
      i = 35
    END FUNCTION random
END SUBROUTINE do_concurrent_c1121
