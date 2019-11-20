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

! Also, check that the step expressions are not zero.  This is prohibited by
! Section 11.1.7.4.1, paragraph 1.

SUBROUTINE do_concurrent_c1121(i,n)
  IMPLICIT NONE
  INTEGER :: i, n, flag
!ERROR: Concurrent-header mask expression cannot reference an impure procedure
  DO CONCURRENT (i = 1:n, random() < 3)
    flag = 3
  END DO

  CONTAINS
    IMPURE FUNCTION random() RESULT(i)
      INTEGER :: i
      i = 35
    END FUNCTION random
END SUBROUTINE do_concurrent_c1121

SUBROUTINE s1()
  INTEGER, PARAMETER :: constInt = 0

  ! Warn on this one for backwards compatibility
  DO 10 I = 1, 10, 0
  10 CONTINUE

  ! Warn on this one for backwards compatibility
  DO 20 I = 1, 10, 5 - 5
  20 CONTINUE

  ! Error, no compatibility requirement for DO CONCURRENT
!ERROR: DO CONCURRENT step expression should not be zero
  DO CONCURRENT (I = 1 : 10 : 0)
  END DO

  ! Error, this time with an integer constant
!ERROR: DO CONCURRENT step expression should not be zero
  DO CONCURRENT (I = 1 : 10 : constInt)
  END DO
end subroutine s1
