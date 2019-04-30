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

! C1131 -- check valid and invalid DO loop naming

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK-NOT: 34:.*name required but missing
! CHECK-NOT: 34:.*DO construct name unexpected
! CHECK-NOT: 32:.*should be
! CHECK-NOT: 32:.*unnamed DO statement
! CHECK: 39:.*name required but missing
! CHECK: 37:.*should be
! CHECK: 44:.*DO construct name unexpected
! CHECK: 42:.*unnamed DO statement
PROGRAM C1131

  IMPLICIT NONE
  ! Test R1121 label-do-stmt

  ! Valid construct
  validDo: DO WHILE (.true.)
      PRINT *, "Hello"
    END DO ValidDo

  ! Missing name on END DO
  missingEndDo: DO WHILE (.true.)
      PRINT *, "Hello"
    END DO

  ! Missing name on DO
  DO WHILE (.true.)
      PRINT *, "Hello"
    END DO missingDO

END PROGRAM C1131
