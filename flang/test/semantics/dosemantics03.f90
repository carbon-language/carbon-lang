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

! Issue 458 -- semantic checks for a normal DO loop.  The DO variable
! and the initial, final, and step expressions must be INTEGER if the
! options for standard conformance and turning warnings into errors
! are both in effect.  This test turns on the options for standards
! conformance and turning warnings into errors.  This produces error
! messages for the cases where REAL and DOUBLE PRECISION variables
! and expressions are used in the DO controls.
!
! This test is just like dosemantics04.f90 but with the options
! to produce error messages when using REAL and DOUBLE PRECISION DO
! loop controls.

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
!OPTIONS: -Mstandard -Werror

PROGRAM do_issue_458
  IMPLICIT NONE
  INTEGER :: ivar
  REAL :: rvar
  DOUBLE PRECISION :: dvar
  LOGICAL :: lvar
  COMPLEX :: cvar
  CHARACTER :: chvar
  INTEGER, DIMENSION(3) :: avar
  TYPE derived
    REAL :: first
    INTEGER :: second
  END TYPE derived
  TYPE(derived) :: devar
  INTEGER, POINTER :: pivar
  REAL, POINTER :: prvar
  DOUBLE PRECISION, POINTER :: pdvar
  LOGICAL, POINTER :: plvar

! DO variables
! INTEGER DO variable
  DO ivar = 1, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! REAL DO variable
!ERROR: DO controls should be INTEGER
  DO rvar = 1, 10, 3
    PRINT *, "rvar is: ", rvar
  END DO

! DOUBLE PRECISISON DO variable
!ERROR: DO controls should be INTEGER
  DO dvar = 1, 10, 3
    PRINT *, "dvar is: ", dvar
  END DO

! Pointer to INTEGER DO variable
  ALLOCATE(pivar)
  DO pivar = 1, 10, 3
    PRINT *, "pivar is: ", pivar
  END DO

! Pointer to REAL DO variable
  ALLOCATE(prvar)
!ERROR: DO controls should be INTEGER
  DO prvar = 1, 10, 3
    PRINT *, "prvar is: ", prvar
  END DO

! Pointer to DOUBLE PRECISION DO variable
  ALLOCATE(pdvar)
!ERROR: DO controls should be INTEGER
  DO pdvar = 1, 10, 3
    PRINT *, "pdvar is: ", pdvar
  END DO

! CHARACTER DO variable
!ERROR: DO controls should be INTEGER
  DO chvar = 1, 10, 3
    PRINT *, "chvar is: ", chvar
  END DO

! LOGICAL DO variable
!ERROR: DO controls should be INTEGER
  DO lvar = 1, 10, 3
    PRINT *, "lvar is: ", lvar
  END DO

! COMPLEX DO variable
!ERROR: DO controls should be INTEGER
  DO cvar = 1, 10, 3
    PRINT *, "cvar is: ", cvar
  END DO

! Derived type DO variable
!ERROR: DO controls should be INTEGER
  DO devar = 1, 10, 3
    PRINT *, "devar is: ", devar
  END DO

! Pointer to LOGICAL DO variable
  ALLOCATE(plvar)
!ERROR: DO controls should be INTEGER
  DO plvar = 1, 10, 3
    PRINT *, "plvar is: ", plvar
  END DO

! Initial expressions
! REAL initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = rvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! DOUBLE PRECISION initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = dvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to INTEGER initial expression
  DO ivar = pivar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to REAL initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = prvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to DOUBLE PRECISION initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = pdvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! LOGICAL initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = lvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! COMPLEX initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = cvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Derived type initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = devar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to LOGICAL initial expression
!ERROR: DO controls should be INTEGER
  DO ivar = plvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Final expression
! REAL final expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, rvar, 3
    PRINT *, "ivar is: ", ivar
  END DO

! DOUBLE PRECISION final expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, dvar, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to INTEGER final expression
  DO ivar = 1, pivar, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to REAL final expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, prvar, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to DOUBLE PRECISION final expression
!ERROR: DO controls should be INTEGER
  DO ivar = pdvar, 10, 3
    PRINT *, "ivar is: ", ivar
  END DO

! COMPLEX final expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, cvar, 3
    PRINT *, "ivar is: ", ivar
  END DO

! Step expression
! REAL step expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, 10, rvar
    PRINT *, "ivar is: ", ivar
  END DO

! DOUBLE PRECISION step expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, 10, dvar
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to INTEGER step expression
  DO ivar = 1, 10, pivar
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to REAL step expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, 10, prvar
    PRINT *, "ivar is: ", ivar
  END DO

! Pointer to DOUBLE PRECISION step expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, 10, pdvar
    PRINT *, "ivar is: ", ivar
  END DO

! COMPLEX Step expression
!ERROR: DO controls should be INTEGER
  DO ivar = 1, 10, cvar
    PRINT *, "ivar is: ", ivar
  END DO

! Array DO variable
!ERROR: Must be a scalar value, but is a rank-1 array
  DO avar = 1, 10, 3
    PRINT *, "plvar is: ", plvar
  END DO

END PROGRAM do_issue_458
