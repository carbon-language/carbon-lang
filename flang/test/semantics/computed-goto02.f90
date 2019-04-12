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

! Check that computed goto express must be a scalar integer expression
! TODO: PGI, for example, accepts a float & converts the value to int.

REAL R
COMPLEX Z
LOGICAL L
INTEGER, DIMENSION (2) :: B

!ERROR: Must have INTEGER type, but is REAL(4)
GOTO (100) 1.5
!ERROR: Must have INTEGER type, but is LOGICAL(4)
GOTO (100) .TRUE.
!ERROR: Must have INTEGER type, but is REAL(4)
GOTO (100) R
!ERROR: Must have INTEGER type, but is COMPLEX(4)
GOTO (100) Z
!ERROR: Must be a scalar value, but is a rank-1 array
GOTO (100) B

100 CONTINUE

END
