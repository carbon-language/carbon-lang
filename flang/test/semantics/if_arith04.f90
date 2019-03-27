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

! Make sure arithmetic if expressions are non-complex numeric exprs.

INTEGER I
COMPLEX Z
LOGICAL L
INTEGER, DIMENSION (2) :: B

if ( I ) 100, 200, 300
100 CONTINUE
200 CONTINUE
300 CONTINUE

!ERROR: ARITHMETIC IF expression must not be a COMPLEX expression
if ( Z ) 101, 201, 301
101 CONTINUE
201 CONTINUE
301 CONTINUE

!ERROR: ARITHMETIC IF expression must be a numeric expression
if ( L ) 102, 202, 302
102 CONTINUE
202 CONTINUE
302 CONTINUE

!ERROR: ARITHMETIC IF expression must be a scalar expression
if ( B ) 103, 203, 303
103 CONTINUE
203 CONTINUE
303 CONTINUE

END
