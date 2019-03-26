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

! Check that only labels are allowed in arithmetic if statements.
! TODO: Revisit error message "expected 'ASSIGN'" etc.
! TODO: Revisit error message "expected one of '0123456789'"

! TODO: BUG: Note that labels 500 and 600 do not exist and
! ought to be flagged as errors. This oversight may be the
! result of disabling semantic checking after syntax errors.

if ( A ) 500, 600, 600
100 CONTINUE
200 CONTINUE
300 CONTINUE

!ERROR: expected 'ASSIGN'
!ERROR: expected 'ALLOCATE ('
!ERROR: expected '=>'
!ERROR: expected '('
!ERROR: expected '='
if ( B ) A, 101, 301
101 CONTINUE
201 CONTINUE
301 CONTINUE

!ERROR: expected one of '0123456789'
if ( B ) 102, A, 302
102 CONTINUE
202 CONTINUE
302 CONTINUE

!ERROR: expected one of '0123456789'
if ( B ) 103, 103, A
103 CONTINUE
203 CONTINUE
303 CONTINUE

END
