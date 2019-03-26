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



!ERROR: label '600' was not found
if ( A ) 100, 200, 600
100 CONTINUE
200 CONTINUE
300 CONTINUE

!ERROR: label '601' was not found
if ( A ) 101, 601, 301
101 CONTINUE
201 CONTINUE
301 CONTINUE

!ERROR: label '602' was not found
if ( A ) 602, 202, 302
102 CONTINUE
202 CONTINUE
302 CONTINUE

END
