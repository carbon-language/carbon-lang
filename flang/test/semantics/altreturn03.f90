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

! Check for various alt return error conditions

       SUBROUTINE TEST (N, *, *)
       REAL :: R
       COMPLEX :: Z
       INTEGER, DIMENSION(2) :: B
       IF ( N .EQ. 0 ) RETURN
       IF ( N .EQ. 1 ) RETURN 1
       IF ( N .EQ. 2 ) RETURN 2
       IF ( N .EQ. 3 ) RETURN 3
       IF ( N .EQ. 3 ) RETURN N
       IF ( N .EQ. 3 ) RETURN N * N
       IF ( N .EQ. 3 ) RETURN B(N)
       IF ( N .EQ. 3 ) RETURN B
       IF ( N .EQ. 3 ) RETURN R
       IF ( N .EQ. 3 ) RETURN Z
       RETURN 2
       END
