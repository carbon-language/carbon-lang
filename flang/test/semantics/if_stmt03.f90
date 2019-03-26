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

! Check that non-logical expressions are not allowed.
! Check that non-scalar expressions are not allowed.
! TODO: Insure all non-logicals are prohibited.

LOGICAL, DIMENSION (2) :: B

!ERROR: Expected a LOGICAL expression
IF (A) A = LOG (A)
!ERROR: Expected a scalar LOGICAL expression
IF (B) A = LOG (A)

END
