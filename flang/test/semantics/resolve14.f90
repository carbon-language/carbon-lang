! Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

module m1
  integer :: x
  integer :: y
  integer :: z
end
module m2
  real :: y
  real :: z
  real :: w
end

use m1, xx => x, y => z
use m2
volatile w
!ERROR: Cannot change CONTIGUOUS attribute on use-associated 'w'
contiguous w
!ERROR: 'z' is use-associated from module 'm2' and cannot be re-declared
integer z
!ERROR: Reference to 'y' is ambiguous
y = 1
end
