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

! Test section subscript
program p1
  real :: a(10,10)
  real :: b(5,5)
  real :: c
  integer :: n
  n = 2
  b = a(1:10:n,1:n+3)
end

! Test substring
program p2
  character :: a(10)
  character :: b(5)
  integer :: n
  n = 3
  b = a(n:7)
  b = a(n+3:)
  b = a(:n+2)
  a(n:7) = b
  a(n+3:) = b
  a(:n+2) = b
end

! Test pointer assignment with bounds
program p3
  integer, pointer :: a(:,:)
  integer, target :: b(2,2)
  integer :: n
  n = 2
  a(n:,n:) => b
  a(1:n,1:n) => b
end
