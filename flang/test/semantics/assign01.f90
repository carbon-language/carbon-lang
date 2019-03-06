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

integer :: a1(10), a2(10)
logical :: m1(10), m2(5,5)
m1 = .true.
m2 = .false.
a1 = [((i),i=1,10)]
where (m1)
  a2 = 1
!ERROR: mask of ELSEWHERE statement is not conformable with the prior mask(s) in its WHERE construct
elsewhere (m2)
  a2 = 2
elsewhere
  a2 = 3
end where
end
