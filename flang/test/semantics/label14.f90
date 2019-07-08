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

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: A DO loop should terminate with an END DO or CONTINUE inside its scope
! CHECK: A DO loop should terminate with an END DO or CONTINUE inside its scope
! CHECK: A DO loop should terminate with an END DO or CONTINUE inside its scope

  do 1 j=1,2
    if (.true.) then
1   end if
  do 2 k=1,2
    do i=3,4
      print*, i+k
2    end do
  do 3 l=1,2
    select case (l)
    case default
      print*, "default"
    case (1)
      print*, "start"
3    end select
  end
