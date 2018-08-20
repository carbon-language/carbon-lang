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

! negative test -- invalid labels, out of range

! RUN: f18 < %s | FileCheck %s
! CHECK: label '10' is not in scope
! CHECK: label '20' was not found
! CHECK: label '30' is not an action stmt
! CHECK: label '60' was not found

subroutine sub00(n,m)
30 format (i6,f6.2)
  if (n .eq. m) then
10   print *,"equal"
  end if
  call sub01(n,*10,*20,*30)
  write (*,60) n, m
end subroutine sub00
