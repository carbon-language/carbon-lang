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
! CHECK: label '40' is not in scope
! CHECK: label '50' is not in scope (FIXME is that correct?)

subroutine sub00(n)
  GOTO (10,20,30) n
  if (n .eq. 1) then
10   print *, "xyz"
  end if
30 FORMAT (1x,i6)
end subroutine sub00

subroutine sub01(n)
  real n
  GOTO (40,50,60) n
  if (n .eq. 1) then
40   print *, "xyz"
50 end if
60 continue
end subroutine sub01
