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
! CHECK: label '50' was not found
! CHECK: label '55' is not in scope
! CHECK: label '70' is not an action stmt

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 50
  end if
6 n = m
end subroutine sub00

subroutine sub01(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 55
  else
55   continue
  end if
60 n = m
end subroutine sub01

subroutine sub02(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 70
  else
     return
  end if
70 FORMAT (1x,i6)
end subroutine sub02
