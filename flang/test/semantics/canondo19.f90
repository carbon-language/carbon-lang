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

! Check that if there is a label or a name on an label-do-stmt,
! then it is not lost when rewriting it to an non-label-do-stmt.

! RUN: ${F18} -funparse-with-symbols -Mstandard %s 2>&1 | ${FileCheck} %s

! CHECK: end do
! CHECK: 2 do
! CHECK: mainloop: do
! CHECK: end do mainloop

! CHECK-NOT: do [1-9]

subroutine foo()
  do 1 i=1,2
    goto 2
1 continue
2 do 3 i=1,2
3 continue

  mainloop : do 4 i=1,100
    do j=1,20
      if (j==i) then
        ! cycle mainloop: TODO: fix invalid complaints that mainloop construct
        ! is not in scope.
      end if
    end do
4 end do mainloop
end subroutine
