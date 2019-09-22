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
!
! C1134 A CYCLE statement must be within a DO construct
!
! C1166 An EXIT statement must be within a DO construct

subroutine s1()
! this one's OK
  do i = 1,10
    cycle
  end do

! this one's OK
  do i = 1,10
    exit
  end do

! all of these are OK
  outer: do i = 1,10
    cycle
    inner: do j = 1,10
      cycle
    end do inner
    cycle
  end do outer

!ERROR: No matching construct for CYCLE statement
  cycle

!ERROR: No matching construct for EXIT statement
  exit

!ERROR: No matching construct for CYCLE statement
  if(.true.) cycle

!ERROR: No matching construct for EXIT statement
  if(.true.) exit

end subroutine s1
