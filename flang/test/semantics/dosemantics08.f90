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

! C1138 -- 
! A branch (11.2) within a DO CONCURRENT construct shall not have a branch
! target that is outside the construct.

subroutine s1()
  do concurrent (i=1:10)
!ERROR: control flow escapes from DO CONCURRENT
    goto 99
  end do

99 print *, "Hello"

end subroutine s1
