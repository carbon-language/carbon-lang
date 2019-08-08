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

!C1132
! If the do-stmt is a nonlabel-do-stmt, the corresponding end-do shall be an
! end-do-stmt.
subroutine s1()
  do while (.true.)
    print *, "Hello"
  continue
!ERROR: expected 'END DO'
end subroutine s1
