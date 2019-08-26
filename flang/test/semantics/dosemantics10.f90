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
!ERROR: CYCLE must be within a DO construct
  cycle

!ERROR: EXIT must be within a DO construct
  exit

!ERROR: CYCLE must be within a DO construct
  if(.true.) cycle

!ERROR: EXIT must be within a DO construct
  if(.true.) exit

end subroutine s1
