! Copyright (c) 2019, ARM Ltd.  All rights reserved.
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

! C1108  --  Save statement in a BLOCK construct shall not conatin a
!            saved-entity-list that does not specify a common-block-name

program  main
  integer x, y, z
  real r, s, t
  common /argmnt2/ r, s, t
  !ERROR: 'argmnt1' appears as a COMMON block in a SAVE statement but not in a COMMON statement
  save /argmnt1/
  block
    !ERROR: SAVE statement in BLOCK construct may not contain a common block name 'argmnt2'
    save /argmnt2/
  end block
end program
