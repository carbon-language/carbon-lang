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

subroutine s1
  integer x
  block
    import, none
    !ERROR: 'x' from host scoping unit is not accessible due to IMPORT
    x = 1
  end block
end

subroutine s2
  block
    import, none
    !ERROR: 'y' from host scoping unit is not accessible due to IMPORT
    y = 1
  end block
end

subroutine s3
  integer j
  block
    import, only: j
    type t
      !ERROR: 'i' from host scoping unit is not accessible due to IMPORT
      real :: x(10) = [(i, i=1,10)]
    end type
  end block
end subroutine
