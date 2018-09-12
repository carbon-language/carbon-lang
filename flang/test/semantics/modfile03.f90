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

! Check modfile generation with use-association.

module m1
  integer :: x1
  integer, private :: x2
end

module m2
  use m1
  integer :: y1
end

module m3
  use m2, z1 => x1
end

module m4
  use m1
  use m2
end

!Expect: m1.mod
!module m1
!integer(4)::x1
!integer(4),private::x2
!end

!Expect: m2.mod
!module m2
!use m1,only:x1
!integer(4)::y1
!end

!Expect: m3.mod
!module m3
!use m2,only:y1
!use m2,only:z1=>x1
!end

!Expect: m4.mod
!module m4
!use m1,only:x1
!use m2,only:y1
!end
