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

! Use-association with VOLATILE or ASYNCHRONOUS

module m1
  real x
  integer y
  volatile z
contains
end

module m2
  use m1
  volatile x
  asynchronous y
end

!Expect: m1.mod
!module m1
!real(4)::x
!integer(4)::y
!real(4),volatile::z
!end

!Expect: m2.mod
!module m2
!use m1,only:x
!use m1,only:y
!use m1,only:z
!volatile::x
!asynchronous::y
!end
