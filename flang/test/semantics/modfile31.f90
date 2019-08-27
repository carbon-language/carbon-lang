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

! Test 7.6 enum values

module m1
  integer, parameter :: x(1) = [4]
  enum, bind(C)
    enumerator :: red, green
    enumerator blue
    enumerator yellow
    enumerator :: purple = 2
    enumerator :: brown
  end enum

  enum, bind(C)
    enumerator :: oak, beech = -rank(x)*x(1), pine, poplar = brown
  end enum

end

!Expect: m1.mod
!module m1
!integer(4),parameter::x(1_8:1_8)=[Integer(4)::4_4]
!integer(4),parameter::red=0_4
!integer(4),parameter::green=1_4
!integer(4),parameter::blue=2_4
!integer(4),parameter::yellow=3_4
!integer(4),parameter::purple=2_4
!integer(4),parameter::brown=3_4
!integer(4),parameter::oak=0_4
!integer(4),parameter::beech=-4_4
!integer(4),parameter::pine=-3_4
!integer(4),parameter::poplar=3_4
!end

