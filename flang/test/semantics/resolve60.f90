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

! Testing 7.6 enum

  ! OK
  enum, bind(C)
    enumerator :: red, green
    enumerator blue, pink
    enumerator yellow
    enumerator :: purple = 2
  end enum

  integer(yellow) anint4

  enum, bind(C)
    enumerator :: square, cicrle
    !ERROR: 'square' is already declared in this scoping unit
    enumerator square
  end enum

  dimension :: apple(4)
  real :: peach

  enum, bind(C)
    !ERROR: 'apple' is already declared in this scoping unit
    enumerator :: apple
    enumerator :: pear
    !ERROR: 'peach' is already declared in this scoping unit
    enumerator :: peach
    !ERROR: 'red' is already declared in this scoping unit
    enumerator :: red
  end enum

  enum, bind(C)
    !ERROR: Must be a constant value
    enumerator :: wrong = 0/0
  end enum

end
