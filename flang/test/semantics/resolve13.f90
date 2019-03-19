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

module m1
  integer :: x
  integer, private :: y
  interface operator(.foo.)
    module procedure ifoo
  end interface
  interface operator(-)
    module procedure ifoo
  end interface
  interface operator(.priv.)
    module procedure ifoo
  end interface
  interface operator(*)
    module procedure ifoo
  end interface
  private :: operator(.priv.), operator(*)
contains
  integer function ifoo(x, y)
    integer, intent(in) :: x, y
  end
end

use m1, local_x => x
!ERROR: 'y' is PRIVATE in 'm1'
use m1, local_y => y
!ERROR: 'z' not found in module 'm1'
use m1, local_z => z
use m1, operator(.localfoo.) => operator(.foo.)
!ERROR: Operator '.bar.' not found in module 'm1'
use m1, operator(.localbar.) => operator(.bar.)

!ERROR: 'y' is PRIVATE in 'm1'
use m1, only: y
!ERROR: Operator '.priv.' is PRIVATE in 'm1'
use m1, only: operator(.priv.)
!ERROR: 'operator(*)' is PRIVATE in 'm1'
use m1, only: operator(*)
!ERROR: 'z' not found in module 'm1'
use m1, only: z
!ERROR: 'z' not found in module 'm1'
use m1, only: my_x => z
use m1, only: operator(.foo.)
!ERROR: Operator '.bar.' not found in module 'm1'
use m1, only: operator(.bar.)
use m1, only: operator(-) , ifoo
!ERROR: 'operator(+)' not found in module 'm1'
use m1, only: operator(+)

end
