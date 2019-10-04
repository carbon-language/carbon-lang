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

* KWM expansion is before token pasting due to fixed-form space removal
      integer, parameter :: IKWM2Z = 777
#define KWM KWM2
      integer :: res
      res = I KWM Z
      if (res .eq. 777) then
        print *, 'pp025.F pass'
      else
        print *, 'pp025.F FAIL: ', res
      end if
      end
