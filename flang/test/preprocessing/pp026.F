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

* ## token pasting works in FLM
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
      integer, parameter :: KWM = 668, KWM21 = 111, KWM1 = 669
#define KWM1 111
#define KWM2 33
#define KWM KWM2
#define IFLM(x) ((x##1)+6##6##6)
      integer :: res
      res = IFLM(KWM)
      if (res .eq. 777) then
        print *, 'pp026.F pass'
      else
        print *, 'pp026.F FAIL: ', res
      end if
      end
