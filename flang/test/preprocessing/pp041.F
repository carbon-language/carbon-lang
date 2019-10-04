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

* use KWM expansion as continuation indicators
#define KWM 0
#define KWM2 1
      integer :: j
      j = 666
     KWM j = j + 1
     KWM2 11
      if (j .eq. 777) then
        print *, 'pp041.F pass'
      else
        print *, 'pp041.F FAIL', j
      end if
      end
