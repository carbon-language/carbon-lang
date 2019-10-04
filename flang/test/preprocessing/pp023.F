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

* KWM NOT expanded in 9HHOLLERITH literal
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
      ch = 3HKWM
      if (ch .eq. 'KWM') then
        print *, 'pp023.F pass'
      else
        print *, 'pp023.F FAIL: ', ch
      end if
      end
