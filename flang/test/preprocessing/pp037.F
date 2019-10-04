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

* fixed-form clipping NOT applied to #define
      integer, parameter :: KWM = 666
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
#define KWM                                                          7777
      if (KWM .eq. 777) then
        print *, 'pp037.F pass'
      else
        print *, 'pp037.F FAIL: ', KWM
      end if
      end
