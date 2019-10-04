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

* fixed-form clipping done before KWM expansion on source line
      integer, parameter :: KW = 777
#define KWM 666
      integer :: res
* 'M' is in column 73
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
      res =                                                           KWM
      if (res .eq. 777) then
        print *, 'pp028.F pass'
      else
        print *, 'pp028.F FAIL: ', res
      end if
      end
