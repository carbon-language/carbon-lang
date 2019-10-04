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

* \ newline allowed in name on KWM definition
      integer, parameter :: KWMC = 666
#define KWM\
C 777
      if (KWMC .eq. 777) then
        print *, 'pp034.F pass'
      else
        print *, 'pp034.F FAIL: ', KWMC
      end if
      end
