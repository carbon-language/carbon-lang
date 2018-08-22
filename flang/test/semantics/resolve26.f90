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
  interface
    module subroutine s()
    end subroutine
  end interface
end

module m2
  interface
    module subroutine s()
    end subroutine
  end interface
end

submodule(m1) s1
end

!ERROR: Cannot find module file for submodule 's1' of module 'm2'
submodule(m2:s1) s2
end

!ERROR: Cannot find module file for 'm3'
submodule(m3:s1) s3
end
