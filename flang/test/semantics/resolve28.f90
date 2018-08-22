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

subroutine s
  type t
  end type
  interface
    subroutine s1
      import, none
      !ERROR: IMPORT,NONE must be the only IMPORT statement in a scope
      import, all
    end subroutine
    subroutine s2
      import :: t
      !ERROR: IMPORT,NONE must be the only IMPORT statement in a scope
      import, none
    end subroutine
    subroutine s3
      import, all
      !ERROR: IMPORT,ALL must be the only IMPORT statement in a scope
      import :: t
    end subroutine
    subroutine s4
      import :: t
      !ERROR: IMPORT,ALL must be the only IMPORT statement in a scope
      import, all
    end subroutine
  end interface
end

module m
  !ERROR: IMPORT is not allowed in a module scoping unit
  import, none
end

submodule(m) sub1
  import, all !OK
end

submodule(m) sub2
  !ERROR: IMPORT,NONE is not allowed in a submodule scoping unit
  import, none
end

function f
  !ERROR: IMPORT is not allowed in an external subprogram scoping unit
  import, all
end

!ERROR: IMPORT is not allowed in a main program scoping unit
import
end
