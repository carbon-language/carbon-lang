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

! C1131, C1133 -- check valid and invalid DO loop naming
! C1131 (R1119) If the do-stmt of a do-construct specifies a do-construct-name,
! the corresponding end-do shall be an end-do-stmt specifying the same
! do-construct-name. If the do-stmt of a do-construct does not specify a
! do-construct-name, the corresponding end-do shall not specify a
! do-construct-name.
!
! C1133 (R1119) If the do-stmt is a label-do-stmt, the corresponding end-do
! shall be identified with the same label.

subroutine s1()
  implicit none
  ! Valid construct
  validdo: do while (.true.)
      print *, "hello"
      cycle validdo
      print *, "Weird to get here"
    end do validdo

  validdo: do while (.true.)
      print *, "Hello"
    end do validdo

  ! Missing name on initial DO
  do while (.true.)
      print *, "Hello"
!ERROR: DO construct name unexpected
    end do formerlabelmissing

  dolabel: do while (.true.)
      print *, "Hello"
!ERROR: DO construct name mismatch
    end do differentlabel

  dowithcycle: do while (.true.)
      print *, "Hello"
!ERROR: CYCLE construct-name is not in scope
      cycle validdo
    end do dowithcycle

end subroutine s1
