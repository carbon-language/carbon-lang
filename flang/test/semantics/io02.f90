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

  integer :: unit10 = 10
  integer :: unit11 = 11

  integer(kind=1) :: stat1
  integer(kind=8) :: stat8

  character(len=55) :: msg

  close(unit10)
  close(unit=unit11, err=9, iomsg=msg, iostat=stat1)
  close(12, status='Keep')

  close(iostat=stat8, 11) ! nonstandard

  !ERROR: CLOSE statement must have a UNIT number specifier
  close(iostat=stat1)

  !ERROR: duplicate UNIT specifier
  close(13, unit=14, err=9)

  !ERROR: duplicate ERR specifier
  close(err=9, unit=15, err=9, iostat=stat8)

  !ERROR: invalid STATUS value 'kept'
  close(status='kept', unit=16)

  !ERROR: invalid STATUS value 'old'
  close(status='old', unit=17)

9 continue
end
