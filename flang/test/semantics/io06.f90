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

  character(kind=1,len=100) msg1
  character(kind=2,len=200) msg2
  integer(1) stat1
  integer(2) stat2
  integer(8) stat8

  open(10)

  backspace(10)
  backspace(10, iomsg=msg1, iostat=stat1, err=9)

  endfile(unit=10)
  endfile(iostat=stat2, err=9, unit=10, iomsg=msg1)

  rewind(10)
  rewind(iomsg=msg1, iostat=stat2, err=9, unit=10)

  flush(10)
  flush(iomsg=msg1, unit=10, iostat=stat8, err=9)

  wait(10)
  wait(99, id=id1, end=9, eor=9, err=9, iostat=stat1, iomsg=msg1)

  !ERROR: Duplicate UNIT specifier
  backspace(10, unit=11)

  !ERROR: Duplicate IOSTAT specifier
  endfile(iostat=stat2, err=9, unit=10, iostat=stat8, iomsg=msg1)

  !ERROR: REWIND statement must have a UNIT number specifier
  rewind(iostat=stat2)

  !ERROR: Duplicate ERR specifier
  !ERROR: Duplicate ERR specifier
  flush(err=9, unit=10, &
        err=9, &
        err=9)

  !ERROR: Duplicate ID specifier
  !ERROR: WAIT statement must have a UNIT number specifier
  wait(id=id2, eor=9, id=id3)

9 continue
end
