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

!OPTIONS: -Mstandard

  write(*, '(B0)')
  write(*, '(B3)')

  !WARNING: Expected 'B' edit descriptor 'w' value
  write(*, '(B)')

  !WARNING: Expected 'EN' edit descriptor 'w' value
  !WARNING: Non-standard '$' edit descriptor
  write(*, '(EN,$)')

  !WARNING: Expected 'G' edit descriptor 'w' value
  write(*, '(3G)')

  !WARNING: Non-standard '\' edit descriptor
  write(*,'(A, \)') 'Hello'

  !WARNING: 'X' edit descriptor must have a positive position value
  write(*, '(X)')

  !WARNING: Legacy 'H' edit descriptor
  write(*, '(3Habc)')

  !WARNING: 'X' edit descriptor must have a positive position value
  !WARNING: Expected ',' or ')' in format expression
  !WARNING: 'X' edit descriptor must have a positive position value
  write(*,'(XX)')

  !WARNING: Expected ',' or ')' in format expression
  write(*,'(RZEN8.2)')

  !WARNING: Expected ',' or ')' in format expression
  write(*,'(3P7I2)')

  !WARNING: Expected ',' or ')' in format expression
  write(*,'(5X i3)')
end
