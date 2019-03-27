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

! Check that a basic computed goto compiles

INTEGER, DIMENSION (2) :: B

GOTO (100) 1
GOTO (100) I
GOTO (100) I+J
GOTO (100) B(1)

GOTO (100, 200) 1
GOTO (100, 200) I
GOTO (100, 200) I+J
GOTO (100, 200) B(1)

GOTO (100, 200, 300) 1
GOTO (100, 200, 300) I
GOTO (100, 200, 300) I+J
GOTO (100, 200, 300) B(1)

100 CONTINUE
200 CONTINUE
300 CONTINUE
END
