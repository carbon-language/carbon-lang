// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Defines the macro RTNAME(n) which decorates the external name of a runtime
// library function or object with extra characters so that it
// (a) is not in the user's name space,
// (b) doesn't conflict with other libraries, and
// (c) prevents incompatible versions of the runtime library from linking
//
// The value of REVISION should not be changed until/unless the API to the
// runtime library must change in some way that breaks backward compatibility.

#ifndef RTNAME
#define PREFIX _Fortran
#define REVISION A
#define RTNAME(name) PREFIX##REVISION##name
#endif
