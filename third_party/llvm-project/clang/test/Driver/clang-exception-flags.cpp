// We force the target to unknown because clang's default behavior for
// exception handling is target dependent.
// RUN: %clang -### -target unknown %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// DEFAULT: "-cc1" {{.*}} "-fcxx-exceptions" "-fexceptions"
//
// RUN: %clang -### -fexceptions %s 2>&1 | FileCheck %s -check-prefix=ON1
// ON1: "-cc1" {{.*}} "-fcxx-exceptions" "-fexceptions"
//
// RUN: %clang -### -fno-exceptions -fcxx-exceptions %s 2>&1 | FileCheck %s -check-prefix=ON2
// ON2: "-cc1" {{.*}} "-fcxx-exceptions" "-fexceptions"
//
// RUN: %clang -### -fno-cxx-exceptions -fexceptions %s 2>&1 | FileCheck %s -check-prefix=ON3
// ON3: "-cc1" {{.*}} "-fcxx-exceptions" "-fexceptions"
//
// RUN: %clang -### -fno-exceptions %s 2>&1 | FileCheck %s -check-prefix=OFF1
// OFF1-NOT: "-cc1" {{.*}} "-fcxx-exceptions"
//
// RUN: %clang -### -fno-cxx-exceptions %s 2>&1 | FileCheck %s -check-prefix=OFF2
// OFF2-NOT: "-cc1" {{.*}} "-fcxx-exceptions"
//
// RUN: %clang -### -fcxx-exceptions -fno-exceptions %s 2>&1 | FileCheck %s -check-prefix=OFF3
// OFF3-NOT: "-cc1" {{.*}} "-fcxx-exceptions"
//
// RUN: %clang -### -fexceptions -fno-cxx-exceptions %s 2>&1 | FileCheck %s -check-prefix=OFF4
// OFF4-NOT: "-cc1" {{.*}} "-fcxx-exceptions"
//
// RUN: %clang -### -target x86_64-scei-ps4 %s 2>&1 | FileCheck %s -check-prefix=PS4-OFF
// PS4-OFF-NOT: "-cc1" {{.*}} "-f{{(cxx-)?}}exceptions"
