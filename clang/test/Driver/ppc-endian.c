// RUN: %clang -target powerpc64le -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE %s
// RUN: %clang -target powerpc64le -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE %s
// RUN: %clang -target powerpc64 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE %s
// CHECK-LE: "-cc1"{{.*}} "-triple" "powerpc64le{{.*}}"

// RUN: %clang -target powerpc64 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE %s
// RUN: %clang -target powerpc64 -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE %s
// RUN: %clang -target powerpc64le -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE %s
// CHECK-BE: "-cc1"{{.*}} "-triple" "powerpc64{{.*}}"
