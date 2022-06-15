// RUN: %clang -target aarch64 -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64 -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// GENERICV87A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.7a"

// RUN: %clang -target aarch64_be -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// GENERICV87A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.7a"
