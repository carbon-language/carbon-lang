// ================== Check default Architecture on krait CPU
// RUN: %clang -target arm-linux-gnueabi -mcpu=krait -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CPUV7A %s
// CHECK-CPUV7A: "-cc1"{{.*}} "-triple" "armv7-{{.*}}
