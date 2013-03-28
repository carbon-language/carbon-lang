// REQUIRES: mips-registered-target
//
// Check that -EL/-EB options adjust the toolchain flags.
//
// RUN: %clang -target mips-unknown-linux-gnu -### \
// RUN:        -EL -no-integrated-as %s 2>&1 \
// RUN:        | FileCheck -check-prefix=MIPS32-EL %s
// MIPS32-EL: "{{.*}}clang{{.*}}" "-cc1" "-triple" "mipsel-unknown-linux-gnu"
// MIPS32-EL: "{{.*}}as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EL"
// MIPS32-EL: "{{.*}}ld{{(.exe)?}}" {{.*}} "-m" "elf32ltsmip"
//
// RUN: %clang -target mips64-unknown-linux-gnu -### \
// RUN:        -EL -no-integrated-as %s 2>&1 \
// RUN:        | FileCheck -check-prefix=MIPS64-EL %s
// MIPS64-EL: "{{.*}}clang{{.*}}" "-cc1" "-triple" "mips64el-unknown-linux-gnu"
// MIPS64-EL: "{{.*}}as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EL"
// MIPS64-EL: "{{.*}}ld{{(.exe)?}}" {{.*}} "-m" "elf64ltsmip"
//
// RUN: %clang -target mipsel-unknown-linux-gnu -### \
// RUN:        -EB -no-integrated-as %s 2>&1 \
// RUN:        | FileCheck -check-prefix=MIPS32-EB %s
// MIPS32-EB: "{{.*}}clang{{.*}}" "-cc1" "-triple" "mips-unknown-linux-gnu"
// MIPS32-EB: "{{.*}}as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB"
// MIPS32-EB: "{{.*}}ld{{(.exe)?}}" {{.*}} "-m" "elf32btsmip"
//
// RUN: %clang -target mips64el-unknown-linux-gnu -### \
// RUN:        -EB -no-integrated-as %s 2>&1 \
// RUN:        | FileCheck -check-prefix=MIPS64-EB %s
// MIPS64-EB: "{{.*}}clang{{.*}}" "-cc1" "-triple" "mips64-unknown-linux-gnu"
// MIPS64-EB: "{{.*}}as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EB"
// MIPS64-EB: "{{.*}}ld{{(.exe)?}}" {{.*}} "-m" "elf64btsmip"
