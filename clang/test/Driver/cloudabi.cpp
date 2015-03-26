// RUN: %clangxx %s -### -target x86_64-unknown-cloudabi 2>&1 | FileCheck %s
// CHECK: "-cc1" "-triple" "x86_64-unknown-cloudabi" {{.*}} "-ffunction-sections" "-fdata-sections"
// CHECK: "-Bstatic" "--eh-frame-hdr" "--gc-sections" "-o" "a.out" "crt0.o" "crtbegin.o" "{{.*}}" "{{.*}}" "-lc++" "-lc++abi" "-lunwind" "-lc" "-lcompiler_rt" "crtend.o"
