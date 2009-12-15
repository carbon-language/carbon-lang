// RUN: %clang -ccc-clang-archs "" -ccc-host-triple i686-pc-openbsd %s -### 2> %t.log
// RUN: FileCheck -input-file %t.log %s

// CHECK: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK: as{{.*}}" "-o" "{{.*}}.o" "{{.*}}.s
// CHECK: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"
