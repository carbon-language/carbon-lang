// RUN: clang -ccc-clang-archs "" -ccc-host-triple powerpc64-pc-freebsd8 %s -### 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -input-file %t.log %s

// CHECK: clang{{.*}}" "-cc1" "-triple" "powerpc64-pc-freebsd8"
// CHECK: as{{.*}}" "-o" "{{.*}}.o" "{{.*}}.s
// CHECK: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"
