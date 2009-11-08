// RUN: clang -ccc-host-triple amd64-pc-dragonfly %s -### 2> %t.log
// RUN: FileCheck -input-file %t.log %s

// CHECK: clang-cc{{.*}}" "-triple" "amd64-pc-dragonfly"
// CHECK: as{{.*}}" "-o" "{{.*}}.o" "{{.*}}.s
// CHECK: ld{{.*}}" "-dynamic-linker" "{{.*}}ld-elf.{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-L{{.*}}/gcc{{.*}}" {{.*}} "-lc" "-lgcc" "{{.*}}crtend.o" "{{.*}}crtn.o"


