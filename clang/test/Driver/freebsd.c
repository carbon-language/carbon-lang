// RUN: %clang -ccc-clang-archs "" -ccc-host-triple powerpc64-pc-freebsd8 %s -### 2> %t
// RUN: FileCheck --check-prefix=CHECK-PPC < %t %s
//
// CHECK-PPC: clang{{.*}}" "-cc1" "-triple" "powerpc64-pc-freebsd8"
// CHECK-PPC: as{{.*}}" "-o" "{{.*}}.o" "{{.*}}.s
// CHECK-PPC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"


// Check that -m32 properly adjusts the toolchain flags.
//
// RUN: %clang -ccc-host-triple x86_64-pc-freebsd8 -m32 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-LIB32 < %t %s
//
// CHECK-LIB32: clang{{.*}}" "-cc1" "-triple" "i386-pc-freebsd8"
// CHECK-LIB32: as{{.*}}" "--32"
// CHECK-LIB32: ld{{.*}}" {{.*}} "-m" "elf_i386_fbsd"
//
// RUN: %clang -ccc-host-triple x86_64-pc-freebsd8 -m32 -print-search-dirs %s > %t
// RUN: FileCheck --check-prefix=CHECK-LIB32PATHS < %t %s
//
// CHECK-LIB32PATHS: libraries: ={{.*}}:/usr/lib32
