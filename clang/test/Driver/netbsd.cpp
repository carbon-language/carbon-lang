// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd6.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-6 %s

// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd6.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-6 %s

// X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// X86_64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64-7: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// X86_64-6: "-lm" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// S-X86_64: ld{{.*}}" "-Bstatic"
// S-X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// S-X86_64-7: ld{{.*}}" "-Bstatic"
// S-X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64-7: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// S-X86_64-6: ld{{.*}}" "-Bstatic"
// S-X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// S-X86_64-6: "-lm" "-lc" "-lgcc_eh" "-lc" "-lgcc"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"
