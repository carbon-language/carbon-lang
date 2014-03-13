// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-7 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd6.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-6 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-APCS %s
// RUN: %clang -no-canonical-prefixes -target thumb--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=THUMB %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd7.0.0-eabi \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd6.0.0-eabi \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-6 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd-eabihf \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-HF %s
// RUN: %clang -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s
// RUN: %clang -no-canonical-prefixes -target sparc64--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64 %s

// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-7 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd6.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-6 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd7.0.0-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd6.0.0-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM-6 %s
// RUN: %clang -no-canonical-prefixes -target sparc--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC %s
// RUN: %clang -no-canonical-prefixes -target sparc64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64 %s

// X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// X86_64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd"
// AARCH64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64-7: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd7.0.0"
// AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64-7: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabi"
// ARM: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// ARM: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM: "-m" "armelf_nbsd_eabi"
// ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-APCS: clang{{.*}}" "-cc1" "-triple" "armv4--netbsd"
// ARM-APCS: as{{.*}}" "-mcpu=strongarm" "-o"
// ARM-APCS: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-APCS: "-m" "armelf_nbsd"
// ARM-APCS: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// THUMB: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabi"
// THUMB: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// THUMB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// THUMB: "-m" "armelf_nbsd_eabi"
// THUMB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-7: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd7.0.0-eabi"
// ARM-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-7: "-m" "armelf_nbsd_eabi"
// ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// ARM-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-6: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd6.0.0-eabi"
// ARM-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-6: "-m" "armelf_nbsd_eabi"
// ARM-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-HF: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabihf"
// ARM-HF: ld{{.*}}" "-m" "armelf_nbsd_eabihf"

// SPARC: clang{{.*}}" "-cc1" "-triple" "sparc--netbsd"
// SPARC: as{{.*}}" "-32" "-o"
// SPARC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC: "-m" "elf32_sparc"
// SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64: clang{{.*}}" "-cc1" "-triple" "sparc64--netbsd"
// SPARC64: as{{.*}}" "-64" "-Av9" "-o"
// SPARC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64: "-m" "elf64_sparc"
// SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// S-X86_64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// S-X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// S-X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64-6: "-lgcc_eh" "-lc" "-lgcc"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd"
// S-AARCH64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64: "-lgcc_eh" "-lc" "-lgcc"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64-7: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd7.0.0"
// S-AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64-7: "-lgcc_eh" "-lc" "-lgcc"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabi"
// S-ARM: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM: "-m" "armelf_nbsd_eabi"
// S-ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM-7: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd7.0.0-eabi"
// S-ARM-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM-7: "-m" "armelf_nbsd_eabi"
// S-ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM-7: "-lgcc_eh" "-lc" "-lgcc"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM-6: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd6.0.0-eabi"
// S-ARM-6: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM-6: "-m" "armelf_nbsd_eabi"
// S-ARM-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM-6: "-lgcc_eh" "-lc" "-lgcc"
// S-ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC: clang{{.*}}" "-cc1" "-triple" "sparc--netbsd"
// S-SPARC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC: "-m" "elf32_sparc"
// S-SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-SPARC: "-lgcc_eh" "-lc" "-lgcc"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64: clang{{.*}}" "-cc1" "-triple" "sparc64--netbsd"
// S-SPARC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64: "-m" "elf64_sparc"
// S-SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-SPARC64: "-lgcc_eh" "-lc" "-lgcc"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"
