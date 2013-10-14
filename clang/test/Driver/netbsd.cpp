// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd7.0.0 %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd6.0.0 %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-6 %s

// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd %s -static -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd7.0.0 -static %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64--netbsd6.0.0 -static %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-6 %s

// X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// X86_64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64: "-o" "a.out" "crt0.o" "crti.o" "crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64: "-lm" "-lc" "crtend.o" "crtn.o"

// X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-7: "-o" "a.out" "crt0.o" "crti.o" "crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64-7: "-lm" "-lc" "crtend.o" "crtn.o"

// X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-6: "-o" "a.out" "crt0.o" "crti.o" "crtbegin.o" "{{.*}}.o" "-lstdc++"
// X86_64-6: "-lm" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// X86_64-6: "crtend.o" "crtn.o"

// S-X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// S-X86_64: ld{{.*}}" "-Bstatic"
// S-X86_64: "-o" "a.out" "crt0.o" "crti.o" "crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64: "-lm" "-lc" "crtend.o" "crtn.o"

// S-X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// S-X86_64-7: ld{{.*}}" "-Bstatic"
// S-X86_64-7: "-o" "a.out" "crt0.o" "crti.o" "crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64-7: "-lm" "-lc" "crtend.o" "crtn.o"

// S-X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// S-X86_64-6: ld{{.*}}" "-Bstatic"
// S-X86_64-6: "-o" "a.out" "crt0.o" "crti.o" "crtbegin.o" "{{.*}}.o" "-lstdc++"
// S-X86_64-6: "-lm" "-lc" "-lgcc_eh" "-lc" "-lgcc"
// S-X86_64-6: "crtend.o" "crtn.o"
