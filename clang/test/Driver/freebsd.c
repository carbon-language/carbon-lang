// RUN: clang -ccc-clang-archs "" -ccc-host-triple ppc64-pc-freebsd8 %s -### 2> %t.log &&
// RUN: cat %t.log &&
// RUN: grep 'clang-cc" "-triple" "powerpc64-pc-freebsd8"' %t.log &&
// RUN: grep 'as" "-o" ".*\.o" ".*\.s' %t.log &&
// RUN: grep 'ld" "--eh-frame-hdr" "-dynamic-linker" ".*ld-elf.*" "-o" "a\.out" ".*crt1.o" ".*crti.o" "crtbegin.o" ".*\.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" ".*crtend.o" ".*crtn.o"' %t.log &&
// RUN: true

