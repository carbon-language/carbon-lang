// RUN: clang -ccc-clang-archs "" -ccc-host-triple i686-pc-openbsd %s -### 2> %t.log &&
// RUN: grep 'clang-cc" "-triple" "i386-pc-openbsd"' %t.log &&
// RUN: grep 'as" "-o" ".*\.o" ".*\.s' %t.log &&
// RUN: grep 'ld" "--eh-frame-hdr" "-dynamic-linker" ".*ld.so" "-o" "a\.out" ".*crt0.o" ".*crtbegin.o" ".*\.o" "-lc" ".*crtend.o"' %t.log &&
// RUN: true

