// RUN: clang -ccc-host-triple amd64-pc-dragonfly %s -### 2> %t.log &&
// RUN: grep 'clang-cc" "-triple" "amd64-pc-dragonfly"' %t.log &&
// RUN: grep 'as" "-o" ".*\.o" ".*\.s' %t.log &&
// RUN: grep 'ld" "-dynamic-linker" ".*ld-elf.*" "-o" "a\.out" ".*crt1.o" ".*crti.o" ".*crtbegin.o" ".*\.o" "-L.*/gcc.*" .* "-lc" "-lgcc" ".*crtend.o" ".*crtn.o"' %t.log &&
// RUN: true

