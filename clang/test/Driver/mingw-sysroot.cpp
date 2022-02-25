// REQUIRES: shell
// UNSUPPORTED: system-windows

// RUN: rm -rf %T/testroot-gcc
// RUN: mkdir -p %T/testroot-gcc/bin
// RUN: ln -s %clang %T/testroot-gcc/bin/x86_64-w64-mingw32-gcc
// RUN: ln -s %clang %T/testroot-gcc/bin/x86_64-w64-mingw32-clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-gcc/x86_64-w64-mingw32
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/lib %T/testroot-gcc/lib

// RUN: rm -rf %T/testroot-clang
// RUN: mkdir -p %T/testroot-clang/bin
// RUN: ln -s %clang %T/testroot-clang/bin/x86_64-w64-mingw32-clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-clang/x86_64-w64-mingw32


// If we find a gcc in the path with the right triplet prefix, pick that as
// sysroot:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE:[^"]+]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_TESTROOT_GCC-SAME: {{^}} "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_TESTROOT_GCC-SAME: {{^}} "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If there's a matching sysroot next to the clang binary itself, prefer that
// over a gcc in the path:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang/bin/x86_64-w64-mingw32-clang -target x86_64-w64-mingw32 -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CLANG %s
// CHECK_TESTROOT_CLANG: "{{[^"]+}}/testroot-clang{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If we pick a root based on a sysroot next to the clang binary, which also
// happens to be in the same directory as gcc, make sure we still can pick up
// the libgcc directory:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-gcc/bin/x86_64-w64-mingw32-clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s
