// REQUIRES: system-darwin

// Check that ld gets "-lto_library".

// RUN: mkdir -p %t/bin
// RUN: mkdir -p %t/lib
// RUN: touch %t/lib/libLTO.dylib
// RUN: %clang -fuse-ld= -target x86_64-apple-darwin10 -### %s \
// RUN:   -ccc-install-dir %t/bin -mlinker-version=133 2> %t.log
// RUN: FileCheck -check-prefix=LINK_LTOLIB_PATH %s -input-file %t.log
//
// LINK_LTOLIB_PATH: {{ld(.exe)?"}}
// LINK_LTOLIB_PATH: "-lto_library"

// Also pass -lto_library even if the file doesn't exist; if it's needed at
// link time, ld will complain instead.
// RUN: %clang -fuse-ld= -target x86_64-apple-darwin10 -### %s \
// RUN:   -ccc-install-dir %S/dummytestdir -mlinker-version=133 2> %t.log
// RUN: FileCheck -check-prefix=LINK_LTOLIB_PATH %s -input-file %t.log


// Check that -object_lto_path is passed correctly to ld64
// RUN: %clang -fuse-ld= -target x86_64-apple-darwin10 %s -flto=full -### 2>&1 \
// RUN:     | FileCheck -check-prefix=FULL_LTO_OBJECT_PATH %s
// FULL_LTO_OBJECT_PATH: {{ld(.exe)?"}}
// FULL_LTO_OBJECT_PATH-SAME: "-object_path_lto"
// FULL_LTO_OBJECT_PATH-SAME: {{cc\-[a-zA-Z0-9_]+.o}}"
// RUN: %clang -fuse-ld= -target x86_64-apple-darwin10 %s -flto=thin -### 2>&1 \
// RUN:     | FileCheck -check-prefix=THIN_LTO_OBJECT_PATH %s
// THIN_LTO_OBJECT_PATH: {{ld(.exe)?"}}
// THIN_LTO_OBJECT_PATH-SAME: "-object_path_lto"
// THIN_LTO_OBJECT_PATH-SAME: {{thinlto\-[a-zA-Z0-9_]+}}


// Check that we pass through -fglobal-isel flags to libLTO.
// RUN: %clang -target arm64-apple-darwin %s -flto -fglobal-isel -### 2>&1 | \
// RUN:   FileCheck --check-prefix=GISEL %s
// GISEL: {{ld(.exe)?"}}
// GISEL: "-mllvm" "-global-isel"
// GISEL: "-mllvm" "-global-isel-abort=0"
