// PR21000: Test that -I is passed to both external and integrated assemblers.

// RUN: %clang -target x86_64-linux-gnu -c -no-integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target x86_64-linux-gnu -c -no-integrated-as %s \
// RUN:   -I foo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target x86_64-linux-gnu -c -integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target x86_64-linux-gnu -c -integrated-as %s \
// RUN:   -I foo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// Other GNU targets

// RUN: %clang -target aarch64-linux-gnu -c -no-integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target aarch64-linux-gnu -c -integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target armv7-linux-gnueabihf -c -no-integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target armv7-linux-gnueabihf -c -integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-I" "foo_dir"

// Test that assembler options don't cause warnings when there's no assembler
// stage.

// RUN: %clang -mincremental-linker-compatible -E -fintegrated-as \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -mincremental-linker-compatible -E -fno-integrated-as \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s

// RUN: %clang -mincremental-linker-compatible -E -fintegrated-as \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -mincremental-linker-compatible -E -fno-integrated-as \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s

// RUN: %clang -mimplicit-it=always -target armv7-linux-gnueabi -E \
// RUN:   -fintegrated-as -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -mimplicit-it=always -target armv7-linux-gnueabi -E \
// RUN:   -fno-integrated-as -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s

// RUN: %clang -mimplicit-it=always -target armv7-linux-gnueabi -E \
// RUN:   -fintegrated-as -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -mimplicit-it=always -target armv7-linux-gnueabi -E \
// RUN:   -fno-integrated-as -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s

// RUN: %clang -Wa,-mbig-obj -target i386-pc-windows -E -fintegrated-as \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -Wa,-mbig-obj -target i386-pc-windows -E -fno-integrated-as \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s

// RUN: %clang -Wa,-mbig-obj -target i386-pc-windows -E -fintegrated-as \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -Wa,-mbig-obj -target i386-pc-windows -E -fno-integrated-as \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s

// RUN: %clang -Xassembler -mbig-obj -target i386-pc-windows -E -fintegrated-as \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -Xassembler -mbig-obj -target i386-pc-windows -E \
// RUN:   -fno-integrated-as -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s

// RUN: %clang -Xassembler -mbig-obj -target i386-pc-windows -E -fintegrated-as \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -Xassembler -mbig-obj -target i386-pc-windows -E \
// RUN:   -fno-integrated-as -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s

// NOWARN-NOT: unused

// Test that unsupported arguments do not cause errors when -fno-integrated-as
// is set.
// RUN: %clang -Wa,-mno-warn-deprecated -fno-integrated-as %s -S 2>&1 \
// RUN:   | FileCheck --check-prefix=NOERROR --allow-empty %s
// NOERROR-NOT: error: unsupported argument '-mno-warn-deprecated' to option 'Wa,'

// -Wa flags shouldn't cause warnings without an assembler stage with
// -fno-integrated-as either.
// RUN: %clang -Wa,-mno-warn-deprecated -fno-integrated-as -x c++ %s -S 2>&1 \
// RUN:   -o /dev/null \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s

// But -m flags for the integrated assembler _should_ warn if the integrated
// assembler is not in use.
// RUN: %clang -mrelax-all -fintegrated-as -x c++ %s -S -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWARN --allow-empty %s
// RUN: %clang -mrelax-all -fno-integrated-as -x c++ %s -S -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s
// WARN: unused
