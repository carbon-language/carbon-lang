// Should save and read back the assembly from a file
// RUN: %clang -integrated-as -via-file-asm %s -### 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK: "-o" "[[TMP:[^"]*]]"
// CHECK: -cc1as
// CHECK: [[TMP]]

// Should not force using the integrated assembler
// RUN: %clang -no-integrated-as -via-file-asm %s -### 2>&1 | FileCheck --check-prefix=NO_IAS %s
// NO_IAS-NOT: "-cc1as"

// Test arm target specifically for the same behavior
// RUN: %clang -target arm -integrated-as -via-file-asm %s -### 2>&1 | FileCheck %s
// RUN: %clang -target arm -no-integrated-as -via-file-asm %s -### 2>&1 | FileCheck --check-prefix=NO_IAS %s
