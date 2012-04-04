// RUN: %clang -c %s -target i386-apple-darwin -### 2>&1 | FileCheck %s --check-prefix=PIC_ON_BY_DEFAULT
// PIC_ON_BY_DEFAULT: "-pic-level" "1"

// RUN: %clang -c %s -target i386-apple-darwin -### -fno-pic 2>&1 | FileCheck %s --check-prefix=FNO_PIC
// FNO_PIC: "-mrelocation-model" "static"
