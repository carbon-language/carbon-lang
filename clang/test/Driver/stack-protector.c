// RUN: %clang -fno-stack-protector -### %s 2>&1 | FileCheck %s -check-prefix=NOSSP
// NOSSP-NOT: "-stack-protector" "1"
// NOSSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=SSP
// SSP: "-stack-protector" "1"
// SSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -fstack-protector --param ssp-buffer-size=16 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-BUF
// SSP-BUF: "-stack-protector" "1"
// SSP-BUF: "-stack-protector-buffer-size" "16" 

// RUN: %clang -target i386-pc-openbsd -### %s 2>&1 | FileCheck %s -check-prefix=OPENBSD
// OPENBSD: "-stack-protector" "1"

// RUN: %clang -target i386-pc-openbsd -fno-stack-protector -### %s 2>&1 | FileCheck %s -check-prefix=OPENBSD_OFF
// OPENBSD_OFF-NOT: "-stack-protector"

// RUN: %clang -fstack-protector-strong -### %s 2>&1 | FileCheck %s -check-prefix=SSP-STRONG
// SSP-STRONG: "-stack-protector" "2"
// SSP-STRONG-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -fstack-protector-all -### %s 2>&1 | FileCheck %s -check-prefix=SSP-ALL
// SSP-ALL: "-stack-protector" "3"
// SSP-ALL-NOT: "-stack-protector-buffer-size" 
