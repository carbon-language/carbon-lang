// RUN: %clang -target x86_64-apple-darwin -save-temps -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-o" "save-temps.i"
// CHECK: "-disable-llvm-optzns"
// CHECK: "-o" "save-temps.bc"
// CHECK: "-o" "save-temps.s"
// CHECK: "-o" "save-temps.o"
// CHECK: "-o" "a.out" 

// Check for a single clang cc1 invocation when NOT using -save-temps.
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -S %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO-TEMPS
// NO-TEMPS: "-cc1"
// NO-TEMPS: "-S"
// NO-TEMPS: "-x" "c"

// RUN: %clang -target x86_64-apple-darwin -save-temps -arch i386 -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MULT-ARCH
// MULT-ARCH: "-o" "save-temps-i386.i"
// MULT-ARCH: "-o" "save-temps-i386.bc"
// MULT-ARCH: "-o" "save-temps-i386.s"
// MULT-ARCH: "-o" "save-temps-i386.o"
// MULT-ARCH: "-o" "a.out-i386" 
// MULT-ARCH: "-o" "save-temps-x86_64.i"
// MULT-ARCH: "-o" "save-temps-x86_64.bc"
// MULT-ARCH: "-o" "save-temps-x86_64.s"
// MULT-ARCH: "-o" "save-temps-x86_64.o"
// MULT-ARCH: "-o" "a.out-x86_64" 
// MULT-ARCH: lipo
// MULT-ARCH: "-create" "-output" "a.out" "a.out-i386" "a.out-x86_64"
