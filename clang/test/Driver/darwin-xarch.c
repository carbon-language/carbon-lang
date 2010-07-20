// RUN: %clang -ccc-host-triple x86_64-apple-darwin10 -### \
// RUN:   -arch i386 -Xarch_i386 -mmacosx-version-min=10.4 \
// RUN:   -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5 \
// RUN:   -c %s 2> %t
// RUN: FileCheck < %t %s
//
// CHECK: clang{{.*}}" "-cc1" "-triple" "i386-apple-darwin8.0.0" 
// CHECK: clang{{.*}}" "-cc1" "-triple" "x86_64-apple-darwin9.0.0" 
