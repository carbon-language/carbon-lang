// RUN: clang -ccc-host-triple i386-apple-darwin10 -### -x assembler -c %s -static -dynamic 2>%t &&
// RUN: FileCheck -check-prefix=STATIC_AND_DYNAMIC-32 --input-file %t %s &&

// CHECK-STATIC_AND_DYNAMIC-32: as{{(.exe)?}}" "-arch" "i386" "-force_cpusubtype_ALL" "-static" "-o"

// RUN: clang -ccc-host-triple x86_64-apple-darwin10 -### -x assembler -c %s -static 2>%t &&
// RUN: FileCheck -check-prefix=STATIC-64 --input-file %t %s

// CHECK-STATIC-64: as{{(.exe)?}}" "-arch" "x86_64" "-force_cpusubtype_ALL" "-o"

