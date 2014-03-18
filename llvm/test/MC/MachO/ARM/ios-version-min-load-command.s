// RUN: llvm-mc -triple armv7-apple-ios %s -filetype=obj -o - | macho-dump | FileCheck %s

// Test the formation of the version-min load command in the MachO.
// use a nonsense but well formed version.
.ios_version_min 99,8,7
// CHECK:  (('command', 37)
// CHECK:   ('size', 16)
// CHECK:   ('version, 6490119)
// CHECK:   ('reserved, 0)
// CHECK:  ),
