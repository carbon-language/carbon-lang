// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | macho-dump | FileCheck %s

// Test the formation of the version-min load command in the MachO.
// use a nonsense but well formed version.
.macosx_version_min 25,3,1
// CHECK:  (('command', 36)
// CHECK:   ('size', 16)
// CHECK:   ('version, 1639169)
// CHECK:   ('sdk, 0)
// CHECK:  ),
