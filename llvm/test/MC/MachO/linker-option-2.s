// RUN: llvm-mc -n -triple x86_64-apple-darwin10 %s -filetype=obj | macho-dump | FileCheck %s

// CHECK: ('load_commands_size', 120)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 2
// CHECK:  (('command', 45)
// CHECK:   ('size', 16)
// CHECK:   ('count', 1)
// CHECK:   ('_strings', [
// CHECK: 	"a",
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 3
// CHECK:  (('command', 45)
// CHECK:   ('size', 16)
// CHECK:   ('count', 2)
// CHECK:   ('_strings', [
// CHECK: 	"a",
// CHECK: 	"b",
// CHECK:   ])
// CHECK:  ),
// CHECK: ])

.linker_option "a"
.linker_option "a", "b"
