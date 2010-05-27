// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

   ja 1f
1: nop
   jae 1f
1: nop
   jb 1f
1: nop
   jbe 1f
1: nop
   jc 1f
1: nop
   jcxz 1f
1: nop
   jecxz 1f
1: nop
   je 1f
1: nop
   jg 1f
1: nop
   jge 1f
1: nop
   jl 1f
1: nop
   jle 1f
1: nop
   jna 1f
1: nop
   jnae 1f
1: nop
   jnb 1f
1: nop
   jnbe 1f
1: nop
   jnc 1f
1: nop
   jne 1f
1: nop
   jng 1f
1: nop
   jnge 1f
1: nop
   jnl 1f
1: nop
   jnle 1f
1: nop
   jno 1f
1: nop
   jnp 1f
1: nop
   jns 1f
1: nop
   jnz 1f
1: nop
   jo 1f
1: nop
   jp 1f
1: nop
   jpe 1f
1: nop
   jpo 1f
1: nop
   js 1f
1: nop
   jz 1f
1: nop

// CHECK: ('cputype', 7)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 1)
// CHECK: ('load_commands_size', 124)
// CHECK: ('flag', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 1)
// CHECK:   ('size', 124)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 96)
// CHECK:   ('file_offset', 152)
// CHECK:   ('file_size', 96)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 1)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 96)
// CHECK:     ('offset', 152)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', 'w\x00\x90s\x00\x90r\x00\x90v\x00\x90r\x00\x90\xe3\x00\x90\xe3\x00\x90t\x00\x90\x7f\x00\x90}\x00\x90|\x00\x90~\x00\x90v\x00\x90r\x00\x90s\x00\x90w\x00\x90s\x00\x90u\x00\x90~\x00\x90|\x00\x90}\x00\x90\x7f\x00\x90q\x00\x90{\x00\x90y\x00\x90u\x00\x90p\x00\x90z\x00\x90z\x00\x90{\x00\x90x\x00\x90t\x00\x90')
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
