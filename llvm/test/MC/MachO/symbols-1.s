// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

sym_local_B:
.globl sym_globl_def_B
.globl sym_globl_undef_B
sym_local_A:
.globl sym_globl_def_A
.globl sym_globl_undef_A
sym_local_C:
.globl sym_globl_def_C
.globl sym_globl_undef_C
        
sym_globl_def_A: 
sym_globl_def_B: 
sym_globl_def_C: 
        .long 0

// CHECK: ('cputype', 7)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 1)
// CHECK: ('load_commands_size', 228)
// CHECK: ('flag', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 1)
// CHECK:   ('size', 124)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 4)
// CHECK:   ('file_offset', 256)
// CHECK:   ('file_size', 4)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 1)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 4)
// CHECK:     ('offset', 256)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 260)
// CHECK:   ('nsyms', 9)
// CHECK:   ('stroff', 368)
// CHECK:   ('strsize', 140)
// CHECK:   ('_string_data', '\x00sym_globl_def_B\x00sym_globl_undef_B\x00sym_globl_def_A\x00sym_globl_undef_A\x00sym_globl_def_C\x00sym_globl_undef_C\x00sym_local_B\x00sym_local_A\x00sym_local_C\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 103)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_local_B')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 115)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_local_A')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 127)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_local_C')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 35)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_globl_def_A')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_globl_def_B')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 69)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_globl_def_C')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 51)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_globl_undef_A')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 17)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_globl_undef_B')
// CHECK:    ),
// CHECK:     # Symbol 8
// CHECK:    (('n_strx', 85)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_globl_undef_C')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 3)
// CHECK:   ('iextdefsym', 3)
// CHECK:   ('nextdefsym', 3)
// CHECK:   ('iundefsym', 6)
// CHECK:   ('nundefsym', 3)
// CHECK:   ('tocoff', 0)
// CHECK:   ('ntoc', 0)
// CHECK:   ('modtaboff', 0)
// CHECK:   ('nmodtab', 0)
// CHECK:   ('extrefsymoff', 0)
// CHECK:   ('nextrefsyms', 0)
// CHECK:   ('indirectsymoff', 0)
// CHECK:   ('nindirectsyms', 0)
// CHECK:   ('extreloff', 0)
// CHECK:   ('nextrel', 0)
// CHECK:   ('locreloff', 0)
// CHECK:   ('nlocrel', 0)
// CHECK:   ('_indirect_symbols', [
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
