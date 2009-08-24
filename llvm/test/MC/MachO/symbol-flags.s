// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

        .reference sym_ref_A
        .reference sym_ref_def_A
sym_ref_def_A:
sym_ref_def_C:  
        .reference sym_ref_def_C
        
        .weak_reference sym_weak_ref_A
        .weak_reference sym_weak_ref_def_A
sym_weak_ref_def_A:        
sym_weak_ref_def_B:
        .weak_reference sym_weak_ref_def_B

        .data
        .globl sym_weak_def_A
        .weak_definition sym_weak_def_A        
sym_weak_def_A:

        .lazy_reference sym_lazy_ref_A
        .lazy_reference sym_lazy_ref_B
sym_lazy_ref_B:
sym_lazy_ref_C:
        .lazy_reference sym_lazy_ref_C

        .private_extern sym_private_ext_A
        .private_extern sym_private_ext_B
sym_private_ext_B:
sym_private_ext_C:
        .private_extern sym_private_ext_C
        .private_extern sym_private_ext_D
        .globl sym_private_ext_D

        .no_dead_strip sym_no_dead_strip_A

        .reference sym_ref_A
        .desc sym_ref_A, 1
        .desc sym_ref_A, 0x1234

        .desc sym_desc_flags,0x47
sym_desc_flags:

// CHECK: ('cputype', 7)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 1)
// CHECK: ('load_commands_size', 296)
// CHECK: ('flag', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 1)
// CHECK:   ('size', 192)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 0)
// CHECK:   ('file_offset', 324)
// CHECK:   ('file_size', 0)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 324)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:     # Section 1
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 324)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 324)
// CHECK:   ('nsyms', 16)
// CHECK:   ('stroff', 516)
// CHECK:   ('strsize', 260)
// CHECK:   ('_string_data', '\x00sym_ref_A\x00sym_weak_ref_A\x00sym_weak_def_A\x00sym_lazy_ref_A\x00sym_private_ext_A\x00sym_private_ext_B\x00sym_private_ext_C\x00sym_private_ext_D\x00sym_no_dead_strip_A\x00sym_ref_def_A\x00sym_ref_def_C\x00sym_weak_ref_def_A\x00sym_weak_ref_def_B\x00sym_lazy_ref_B\x00sym_lazy_ref_C\x00sym_desc_flags\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 148)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_ref_def_A')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 162)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_ref_def_C')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 176)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_weak_ref_def_A')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 195)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_weak_ref_def_B')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 214)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_lazy_ref_B')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 229)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_lazy_ref_C')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 244)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_desc_flags')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 74)
// CHECK:     ('n_type', 0x1f)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_private_ext_B')
// CHECK:    ),
// CHECK:     # Symbol 8
// CHECK:    (('n_strx', 92)
// CHECK:     ('n_type', 0x1f)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_private_ext_C')
// CHECK:    ),
// CHECK:     # Symbol 9
// CHECK:    (('n_strx', 26)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 128)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_weak_def_A')
// CHECK:    ),
// CHECK:     # Symbol 10
// CHECK:    (('n_strx', 41)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 33)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_lazy_ref_A')
// CHECK:    ),
// CHECK:     # Symbol 11
// CHECK:    (('n_strx', 128)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_no_dead_strip_A')
// CHECK:    ),
// CHECK:     # Symbol 12
// CHECK:    (('n_strx', 56)
// CHECK:     ('n_type', 0x11)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_private_ext_A')
// CHECK:    ),
// CHECK:     # Symbol 13
// CHECK:    (('n_strx', 110)
// CHECK:     ('n_type', 0x11)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_private_ext_D')
// CHECK:    ),
// CHECK:     # Symbol 14
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 4660)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_ref_A')
// CHECK:    ),
// CHECK:     # Symbol 15
// CHECK:    (('n_strx', 11)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 64)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_weak_ref_A')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 7)
// CHECK:   ('iextdefsym', 7)
// CHECK:   ('nextdefsym', 3)
// CHECK:   ('iundefsym', 10)
// CHECK:   ('nundefsym', 6)
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
