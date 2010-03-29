// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// These examples are taken from <mach-o/x86_64/reloc.h>.

        .text
_foo:
        ret

_baz:
        call _foo
 	call _foo+4
 	movq _foo@GOTPCREL(%rip), %rax
 	pushq _foo@GOTPCREL(%rip)
 	movl _foo(%rip), %eax
 	movl _foo+4(%rip), %eax
 	movb  $0x12, _foo(%rip)
 	movl  $0x12345678, _foo(%rip)
 	.quad _foo
_bar:
 	.quad _foo+4
 	.quad _foo - _bar
 	.quad _foo - _bar + 4
 	.long _foo - _bar
 	leaq L1(%rip), %rax
 	leaq L0(%rip), %rax
        addl $6,L0(%rip)
        addw $500,L0(%rip)
        addl $500,L0(%rip)

_prev:
        .space 12,0x90
 	.quad L1
L0:
        .quad L0
L_pc:
 	.quad _foo - L_pc
 	.quad _foo - L1
L1:
 	.quad L1 - _prev

        .data
.long	_foobar@GOTPCREL+4
.long	_foo@GOTPCREL+4

// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 1)
// CHECK: ('load_commands_size', 336)
// CHECK: ('flag', 0)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 232)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 189)
// CHECK:   ('file_offset', 368)
// CHECK:   ('file_size', 189)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 181)
// CHECK:     ('offset', 368)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 560)
// CHECK:     ('num_reloc', 27)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0xa5),
// CHECK:      ('word-1', 0x5e000003)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0xa5),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0x9d),
// CHECK:      ('word-1', 0x5e000003)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0x9d),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0x95),
// CHECK:      ('word-1', 0xe000003)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0x8d),
// CHECK:      ('word-1', 0xe000003)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0x79),
// CHECK:      ('word-1', 0x8d000003)),
// CHECK:     # Relocation 7
// CHECK:     (('word-0', 0x71),
// CHECK:      ('word-1', 0x7d000003)),
// CHECK:     # Relocation 8
// CHECK:     (('word-0', 0x69),
// CHECK:      ('word-1', 0x6d000003)),
// CHECK:     # Relocation 9
// CHECK:     (('word-0', 0x63),
// CHECK:      ('word-1', 0x1d000003)),
// CHECK:     # Relocation 10
// CHECK:     (('word-0', 0x5c),
// CHECK:      ('word-1', 0x1d000003)),
// CHECK:     # Relocation 11
// CHECK:     (('word-0', 0x55),
// CHECK:      ('word-1', 0x5c000002)),
// CHECK:     # Relocation 12
// CHECK:     (('word-0', 0x55),
// CHECK:      ('word-1', 0xc000000)),
// CHECK:     # Relocation 13
// CHECK:     (('word-0', 0x4d),
// CHECK:      ('word-1', 0x5e000002)),
// CHECK:     # Relocation 14
// CHECK:     (('word-0', 0x4d),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 15
// CHECK:     (('word-0', 0x45),
// CHECK:      ('word-1', 0x5e000002)),
// CHECK:     # Relocation 16
// CHECK:     (('word-0', 0x45),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 17
// CHECK:     (('word-0', 0x3d),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 18
// CHECK:     (('word-0', 0x35),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 19
// CHECK:     (('word-0', 0x2d),
// CHECK:      ('word-1', 0x8d000000)),
// CHECK:     # Relocation 20
// CHECK:     (('word-0', 0x26),
// CHECK:      ('word-1', 0x6d000000)),
// CHECK:     # Relocation 21
// CHECK:     (('word-0', 0x20),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 22
// CHECK:     (('word-0', 0x1a),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 23
// CHECK:     (('word-0', 0x14),
// CHECK:      ('word-1', 0x4d000000)),
// CHECK:     # Relocation 24
// CHECK:     (('word-0', 0xe),
// CHECK:      ('word-1', 0x3d000000)),
// CHECK:     # Relocation 25
// CHECK:     (('word-0', 0x7),
// CHECK:      ('word-1', 0x2d000000)),
// CHECK:     # Relocation 26
// CHECK:     (('word-0', 0x2),
// CHECK:      ('word-1', 0x2d000000)),
// CHECK:   ])
// CHECK:   ('_section_data', '\xc3\xe8\x00\x00\x00\x00\xe8\x04\x00\x00\x00H\x8b\x05\x00\x00\x00\x00\xff5\x00\x00\x00\x00\x8b\x05\x00\x00\x00\x00\x8b\x05\x04\x00\x00\x00\xc6\x05\xff\xff\xff\xff\x12\xc7\x05\xfc\xff\xff\xffxV4\x12\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00H\x8d\x05,\x00\x00\x00H\x8d\x05\x14\x00\x00\x00\x83\x05\x13\x00\x00\x00\x06f\x81\x05\x12\x00\x00\x00\xf4\x01\x81\x05\x10\x00\x00\x00\xf4\x01\x00\x00\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90,\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\xe4\xff\xff\xff\xff\xff\xff\xff\xd4\xff\xff\xff\xff\xff\xff\xff,\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 181)
// CHECK:     ('size', 8)
// CHECK:     ('offset', 549)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 776)
// CHECK:     ('num_reloc', 2)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x4),
// CHECK:      ('word-1', 0x4d000000)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x0),
// CHECK:      ('word-1', 0x4d000004)),
// CHECK:   ])
// CHECK:   ('_section_data', '\x04\x00\x00\x00\x04\x00\x00\x00')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 792)
// CHECK:   ('nsyms', 5)
// CHECK:   ('stroff', 872)
// CHECK:   ('strsize', 32)
// CHECK:   ('_string_data', '\x00_foobar\x00_foo\x00_baz\x00_bar\x00_prev\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 9)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_foo')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 14)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 1)
// CHECK:     ('_string', '_baz')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 19)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 61)
// CHECK:     ('_string', '_bar')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 24)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 129)
// CHECK:     ('_string', '_prev')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_foobar')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 4)
// CHECK:   ('iextdefsym', 4)
// CHECK:   ('nextdefsym', 0)
// CHECK:   ('iundefsym', 4)
// CHECK:   ('nundefsym', 1)
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
