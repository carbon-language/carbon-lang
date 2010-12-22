@ RUN: llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: macho-dump --dump-section-data < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s

	.syntax unified
        .text
_f0:
        bl _printf

@ CHECK: ('cputype', 12)
@ CHECK: ('cpusubtype', 9)
@ CHECK: ('filetype', 1)
@ CHECK: ('num_load_commands', 3)
@ CHECK: ('load_commands_size', 228)
@ CHECK: ('flag', 0)
@ CHECK: ('load_commands', [
@ CHECK:   # Load Command 0
@ CHECK:  (('command', 1)
@ CHECK:   ('size', 124)
@ CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:   ('vm_addr', 0)
@ CHECK:   ('vm_size', 4)
@ CHECK:   ('file_offset', 256)
@ CHECK:   ('file_size', 4)
@ CHECK:   ('maxprot', 7)
@ CHECK:   ('initprot', 7)
@ CHECK:   ('num_sections', 1)
@ CHECK:   ('flags', 0)
@ CHECK:   ('sections', [
@ CHECK:     # Section 0
@ CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('address', 0)
@ CHECK:     ('size', 4)
@ CHECK:     ('offset', 256)
@ CHECK:     ('alignment', 0)
@ CHECK:     ('reloc_offset', 260)
@ CHECK:     ('num_reloc', 1)
@ CHECK:     ('flags', 0x80000400)
@ CHECK:     ('reserved1', 0)
@ CHECK:     ('reserved2', 0)
@ CHECK:    ),
@ CHECK:   ('_relocations', [
@ CHECK:     # Relocation 0
@ CHECK:     (('word-0', 0x0),
@ CHECK:      ('word-1', 0x5d000001)),
@ CHECK:   ])
@ CHECK:   ('_section_data', 'feffffeb')
@ CHECK:   ])
@ CHECK:  ),
@ CHECK:   # Load Command 1
@ CHECK:  (('command', 2)
@ CHECK:   ('size', 24)
@ CHECK:   ('symoff', 268)
@ CHECK:   ('nsyms', 2)
@ CHECK:   ('stroff', 292)
@ CHECK:   ('strsize', 16)
@ CHECK:   ('_string_data', '\x00_printf\x00_f0\x00\x00\x00\x00')
@ CHECK:   ('_symbols', [
@ CHECK:     # Symbol 0
@ CHECK:    (('n_strx', 9)
@ CHECK:     ('n_type', 0xe)
@ CHECK:     ('n_sect', 1)
@ CHECK:     ('n_desc', 0)
@ CHECK:     ('n_value', 0)
@ CHECK:     ('_string', '_f0')
@ CHECK:    ),
@ CHECK:     # Symbol 1
@ CHECK:    (('n_strx', 1)
@ CHECK:     ('n_type', 0x1)
@ CHECK:     ('n_sect', 0)
@ CHECK:     ('n_desc', 0)
@ CHECK:     ('n_value', 0)
@ CHECK:     ('_string', '_printf')
@ CHECK:    ),
@ CHECK:   ])
@ CHECK:  ),
@ CHECK:   # Load Command 2
@ CHECK:  (('command', 11)
@ CHECK:   ('size', 80)
@ CHECK:   ('ilocalsym', 0)
@ CHECK:   ('nlocalsym', 1)
@ CHECK:   ('iextdefsym', 1)
@ CHECK:   ('nextdefsym', 0)
@ CHECK:   ('iundefsym', 1)
@ CHECK:   ('nundefsym', 1)
@ CHECK:   ('tocoff', 0)
@ CHECK:   ('ntoc', 0)
@ CHECK:   ('modtaboff', 0)
@ CHECK:   ('nmodtab', 0)
@ CHECK:   ('extrefsymoff', 0)
@ CHECK:   ('nextrefsyms', 0)
@ CHECK:   ('indirectsymoff', 0)
@ CHECK:   ('nindirectsyms', 0)
@ CHECK:   ('extreloff', 0)
@ CHECK:   ('nextrel', 0)
@ CHECK:   ('locreloff', 0)
@ CHECK:   ('nlocrel', 0)
@ CHECK:   ('_indirect_symbols', [
@ CHECK:   ])
@ CHECK:  ),
@ CHECK: ])
