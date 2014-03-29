; RUN: llvm-mc -n -triple arm64-apple-darwin10 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

	.text
_fred:
	bl	_func
	bl	_func + 20

	adrp	x3, _data@page
        ldr	w2, [x3, _data@pageoff]

        add	x3, x3, _data@pageoff + 4

	adrp	x3, _data@page+1
        ldr	w2, [x3, _data@pageoff + 4]

	adrp	x3, _data_ext@gotpage
        ldr	w2, [x3, _data_ext@gotpageoff]

	.data
_data:
        .quad _foo
        .quad _foo + 4
        .quad _foo - _bar
        .quad _foo - _bar + 4

        .long _foo - _bar

        .quad _foo@got
        .long _foo@got - .


; CHECK: ('cputype', 16777228)
; CHECK: ('cpusubtype', 0)
; CHECK: ('filetype', 1)
; CHECK: ('num_load_commands', 3)
; CHECK: ('load_commands_size', 336)
; CHECK: ('flag', 0)
; CHECK: ('reserved', 0)
; CHECK: ('load_commands', [
; CHECK:   # Load Command 0
; CHECK:  (('command', 25)
; CHECK:   ('size', 232)
; CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
; CHECK:   ('vm_addr', 0)
; CHECK:   ('vm_size', 84)
; CHECK:   ('file_offset', 368)
; CHECK:   ('file_size', 84)
; CHECK:   ('maxprot', 7)
; CHECK:   ('initprot', 7)
; CHECK:   ('num_sections', 2)
; CHECK:   ('flags', 0)
; CHECK:   ('sections', [
; CHECK:     # Section 0
; CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
; CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
; CHECK:     ('address', 0)
; CHECK:     ('size', 36)
; CHECK:     ('offset', 368)
; CHECK:     ('alignment', 0)
; CHECK:     ('reloc_offset', 452)
; CHECK:     ('num_reloc', 13)
; CHECK:     ('flags', 0x80000400)
; CHECK:     ('reserved1', 0)
; CHECK:     ('reserved2', 0)
; CHECK:     ('reserved3', 0)
; CHECK:    ),
; CHECK:   ('_relocations', [
; CHECK:     # Relocation 0
; CHECK:     (('word-0', 0x20),
; CHECK:      ('word-1', 0x6c000005)),
; CHECK:     # Relocation 1
; CHECK:     (('word-0', 0x1c),
; CHECK:      ('word-1', 0x5d000005)),
; CHECK:     # Relocation 2
; CHECK:     (('word-0', 0x18),
; CHECK:      ('word-1', 0xa4000004)),
; CHECK:     # Relocation 3
; CHECK:     (('word-0', 0x18),
; CHECK:      ('word-1', 0x4c000002)),
; CHECK:     # Relocation 4
; CHECK:     (('word-0', 0x14),
; CHECK:      ('word-1', 0xa4000001)),
; CHECK:     # Relocation 5
; CHECK:     (('word-0', 0x14),
; CHECK:      ('word-1', 0x3d000002)),
; CHECK:     # Relocation 6
; CHECK:     (('word-0', 0x10),
; CHECK:      ('word-1', 0xa4000004)),
; CHECK:     # Relocation 7
; CHECK:     (('word-0', 0x10),
; CHECK:      ('word-1', 0x4c000002)),
; CHECK:     # Relocation 8
; CHECK:     (('word-0', 0xc),
; CHECK:      ('word-1', 0x4c000002)),
; CHECK:     # Relocation 9
; CHECK:     (('word-0', 0x8),
; CHECK:      ('word-1', 0x3d000002)),
; CHECK:     # Relocation 10
; CHECK:     (('word-0', 0x4),
; CHECK:      ('word-1', 0xa4000014)),
; CHECK:     # Relocation 11
; CHECK:     (('word-0', 0x4),
; CHECK:      ('word-1', 0x2d000007)),
; CHECK:     # Relocation 12
; CHECK:     (('word-0', 0x0),
; CHECK:      ('word-1', 0x2d000007)),
; CHECK:   ])
; CHECK:   ('_section_data', '00000094 00000094 03000090 620040b9 63000091 03000090 620040b9 03000090 620040b9')
; CHECK:     # Section 1
; CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
; CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
; CHECK:     ('address', 36)
; CHECK:     ('size', 48)
; CHECK:     ('offset', 404)
; CHECK:     ('alignment', 0)
; CHECK:     ('reloc_offset', 556)
; CHECK:     ('num_reloc', 10)
; CHECK:     ('flags', 0x0)
; CHECK:     ('reserved1', 0)
; CHECK:     ('reserved2', 0)
; CHECK:     ('reserved3', 0)
; CHECK:    ),
; CHECK:   ('_relocations', [
; CHECK:     # Relocation 0
; CHECK:     (('word-0', 0x2c),
; CHECK:      ('word-1', 0x7d000006)),
; CHECK:     # Relocation 1
; CHECK:     (('word-0', 0x24),
; CHECK:      ('word-1', 0x7e000006)),
; CHECK:     # Relocation 2
; CHECK:     (('word-0', 0x20),
; CHECK:      ('word-1', 0x1c000004)),
; CHECK:     # Relocation 3
; CHECK:     (('word-0', 0x20),
; CHECK:      ('word-1', 0xc000006)),
; CHECK:     # Relocation 4
; CHECK:     (('word-0', 0x18),
; CHECK:      ('word-1', 0x1e000004)),
; CHECK:     # Relocation 5
; CHECK:     (('word-0', 0x18),
; CHECK:      ('word-1', 0xe000006)),
; CHECK:     # Relocation 6
; CHECK:     (('word-0', 0x10),
; CHECK:      ('word-1', 0x1e000004)),
; CHECK:     # Relocation 7
; CHECK:     (('word-0', 0x10),
; CHECK:      ('word-1', 0xe000006)),
; CHECK:     # Relocation 8
; CHECK:     (('word-0', 0x8),
; CHECK:      ('word-1', 0xe000006)),
; CHECK:     # Relocation 9
; CHECK:     (('word-0', 0x0),
; CHECK:      ('word-1', 0xe000006)),
; CHECK:   ])
; CHECK:   ('_section_data', '00000000 00000000 04000000 00000000 00000000 00000000 04000000 00000000 00000000 00000000 00000000 d4ffffff')
; CHECK:   ])
; CHECK:  ),
