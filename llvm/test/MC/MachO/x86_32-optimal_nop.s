// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

# 1 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nop
        # 0x90
        .align 1, 0x90
        ret
# 2 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # xchg %ax,%ax
        # 0x66, 0x90
        .align 2, 0x90
        ret
# 3 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl (%[re]ax)
        # 0x0f, 0x1f, 0x00
        .align 2, 0x90
        ret
# 4 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopl 0(%[re]ax)
        # 0x0f, 0x1f, 0x40, 0x00
        .align 3, 0x90
        ret
# 5 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopl 0(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 6 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 7 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 8 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 9 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw 0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 10 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 11 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 12 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 4, 0x90
        ret
# 13 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopl 0L(%[re]ax)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 14 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 15 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret

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
// CHECK:   ('vm_size', 337)
// CHECK:   ('file_offset', 152)
// CHECK:   ('file_size', 337)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 1)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 337)
// CHECK:     ('offset', 152)
// CHECK:     ('alignment', 4)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '\xc3\x90\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\xc3f\x90\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\x0f\x1f\x00\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3\xc3\x0f\x1f@\x00\xc3\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3\x0f\x1fD\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\xc3\xc3f\x0f\x1fD\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\xc3\x0f\x1f\x80\x00\x00\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3\xc3\xc3\xc3\xc3f\x0f\x1f\x84\x00\x00\x00\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3\xc3\xc3\xc3\xc3f\x0f\x1f\x84\x00\x00\x00\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3\xc3\xc3\x0f\x1fD\x00\x00f\x0f\x1fD\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3\xc3f\x0f\x1fD\x00\x00f\x0f\x1fD\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\xc3f\x0f\x1fD\x00\x00\x0f\x1f\x80\x00\x00\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\xc3\x0f\x1f\x80\x00\x00\x00\x00\x0f\x1f\x80\x00\x00\x00\x00\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc3\x0f\x1f\x80\x00\x00\x00\x00\x0f\x1f\x84\x00\x00\x00\x00\x00\xc3')
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
