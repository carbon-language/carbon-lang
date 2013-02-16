# RUN: llvm-mc -triple i386-pc-linux-gnu -filetype=obj -o %t %s
# RUN: elf-dump --dump-section-data < %t | FileCheck %s
.section .nobits
.byte 1
.section .nobits2
.byte 1
.section .nobitsfoo
.byte 1
.section .init_array
.byte 1
.section .init_array2
.byte 1
.section .init_arrayfoo
.byte 1
.section .fini_array
.byte 1
.section .fini_array2
.byte 1
.section .fini_arrayfoo
.byte 1
.section .preinit_array
.byte 1
.section .preinit_array2
.byte 1
.section .preinit_arrayfoo
.byte 1
.section .note
.byte 1
.section .note2
.byte 1
.section .notefoo
.byte 1
# CHECK:      (('sh_name', 0x00000{{...}}) # '.nobits'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.nobits2'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.nobitsfoo'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.init_array'
# CHECK-NEXT:  ('sh_type', 0x0000000e)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.init_array2'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.init_arrayfoo'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.fini_array'
# CHECK-NEXT:  ('sh_type', 0x0000000f)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.fini_array2'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.fini_arrayfoo'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.preinit_array'
# CHECK-NEXT:  ('sh_type', 0x00000010)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.preinit_array2'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.preinit_arrayfoo'
# CHECK-NEXT:  ('sh_type', 0x00000001)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.note'
# CHECK-NEXT:  ('sh_type', 0x00000007)
# CHECK:      (('sh_name', 0x00000{{...}}) # '.note2'
# CHECK-NEXT:  ('sh_type', 0x00000007)
#CHECK:       (('sh_name', 0x00000{{...}}) # '.notefoo'
# CHECK-NEXT:  ('sh_type', 0x00000007)
