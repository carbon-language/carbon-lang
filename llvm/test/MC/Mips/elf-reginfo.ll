 ; RUN: llc -filetype=obj -march=mips64el -mcpu=mips64 %s -o - \
 ; RUN: | elf-dump --dump-section-data  | FileCheck --check-prefix=CHECK_64 %s
 ; RUN: llc -filetype=obj -march=mipsel -mcpu=mips32 %s -o - \
 ; RUN: | elf-dump --dump-section-data  | FileCheck --check-prefix=CHECK_32 %s

; Check for register information sections.
;

@str = private unnamed_addr constant [12 x i8] c"hello world\00"

define i32 @main() nounwind {
entry:
; Check that the appropriate relocations were created.

; check for .MIPS.options
; CHECK_64:      (('sh_name', 0x{{[0-9|a-f]+}}) # '.MIPS.options'
; CHECK_64-NEXT: ('sh_type', 0x7000000d)
; CHECK_64-NEXT: ('sh_flags', 0x0000000008000002)

; check for .reginfo
; CHECK_32:      (('sh_name', 0x{{[0-9|a-f]+}}) # '.reginfo'
; CHECK_32-NEXT: ('sh_type', 0x70000006)
; CHECK_32-NEXT: ('sh_flags', 0x00000002)


  %puts = tail call i32 @puts(i8* getelementptr inbounds ([12 x i8]* @str, i64 0, i64 0))
  ret i32 0

}
declare i32 @puts(i8* nocapture) nounwind
  
