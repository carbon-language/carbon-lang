 ; RUN: llc -filetype=obj -march=mips64el -mcpu=mips64 %s -o - \
 ; RUN: | llvm-readobj -s | FileCheck --check-prefix=CHECK_64 %s
 ; RUN: llc -filetype=obj -march=mipsel -mcpu=mips32 %s -o - \
 ; RUN: | llvm-readobj -s | FileCheck --check-prefix=CHECK_32 %s

; Check for register information sections.
;

@str = private unnamed_addr constant [12 x i8] c"hello world\00"

define i32 @main() nounwind {
entry:
; Check that the appropriate relocations were created.

; check for .MIPS.options
; CHECK_64:      Sections [
; CHECK_64:        Section {
; CHECK_64:          Name: .MIPS.options
; CHECK_64-NEXT:     Type: SHT_MIPS_OPTIONS
; CHECK_64-NEXT:     Flags [ (0x8000002)

; check for .reginfo
; CHECK_32:      Sections [
; CHECK_32:        Section {
; CHECK_32:          Name: .reginfo
; CHECK_32-NEXT:     Type:  SHT_MIPS_REGINFO
; CHECK_32-NEXT:     Flags [ (0x2)


  %puts = tail call i32 @puts(i8* getelementptr inbounds ([12 x i8]* @str, i64 0, i64 0))
  ret i32 0

}
declare i32 @puts(i8* nocapture) nounwind
