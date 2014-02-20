# These *MUST* match the output of gas compiled with the same triple and
# corresponding options (-mabi=64 -> -mattr=+n64 for example).

# RUN: llvm-mc -filetype=obj -triple=mips64el-linux -mattr=-n64,+n64 %s -o - \
# RUN: | llvm-readobj -s | FileCheck --check-prefix=CHECK_64 %s
# RUN: llvm-mc -filetype=obj -triple=mipsel %s -mattr=-o32,+n32 -o - \
# RUN: | llvm-readobj -s | FileCheck --check-prefix=CHECK_32 %s

# Check for register information sections.
#

# Check that the appropriate relocations were created.

# check for .MIPS.options
# CHECK_64:      Sections [
# CHECK_64:        Section {
# CHECK_64:          Name: .MIPS.options
# CHECK_64-NEXT:     Type: SHT_MIPS_OPTIONS
# CHECK_64-NEXT:     Flags [ (0x8000002)

# check for .reginfo
# CHECK_32:      Sections [
# CHECK_32:        Section {
# CHECK_32:          Name: .reginfo
# CHECK_32-NEXT:     Type:  SHT_MIPS_REGINFO
# CHECK_32-NEXT:     Flags [ (0x2)
