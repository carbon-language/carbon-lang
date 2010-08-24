; This test checks that the COFF object emitter works for the most basic
; programs.

; RUN: llc -filetype=obj -mtriple i686-pc-win32 %s -o %t
; RUN: coff-dump.py %abs_tmp | FileCheck %s
; RUN: llc -filetype=obj -mtriple x86_64-pc-win32 %s -o %t

@.str = private constant [12 x i8] c"Hello World\00" ; <[12 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([12 x i8]* @.str, i32 0, i32 0)) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind

; CHECK: {
; CHECK:   MachineType              = IMAGE_FILE_MACHINE_I386 (0x14C)
; CHECK:   NumberOfSections         = 2
; CHECK:   TimeDateStamp            = {{[0-9]+}}
; CHECK:   PointerToSymbolTable     = 0x99
; CHECK:   NumberOfSymbols          = 7
; CHECK:   SizeOfOptionalHeader     = 0
; CHECK:   Characteristics          = 0x0
; CHECK:   Sections                 = [
; CHECK:     0 = {
; CHECK:       Name                     = .text
; CHECK:       VirtualSize              = 0
; CHECK:       VirtualAddress           = 0
; CHECK:       SizeOfRawData            = 21
; CHECK:       PointerToRawData         = 0x64
; CHECK:       PointerToRelocations     = 0x79
; CHECK:       PointerToLineNumbers     = 0x0
; CHECK:       NumberOfRelocations      = 2
; CHECK:       NumberOfLineNumbers      = 0
; CHECK:       Charateristics           = 0x60500020
; CHECK:         IMAGE_SCN_CNT_CODE
; CHECK:         IMAGE_SCN_ALIGN_16BYTES
; CHECK:         IMAGE_SCN_MEM_EXECUTE
; CHECK:         IMAGE_SCN_MEM_READ
; CHECK:       SectionData              =
; CHECK:         83 EC 04 C7 04 24 00 00 - 00 00 E8 00 00 00 00 31 |.....$.........1|
; CHECK:         C0 83 C4 04 C3                                    |.....|
; CHECK:       Relocations              = [
; CHECK:         0 = {
; CHECK:           VirtualAddress           = 0x6
; CHECK:           SymbolTableIndex         = 5
; CHECK:           Type                     = IMAGE_REL_I386_DIR32 (6)
; CHECK:           SymbolName               = _main
; CHECK:         }
; CHECK:         1 = {
; CHECK:           VirtualAddress           = 0xB
; CHECK:           SymbolTableIndex         = 6
; CHECK:           Type                     = IMAGE_REL_I386_REL32 (20)
; CHECK:           SymbolName               = L_.str
; CHECK:         }
; CHECK:       ]
; CHECK:     }
; CHECK:     1 = {
; CHECK:       Name                     = .data
; CHECK:       VirtualSize              = 0
; CHECK:       VirtualAddress           = 0
; CHECK:       SizeOfRawData            = 12
; CHECK:       PointerToRawData         = 0x8D
; CHECK:       PointerToRelocations     = 0x0
; CHECK:       PointerToLineNumbers     = 0x0
; CHECK:       NumberOfRelocations      = 0
; CHECK:       NumberOfLineNumbers      = 0
; CHECK:       Charateristics           = 0xC0100040
; CHECK:         IMAGE_SCN_CNT_INITIALIZED_DATA
; CHECK:         IMAGE_SCN_ALIGN_1BYTES
; CHECK:         IMAGE_SCN_MEM_READ
; CHECK:         IMAGE_SCN_MEM_WRITE
; CHECK:       SectionData              =
; CHECK:         48 65 6C 6C 6F 20 57 6F - 72 6C 64 00             |Hello World.|
; CHECK:       Relocations              = None
; CHECK:     }
; CHECK:   ]
; CHECK:   Symbols                  = [
; CHECK:     0 = {
; CHECK:       Name                     = .text
; CHECK:       Value                    = 0
; CHECK:       SectionNumber            = 1
; CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
; CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
; CHECK:       StorageClass             = IMAGE_SYM_CLASS_STATIC (3)
; CHECK:       NumberOfAuxSymbols       = 1
; CHECK:       AuxillaryData            =
; CHECK:         15 00 00 00 02 00 00 00 - 00 00 00 00 01 00 00 00 |................|
; CHECK:         00 00                                             |..|
; CHECK:     }
; CHECK:     1 = {
; CHECK:       Name                     = .data
; CHECK:       Value                    = 0
; CHECK:       SectionNumber            = 2
; CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
; CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
; CHECK:       StorageClass             = IMAGE_SYM_CLASS_STATIC (3)
; CHECK:       NumberOfAuxSymbols       = 1
; CHECK:       AuxillaryData            =
; CHECK:         0C 00 00 00 00 00 00 00 - 00 00 00 00 02 00 00 00 |................|
; CHECK:         00 00                                             |..|
; CHECK:     }
; CHECK:     2 = {
; CHECK:       Name                     = _main
; CHECK:       Value                    = 0
; CHECK:       SectionNumber            = 1
; CHECK:       SimpleType               = unknown (32)
; CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
; CHECK:       StorageClass             = IMAGE_SYM_CLASS_EXTERNAL (2)
; CHECK:       NumberOfAuxSymbols       = 0
; CHECK:       AuxillaryData            =
; CHECK:     }
; CHECK:     3 = {
; CHECK:       Name                     = L_.str
; CHECK:       Value                    = 0
; CHECK:       SectionNumber            = 2
; CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
; CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
; CHECK:       StorageClass             = IMAGE_SYM_CLASS_STATIC (3)
; CHECK:       NumberOfAuxSymbols       = 0
; CHECK:       AuxillaryData            =
; CHECK:     }
; CHECK:     4 = {
; CHECK:       Name                     = _printf
; CHECK:       Value                    = 0
; CHECK:       SectionNumber            = 0
; CHECK:       SimpleType               = IMAGE_SYM_TYPE_NULL (0)
; CHECK:       ComplexType              = IMAGE_SYM_DTYPE_NULL (0)
; CHECK:       StorageClass             = IMAGE_SYM_CLASS_EXTERNAL (2)
; CHECK:       NumberOfAuxSymbols       = 0
; CHECK:       AuxillaryData            =
; CHECK:     }
; CHECK:   ]
; CHECK: }
