; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:31: error: invalid DWARF language 'DW_LANG_NoSuchLanguage'
!0 = !DICompileUnit(language: DW_LANG_NoSuchLanguage,
                    file: !DIFile(filename: "a", directory: "b"))
