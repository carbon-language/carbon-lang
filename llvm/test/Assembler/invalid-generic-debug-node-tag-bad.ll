; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:29: error: invalid DWARF tag 'DW_TAG_badtag'
!0 = !GenericDebugNode(tag: DW_TAG_badtag)
