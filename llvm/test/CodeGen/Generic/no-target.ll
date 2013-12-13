; RUN: not llc -mtriple le32-unknown-nacl %s -o - 2>&1 | FileCheck %s

; CHECK: error: unable to get target for 'le32-unknown-nacl'
