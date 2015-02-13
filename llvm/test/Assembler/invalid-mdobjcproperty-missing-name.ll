; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:38: error: missing required field 'name'
!0 = !MDObjCProperty(setter: "setFoo")
