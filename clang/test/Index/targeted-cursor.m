
// rdar://10920009
// RUN: c-index-test -write-pch %t.h.pch -fblocks -x objective-c-header %S/targeted-cursor.m.h -Xclang -detailed-preprocessing-record
// RUN: c-index-test -cursor-at=%S/targeted-cursor.m.h:5:13 %s -fblocks -include %t.h | FileCheck %s

// CHECK: ObjCClassRef=I:2:12
