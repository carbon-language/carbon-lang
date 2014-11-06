; RUN: opt < %s -asan -asan-module -S | FileCheck %s

target datalayout = "e"

@foo = private global [19 x i8] c"scannerWithString:\00", section "__TEXT,__objc_methname,cstring_literals"

; CHECK: @foo = private global [19 x i8] c"scannerWithString:\00", section "__TEXT,__objc_methname,cstring_literals"