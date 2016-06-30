; RUN: opt -mtriple=i686-windows-msvc -S -winehprepare %s | FileCheck %s

declare i32 @_except_handler3(...)

define void @test1a() personality i32 (...)* @_except_handler3 {
; CHECK: define void @test1a() personality i32 (...)* @_except_handler3
entry:
  ret void
}

define void @test1b() ssp personality i32 (...)* @_except_handler3 {
; CHECK: define void @test1b() [[attr:.*]] personality i32 (...)* @_except_handler4
entry:
  ret void
}

