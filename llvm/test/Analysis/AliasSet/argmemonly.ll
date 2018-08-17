; RUN: opt -basicaa -print-alias-sets -S -o - < %s 2>&1 | FileCheck %s

@s = global i8 1, align 1
@d = global i8 2, align 1

; CHECK: Alias sets for function 'test1':
; CHECK-NEXT: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Mod/Ref   
; CHECK-NEXT: 1 Unknown instructions:   call void @my_memcpy(i8* %d, i8* %s, i64 1)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test1(i8* %s, i8* %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @my_memcpy(i8* %d, i8* %s, i64 1)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test3':
; CHECK-NEXT: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Mod/Ref   
; CHECK-NEXT:  Unknown instructions:   call void @my_memmove(i8* %d, i8* %s, i64 1)

; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test3(i8* noalias %a, i8* noalias %b, i8* %s, i8* %d) {
entry:
  store i8 1, i8* %a, align 1
  call void @my_memmove(i8* %d, i8* %s, i64 1)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test5':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref Pointers: (i8* %a, 1), (i8* %b, 1)
; CHECK-NEXT: 1 Unknown instructions: call void @my_memcpy(i8* %b, i8* %a, i64 1)
define void @test5(i8* noalias %a, i8* noalias %b) {
entry:
  store i8 1, i8* %a, align 1
  call void @my_memcpy(i8* %b, i8* %a, i64 1)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test6':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref Pointers: (i8* %a, 1), (i8* %b, 1)
; CHECK-NEXT: 1 Unknown instructions: call void @my_memmove(i8* %b, i8* %a, i64 1)
define void @test6(i8* noalias %a, i8* noalias %b) {
entry:
  store i8 1, i8* %a, align 1
  call void @my_memmove(i8* %b, i8* %a, i64 1)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test7':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref Pointers: (i8* %a, 1), (i8* %b, 1)
; CHECK-NEXT: 2 Unknown instructions: call void @my_memcpy(i8* %b, i8* %a, i64 1),   call void @my_memcpy(i8* %a, i8* %b, i64 1)

define void @test7(i8* noalias %a, i8* noalias %b) {
entry:
  store i8 1, i8* %a, align 1
  call void @my_memcpy(i8* %b, i8* %a, i64 1)
  call void @my_memcpy(i8* %a, i8* %b, i64 1)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test_memset1':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 0 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Mod/Ref
; CHECK-NEXT: 1 Unknown instructions: call void @my_memset(i8* %a, i8 0, i64 1)
define void @test_memset1() {
entry:
  %a = alloca i8, align 1
  call void @my_memset(i8* %a, i8 0, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset2':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 0 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Mod/Ref
; CHECK-NEXT: 1 Unknown instructions: call void @my_memset(i8* %a, i8 0, i64 1)
define void @test_memset2(i8* %a) {
entry:
  call void @my_memset(i8* %a, i8 0, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset3':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 0 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Mod/Ref
; CHECK-NEXT: 2 Unknown instructions:   call void @my_memset(i8* %a, i8 0, i64 1),   call void @my_memset(i8* %b, i8 0, i64 1)
define void @test_memset3(i8* %a, i8* %b) {
entry:
  call void @my_memset(i8* %a, i8 0, i64 1)
  call void @my_memset(i8* %b, i8 0, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset4':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 0 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Mod/Ref
; CHECK-NEXT: 1 Unknown instructions:   call void @my_memset(i8* %a, i8 0, i64 1)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Mod/Ref
; CHECK-NEXT: 1 Unknown instructions:   call void @my_memset(i8* %b, i8 0, i64 1)
define void @test_memset4(i8* noalias %a, i8* noalias %b) {
entry:
  call void @my_memset(i8* %a, i8 0, i64 1)
  call void @my_memset(i8* %b, i8 0, i64 1)
  ret void
}

declare void @my_memset(i8* nocapture writeonly, i8, i64) argmemonly
declare void @my_memcpy(i8* nocapture writeonly, i8* nocapture readonly, i64) argmemonly
declare void @my_memmove(i8* nocapture, i8* nocapture readonly, i64) argmemonly
