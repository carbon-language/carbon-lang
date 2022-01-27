; REQUIRES: x86
; NetBSD: noatime mounts currently inhibit 'touch' from updating atime
; UNSUPPORTED: system-netbsd

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: opt -module-hash -module-summary %p/Inputs/lto-cache.ll -o %t2.o

; RUN: rm -Rf %t.cache && mkdir %t.cache
; Create two files that would be removed by cache pruning due to age.
; We should only remove files matching "llvmcache-*" or "Thin-*".
; RUN: touch -t 197001011200 %t.cache/llvmcache-foo %t.cache/Thin-123.tmp.o %t.cache/foo
; RUN: lld-link /lldltocache:%t.cache /lldltocachepolicy:prune_after=1h /out:%t3 /entry:main %t2.o %t.o

; Two cached objects, plus a timestamp file and "foo", minus the file we removed.
; RUN: ls %t.cache | count 4

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @globalfunc() #0 {
entry:
  ret void
}
