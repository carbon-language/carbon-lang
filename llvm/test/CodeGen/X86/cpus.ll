; Test that the CPU names work.
;
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=nocona 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=core2 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=penryn 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=nehalem 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=westmere 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=sandybridge 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=ivybridge 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=haswell 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=broadwell 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bonnell 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=silvermont 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k8 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=opteron 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon64 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-fx 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k8-sse3 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=opteron-sse3 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon64-sse3 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=amdfam10 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=barcelona 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver1 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver2 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver3 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver4 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=btver1 2>&1 | count 0
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=btver2 2>&1 | count 0
