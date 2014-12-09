// Ensure we support the various CPU names.
//
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu nocona
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu core2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu penryn
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu nehalem
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu westmere
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu sandybridge
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu ivybridge
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu haswell
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu broadwell
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu bonnell
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu silvermont
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu k8
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu opteron
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu athlon64
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu athlon-fx
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu k8-sse3
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu opteron-sse3
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu athlon64-sse3
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu amdfam10
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu barcelona
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu bdver1
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu bdver2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu bdver3
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu bdver4
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu btver1
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o /dev/null -target-cpu btver2
