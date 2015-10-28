// RUN: llvm-mc -triple i386-apple-ios %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=IOS
// RUN: llvm-mc -triple i386-apple-watchos %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=WATCHOS
// RUN: llvm-mc -triple i386-apple-tvos %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=TVOS
// RUN: llvm-mc -triple i386-apple-macosx %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=MACOSX

.ios_version_min 1,2,3
// WATCHOS: version-min-diagnostics2.s:[[@LINE-1]]:1: warning: .ios_version_min should only be used for ios targets
// TVOS: version-min-diagnostics2.s:[[@LINE-2]]:1: warning: .ios_version_min should only be used for ios targets
// MACOSX: version-min-diagnostics2.s:[[@LINE-3]]:1: warning: .ios_version_min should only be used for ios targets
// IOS-NOT: warning: .ios_version_min should only be used for ios targets

.macosx_version_min 4,5,6
// WATCHOS: version-min-diagnostics2.s:[[@LINE-1]]:1: warning: .macosx_version_min should only be used for macosx targets
// TVOS: version-min-diagnostics2.s:[[@LINE-2]]:1: warning: .macosx_version_min should only be used for macosx targets
// IOS: version-min-diagnostics2.s:[[@LINE-3]]:1: warning: .macosx_version_min should only be used for macosx targets
// MACOSX-NOT: warning: .macosx_version_min should only be used for macosx targets
// CHECK: version-min-diagnostics2.s:[[@LINE-5]]:1: warning: overriding previous version_min directive
// CHECK: version-min-diagnostics2.s:[[@LINE-12]]:1: note: previous definition is here

.tvos_version_min 7,8,9
// WATCHOS: version-min-diagnostics2.s:[[@LINE-1]]:1: warning: .tvos_version_min should only be used for tvos targets
// MACOSX: version-min-diagnostics2.s:[[@LINE-2]]:1: warning: .tvos_version_min should only be used for tvos targets
// IOS: version-min-diagnostics2.s:[[@LINE-3]]:1: warning: .tvos_version_min should only be used for tvos targets
// TVOS-NOT: warning: .tvos_version_min should only be used for tvos targets
// CHECK: version-min-diagnostics2.s:[[@LINE-5]]:1: warning: overriding previous version_min directive
// CHECK: version-min-diagnostics2.s:[[@LINE-14]]:1: note: previous definition is here

.watchos_version_min 10,11,12
// MACOSX: version-min-diagnostics2.s:[[@LINE-1]]:1: warning: .watchos_version_min should only be used for watchos targets
// IOS: version-min-diagnostics2.s:[[@LINE-2]]:1: warning: .watchos_version_min should only be used for watchos targets
// TVOS-NOT: warning: .tvos_version_min should only be used for tvos targets
// WATCHOS-NOT: warning: .watchos_version_min should only be used for watchos targets
// CHECK: version-min-diagnostics2.s:[[@LINE-5]]:1: warning: overriding previous version_min directive
// CHECK: version-min-diagnostics2.s:[[@LINE-14]]:1: note: previous definition is here
