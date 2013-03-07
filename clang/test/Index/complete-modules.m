// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@import LibA.Extensions;

// RUN: rm -rf %t
// RUN: c-index-test -code-completion-at=%s:4:9 -fmodules-cache-path=%t -fmodules -F %S/Inputs/Frameworks -I %S/Inputs/Headers %s | FileCheck -check-prefix=CHECK-TOP-LEVEL %s
// CHECK-TOP-LEVEL: NotImplemented:{TypedText Framework} (50)
// CHECK-TOP-LEVEL: NotImplemented:{TypedText LibA} (50)
// CHECK-TOP-LEVEL: NotImplemented:{TypedText nested} (50)

// RUN: c-index-test -code-completion-at=%s:4:14 -fmodules-cache-path=%t -fmodules -F %S/Inputs/Frameworks -I %S/Inputs/Headers %s | FileCheck -check-prefix=CHECK-LIBA %s
// CHECK-LIBA: NotImplemented:{TypedText Extensions} (50)

// RUN: c-index-test -code-completion-at=%s:4:1 -fmodules-cache-path=%t -fmodules -F %S/Inputs/Frameworks -I %S/Inputs/Headers %s | FileCheck -check-prefix=CHECK-TOP %s
// CHECK-TOP: NotImplemented:{TypedText @import}{HorizontalSpace  }{Placeholder module} (40)

