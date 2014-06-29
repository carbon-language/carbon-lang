// RUN: %clang -### -target i686-windows-cygnus %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CHECK-EXE -check-prefix CHECK

// RUN: %clang -shared -### -target i686-windows-cygnus %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CHECK-SHARED -check-prefix CHECK

// RUN: %clang -static -### -target i686-windows-cygnus %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CHECK-STATIC -check-prefix CHECK

// CHECK: "{{.*}}ld"
// CHECK: "--wrap" "_Znwj"
// CHECK: "--wrap" "_Znaj"
// CHECK: "--wrap" "_ZdlPv"
// CHECK: "--wrap" "_ZdaPv"
// CHECK: "--wrap" "_ZnwjRKSt9nothrow_t"
// CHECK: "--wrap" "_ZnajRKSt9nothrow_t"
// CHECK: "--wrap" "_ZdlPvRKSt9nothrow_t"
// CHECK: "--wrap" "_ZdaPvRKSt9nothrow_t"
// CHECK-SHARED: "--shared"
// CHECK-STATIC: "-Bstatic"
// CHECK-DYNAMIC: "-Bdynamic"
// CHECK-EXE: "-Bdynamic"
// CHECK-SHARED: "--enable-auto-image-base"
// CHECK-SHARED: "-e" "__cygwin_dll_entry@12"
// CHECK: "--dll-search-prefix=cyg"
// CHECK-EXE: "--large-address-aware"
// CHECK-STATIC: "--large-address-aware"
// CHECK-EXE: "--tsaware"
// CHECK-STATIC: "--tsaware"
// CHECK: .o"
// CHECK-EXE: crt0.o"
// CHECK-STATIC: crt0.o"
// CHECK: crtbegin.o"
// CHECK: "-L/usr/lib"
// CHECK: "-o"
// CHECK-EXE: "-lgcc_s"
// CHECK: "-lgcc"
// CHECK-STATIC: "-lgcc_eh"
// CHECK: "-lcygwin"
// CHECK: "-ladvapi32"
// CHECK: "-lshell32"
// CHECK: "-luser32"
// CHECK: "-lkernel32"
// CHECK-EXE: "-lgcc_s"
// CHECK: "-lgcc"
// CHECK-STATIC: "-lgcc_eh"
// CHECK: crtend.o"

