// RUN: %clang -### -target armv7-windows-itanium -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-BASIC

// CHECK-BASIC: ld" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definitions" "-o" "/dev/null" "/usr/lib/crtbegin.obj" "-L/usr/lib" "-L/usr/lib/gcc" "{{.*}}.o" "-lmsvcrt" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"

// RUN: %clang -### -target armv7-windows-itanium -rtlib=compiler-rt -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-RTLIB

// CHECK-RTLIB: ld" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definitions" "-o" "/dev/null" "/usr/lib/crtbegin.obj" "-L/usr/lib" "-L/usr/lib/gcc" "{{.*}}.o" "-lmsvcrt" "{{.*}}/libclang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot /sysroot/Windows/ARM/8.1 -rtlib=compiler-rt -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-SYSROOT

// CHECK-SYSROOT: ld" "--sysroot=/sysroot/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definitions" "-o" "/dev/null" "/sysroot/Windows/ARM/8.1/usr/lib/crtbegin.obj" "-L/sysroot/Windows/ARM/8.1/usr/lib" "-L/sysroot/Windows/ARM/8.1/usr/lib/gcc" "{{.*}}.o" "-lmsvcrt" "{{.*}}/libclang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot /sysroot/Windows/ARM/8.1 -rtlib=compiler-rt -stdlib=libc++ -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-C-LIBCXX

// CHECK-C-LIBCXX: ld" "--sysroot=/sysroot/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definitions" "-o" "/dev/null" "/sysroot/Windows/ARM/8.1/usr/lib/crtbegin.obj" "{{.*}}.o" "-lmsvcrt" "{{.*}}/libclang_rt.builtins-arm.lib"

// RUN: %clangxx -### -target armv7-windows-itanium --sysroot /sysroot/Windows/ARM/8.1 -rtlib=compiler-rt -stdlib=libc++ -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-LIBCXX

// CHECK-LIBCXX: ld" "--sysroot=/sysroot/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definitions" "-o" "/dev/null" "/sysroot/Windows/ARM/8.1/usr/lib/crtbegin.obj" "{{.*}}.o" "-lc++" "-lmsvcrt" "{{.*}}/libclang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot /sysroot/Windows/ARM/8.1 -shared -rtlib=compiler-rt -stdlib=libc++ -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-SHARED

// CHECK-SHARED: ld" "--sysroot=/sysroot/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definitions" "-o" "shared.dll" "--out-implib" "shared.lib" "/sysroot/Windows/ARM/8.1/usr/lib/crtbeginS.obj" "{{.*}}.o" "-lmsvcrt" "{{.*}}/libclang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot /sysroot/Windows/ARM/8.1 -shared -rtlib=compiler-rt -stdlib=libc++ -nostartfiles -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-NOSTARTFILES

// CHECK-NOSTARTFILES: ld" "--sysroot=/sysroot/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definitions" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}.o" "-lmsvcrt" "{{.*}}/libclang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot /sysroot/Windows/ARM/8.1 -shared -rtlib=compiler-rt -stdlib=libc++ -nostartfiles -nodefaultlibs -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-STANDALONE

// CHECK-STANDALONE: ld" "--sysroot=/sysroot/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definitions" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}.o"

