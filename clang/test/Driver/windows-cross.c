// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-BASIC

// CHECK-BASIC: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/crtbegin.obj" "-L{{.*}}/Inputs/Windows/ARM/8.1/usr/lib" "-L{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/gcc" "{{.*}}.o" "-lmsvcrt" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %s/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -rtlib=compiler-rt -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-RTLIB

// CHECK-RTLIB: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/crtbegin.obj" "-L{{.*}}/Inputs/Windows/ARM/8.1/usr/lib" "-L{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/gcc" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -rtlib=compiler-rt -stdlib=libc++ -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-C-LIBCXX

// CHECK-C-LIBCXX: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/crtbegin.obj" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clangxx -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -rtlib=compiler-rt -stdlib=libc++ -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-LIBCXX

// CHECK-LIBCXX: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-Bdynamic" "--entry" "mainCRTStartup" "--allow-multiple-definition" "-o" "{{[^"]*}}" "{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/crtbegin.obj" "{{.*}}.o" "-lc++" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -shared -rtlib=compiler-rt -stdlib=libc++ -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-SHARED

// CHECK-SHARED: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definition" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}/Inputs/Windows/ARM/8.1/usr/lib/crtbeginS.obj" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %s/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -shared -rtlib=compiler-rt -stdlib=libc++ -nostartfiles -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-NOSTARTFILES

// CHECK-NOSTARTFILES: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definition" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}.o" "-lmsvcrt" "{{.*[\\/]}}clang_rt.builtins-arm.lib"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -shared -rtlib=compiler-rt -stdlib=libc++ -nostartfiles -nodefaultlibs -o shared.dll %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-STANDALONE

// CHECK-STANDALONE: armv7-windows-itanium-ld" "--sysroot={{.*}}/Inputs/Windows/ARM/8.1" "-m" "thumb2pe" "-shared" "-Bdynamic" "--enable-auto-image-base" "--entry" "_DllMainCRTStartup" "--allow-multiple-definition" "-o" "shared.dll" "--out-implib" "shared.lib" "{{.*}}.o"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %/Inputs/Windows/ARM/8.1/usr/bin -shared -o shared.dll -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-LIBSTDCXX

// CHECK-LIBSTDCXX:  "-internal-isystem" "{{.*}}/usr/include/c++" "-internal-isystem" "{{.*}}/usr/include/c++/armv7--windows-itanium" "-internal-isystem" "{{.*}}/usr/include/c++/backwards"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -shared -o shared.dll -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-FUSE-LD

// CHECK-FUSE-LD: "{{.*}}lld-link2"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -shared -o shared.dll -fsanitize=address -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-SANITIZE-ADDRESS

// CHECK-SANITIZE-ADDRESS: "-fsanitize=address"

// RUN: %clang -### -target armv7-windows-itanium --sysroot %S/Inputs/Windows/ARM/8.1 -B %S/Inputs/Windows/ARM/8.1/usr/bin -fuse-ld=lld-link2 -shared -o shared.dll -fsanitize=tsan -x c++ %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix CHECK-SANITIZE-TSAN

// CHECK-SANITIZE-TSAN: clang-3.8: error: unsupported argument 'tsan' to option 'fsanitize='
// CHECK-SANITIZE-TSAN-NOT: "-fsanitize={{.*}}"

