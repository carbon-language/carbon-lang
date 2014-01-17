// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: echo '@import test;' > %t-dir/prefix.h
// RUN: echo 'void foo(void);' > %t-dir/test.h
// RUN: cp %S/modified-module-dependency.module.map %t-dir/module.map

// Precompile prefix.pch.
// RUN: %clang_cc1 -x objective-c -I %t-dir -fmodules -fmodules-cache-path=%t-dir/cache -emit-pch %t-dir/prefix.h -o %t-dir/prefix.pch

// Modify the dependency.
// RUN: echo ' ' >> %t-dir/test.h

// Run and check the diagnostics.
// RUN: not %clang_cc1 -x objective-c -include-pch %t-dir/prefix.pch -fmodules -fmodules-cache-path=%t-dir/cache -fsyntax-only %s 2> %t-dir/log
// RUN: FileCheck %s < %t-dir/log

// CHECK: file '{{.*}}/test.h' has been modified since the precompiled header '{{.*}}prefix.pch' was built
// CHECK: '{{.*}}/test.h' required by '{{.*}}/test.pcm'
// CHECK: '{{.*}}/test.pcm' required by '{{.*}}/prefix.pch'
// CHECK: please rebuild precompiled header '{{.*}}/prefix.pch'
