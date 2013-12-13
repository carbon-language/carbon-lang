// RUN: %clang -target x86_64-apple-darwin -arch armv6m -resource-dir=%S/Inputs/resource_dir %s -### 2> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -mhard-float -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t

// RUN: %clang -target x86_64-apple-darwin -arch armv7m -fPIC -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -fPIC -mfloat-abi=hard -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -fPIC -mfloat-abi=softfp -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t

// RUN: FileCheck %s < %t

// ARMv6m has no float
// CHECK: libclang_rt.soft_static.a

// ARMv7em does, but defaults to soft
// CHECK: libclang_rt.soft_static.a

// Which can be overridden
// CHECK: libclang_rt.hard_static.a

// ARMv7m has no float either
// CHECK: libclang_rt.soft_pic.a

// But it can be enabled on ARMv7em
// CHECK: libclang_rt.hard_pic.a

// "softfp" must link against a soft-float library since that's what the
// callers we're compiling will expect.
// CHECK: libclang_rt.soft_pic.a

// FIXME: test ARMv7a when we switch to -none-macho as the triple
