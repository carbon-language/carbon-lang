//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <stdlib.h>
// ::aligned_alloc

// UNSUPPORTED: c++03, c++11, c++14

// ::aligned_alloc is provided by the C library, but it's marked as unavailable
// until macOS 10.15
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// ::aligned_alloc is not implemented on Windows
// XFAIL: target={{.+}}-windows-{{.+}}

#include <stdlib.h>
#include <type_traits>

static_assert(std::is_same<decltype(aligned_alloc(1, 0)), void*>::value, "");
