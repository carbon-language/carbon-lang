//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// See https://llvm.org/bugs/show_bug.cgi?id=20002

#include <functional>
#include <type_traits>

using Fn = std::function<void()>;
struct S : Fn { using function::function; };

int main() {
    S f1( Fn{} );
    S f2(std::allocator_arg, std::allocator<void>{}, Fn{});
}