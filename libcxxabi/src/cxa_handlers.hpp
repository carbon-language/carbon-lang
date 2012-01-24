//===------------------------- cxa_handlers.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
// This file implements the functionality associated with the terminate_handler,
//   unexpected_handler, and new_handler.
//===----------------------------------------------------------------------===//

#include <exception>

namespace std
{

__attribute__((visibility("hidden"), noreturn))
void
__unexpected(unexpected_handler func);

__attribute__((visibility("hidden"), noreturn))
void
__terminate(terminate_handler func) _NOEXCEPT;

}  // std
