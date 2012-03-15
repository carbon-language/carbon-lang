//===------------------------- cxa_default_handlers.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
// This file declares the default terminate_handler and unexpected_handler.
//===----------------------------------------------------------------------===//


__attribute__((visibility("hidden"), noreturn))
void
default_terminate_handler();

__attribute__((visibility("hidden"), noreturn))
void
default_unexpected_handler();
