//===-- GetOpt.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#ifndef _MSC_VER

#include <unistd.h>
#include <getopt.h>

#else

#include <lldb/Host/windows/getopt/GetOptInc.h>

#endif
