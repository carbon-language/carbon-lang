//===--- Tool.cpp - Compilation Tools -----------------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Tool.h"

using namespace clang::driver;

Tool::Tool(const char *_Name, const ToolChain &TC) : Name(_Name),
                                                     TheToolChain(TC) {
}

Tool::~Tool() {
}
