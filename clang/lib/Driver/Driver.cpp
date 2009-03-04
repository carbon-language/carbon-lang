//===--- Driver.cpp - Clang GCC Compatible Driver -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Options.h"
using namespace clang::driver;

Driver::Driver() : Opts(new OptTable()) {
  
}

Driver::~Driver() {
  delete Opts;
}

Compilation *Driver::BuildCompilation(int argc, const char **argv) {
  return new Compilation();
}
