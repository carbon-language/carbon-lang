//===--- Driver.cpp - Clang GCC Compatible Driver -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
using namespace clang;

Driver::Driver() {
}

Driver::~Driver() {
}

Compilation *Driver::BuildCompilation(int argc, const char **argv) {
  return new Compilation();
}

                                      
