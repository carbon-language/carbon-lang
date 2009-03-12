//===--- Action.cpp - Abstract compilation steps ------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Action.h"

#include <cassert>
using namespace clang::driver;

Action::~Action() {}
