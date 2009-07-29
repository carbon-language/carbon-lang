//===--- Handlers.cpp - Interfaces for receiving information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Abstract interfaces for receiving information.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/Handlers.h"
#include "clang/Index/Entity.h"
using namespace clang;
using namespace idx;

// Out-of-line to give the virtual tables a home.
EntityHandler::~EntityHandler() { }
TranslationUnitHandler::~TranslationUnitHandler() { }
TULocationHandler::~TULocationHandler() { }
