//===-- llvm/MC/MCSymbolizer.cpp - MCSymbolizer class -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSymbolizer.h"
#include "llvm/MC/MCRelocationInfo.h"

using namespace llvm;

MCSymbolizer::MCSymbolizer(MCContext &Ctx,
                           std::unique_ptr<MCRelocationInfo> &RelInfo)
    : Ctx(Ctx), RelInfo(RelInfo.release()) {}

MCSymbolizer::~MCSymbolizer() {
}
