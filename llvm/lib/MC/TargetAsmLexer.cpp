//===-- llvm/MC/TargetAsmLexer.cpp - Target Assembly Lexer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/TargetAsmLexer.h"
using namespace llvm;

TargetAsmLexer::TargetAsmLexer(const Target &T) : TheTarget(T), Lexer(NULL) {}
TargetAsmLexer::~TargetAsmLexer() {}
