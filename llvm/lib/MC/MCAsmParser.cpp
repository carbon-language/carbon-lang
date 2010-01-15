//===-- MCAsmParser.cpp - Abstract Asm Parser Interface -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmParser.h"
#include "llvm/MC/MCParsedAsmOperand.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

MCAsmParser::MCAsmParser() {
}

MCAsmParser::~MCAsmParser() {
}

bool MCAsmParser::ParseExpression(const MCExpr *&Res) {
  SMLoc L;
  return ParseExpression(Res, L, L);
}


/// getStartLoc - Get the location of the first token of this operand.
SMLoc MCParsedAsmOperand::getStartLoc() const { return SMLoc(); }
SMLoc MCParsedAsmOperand::getEndLoc() const { return SMLoc(); }


