//===-- MCAsmParser.cpp - Abstract Asm Parser Interface -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

MCAsmParser::MCAsmParser() {
}

MCAsmParser::~MCAsmParser() {
}

const AsmToken &MCAsmParser::getTok() {
  return getLexer().getTok();
}

bool MCAsmParser::TokError(const char *Msg) {
  Error(getLexer().getLoc(), Msg);
  return true;
}

bool MCAsmParser::ParseExpression(const MCExpr *&Res) {
  SMLoc L;
  return ParseExpression(Res, L);
}

/// getStartLoc - Get the location of the first token of this operand.
SMLoc MCParsedAsmOperand::getStartLoc() const { return SMLoc(); }
SMLoc MCParsedAsmOperand::getEndLoc() const { return SMLoc(); }


