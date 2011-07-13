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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

MCAsmParser::MCAsmParser() : TargetParser(0), ShowParsedOperands(0) {
}

MCAsmParser::~MCAsmParser() {
}

void MCAsmParser::setTargetParser(TargetAsmParser &P) {
  assert(!TargetParser && "Target parser is already initialized!");
  TargetParser = &P;
  TargetParser->Initialize(*this);
}

const AsmToken &MCAsmParser::getTok() {
  return getLexer().getTok();
}

bool MCAsmParser::TokError(const Twine &Msg) {
  Error(getLexer().getLoc(), Msg);
  return true;
}

bool MCAsmParser::ParseExpression(const MCExpr *&Res) {
  SMLoc L;
  return ParseExpression(Res, L);
}

void MCParsedAsmOperand::dump() const {
  dbgs() << "  " << *this;
}
