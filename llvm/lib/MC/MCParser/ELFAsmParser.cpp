//===- ELFAsmParser.cpp - ELF Assembly Parser -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
using namespace llvm;

namespace {

class ELFAsmParser : public MCAsmParserExtension {
  bool ParseSectionSwitch(StringRef Section, unsigned Type,
                          unsigned Flags, SectionKind Kind);

public:
  ELFAsmParser() {}

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    Parser.AddDirectiveHandler(this, ".data", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseSectionDirectiveData));
    Parser.AddDirectiveHandler(this, ".text", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseSectionDirectiveText));
  }

  bool ParseSectionDirectiveData(StringRef, SMLoc) {
    return ParseSectionSwitch(".data", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_WRITE |MCSectionELF::SHF_ALLOC,
                              SectionKind::getDataRel());
  }
  bool ParseSectionDirectiveText(StringRef, SMLoc) {
    return ParseSectionSwitch(".text", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_EXECINSTR |
                              MCSectionELF::SHF_ALLOC, SectionKind::getText());
  }
};

}

bool ELFAsmParser::ParseSectionSwitch(StringRef Section, unsigned Type,
                                      unsigned Flags, SectionKind Kind) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lex();

  getStreamer().SwitchSection(getContext().getELFSection(
                                Section, Type, Flags, Kind));

  return false;
}

namespace llvm {

MCAsmParserExtension *createELFAsmParser() {
  return new ELFAsmParser;
}

}
