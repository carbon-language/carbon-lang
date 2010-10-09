//===- COFFAsmParser.cpp - COFF Assembly Parser ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/COFF.h"
using namespace llvm;

namespace {

class COFFAsmParser : public MCAsmParserExtension {
  template<bool (COFFAsmParser::*Handler)(StringRef, SMLoc)>
  void AddDirectiveHandler(StringRef Directive) {
    getParser().AddDirectiveHandler(this, Directive,
                                    HandleDirective<COFFAsmParser, Handler>);
  }

  bool ParseSectionSwitch(StringRef Section,
                          unsigned Characteristics,
                          SectionKind Kind);

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    MCAsmParserExtension::Initialize(Parser);

    AddDirectiveHandler<&COFFAsmParser::ParseSectionDirectiveText>(".text");
    AddDirectiveHandler<&COFFAsmParser::ParseSectionDirectiveData>(".data");
    AddDirectiveHandler<&COFFAsmParser::ParseSectionDirectiveBSS>(".bss");
    AddDirectiveHandler<&COFFAsmParser::ParseDirectiveDef>(".def");
    AddDirectiveHandler<&COFFAsmParser::ParseDirectiveScl>(".scl");
    AddDirectiveHandler<&COFFAsmParser::ParseDirectiveType>(".type");
    AddDirectiveHandler<&COFFAsmParser::ParseDirectiveEndef>(".endef");
  }

  bool ParseSectionDirectiveText(StringRef, SMLoc) {
    return ParseSectionSwitch(".text",
                              COFF::IMAGE_SCN_CNT_CODE
                            | COFF::IMAGE_SCN_MEM_EXECUTE
                            | COFF::IMAGE_SCN_MEM_READ,
                              SectionKind::getText());
  }
  bool ParseSectionDirectiveData(StringRef, SMLoc) {
    return ParseSectionSwitch(".data",
                              COFF::IMAGE_SCN_CNT_INITIALIZED_DATA
                            | COFF::IMAGE_SCN_MEM_READ
                            | COFF::IMAGE_SCN_MEM_WRITE,
                              SectionKind::getDataRel());
  }
  bool ParseSectionDirectiveBSS(StringRef, SMLoc) {
    return ParseSectionSwitch(".bss",
                              COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA
                            | COFF::IMAGE_SCN_MEM_READ
                            | COFF::IMAGE_SCN_MEM_WRITE,
                              SectionKind::getBSS());
  }

  bool ParseDirectiveDef(StringRef, SMLoc);
  bool ParseDirectiveScl(StringRef, SMLoc);
  bool ParseDirectiveType(StringRef, SMLoc);
  bool ParseDirectiveEndef(StringRef, SMLoc);

public:
  COFFAsmParser() {}
};

} // end annonomous namespace.

bool COFFAsmParser::ParseSectionSwitch(StringRef Section,
                                       unsigned Characteristics,
                                       SectionKind Kind) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lex();

  getStreamer().SwitchSection(getContext().getCOFFSection(
                                Section, Characteristics, Kind));

  return false;
}

bool COFFAsmParser::ParseDirectiveDef(StringRef, SMLoc) {
  StringRef SymbolName;

  if (getParser().ParseIdentifier(SymbolName))
    return TokError("expected identifier in directive");

  MCSymbol *Sym = getContext().GetOrCreateSymbol(SymbolName);

  getStreamer().BeginCOFFSymbolDef(Sym);

  Lex();
  return false;
}

bool COFFAsmParser::ParseDirectiveScl(StringRef, SMLoc) {
  int64_t SymbolStorageClass;
  if (getParser().ParseAbsoluteExpression(SymbolStorageClass))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().EmitCOFFSymbolStorageClass(SymbolStorageClass);
  return false;
}

bool COFFAsmParser::ParseDirectiveType(StringRef, SMLoc) {
  int64_t Type;
  if (getParser().ParseAbsoluteExpression(Type))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().EmitCOFFSymbolType(Type);
  return false;
}

bool COFFAsmParser::ParseDirectiveEndef(StringRef, SMLoc) {
  Lex();
  getStreamer().EndCOFFSymbolDef();
  return false;
}

namespace llvm {

MCAsmParserExtension *createCOFFAsmParser() {
  return new COFFAsmParser;
}

}
