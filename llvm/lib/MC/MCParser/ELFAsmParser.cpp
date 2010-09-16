//===- ELFAsmParser.cpp - ELF Assembly Parser -----------------------------===//
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
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

namespace {

class ELFAsmParser : public MCAsmParserExtension {
  template<bool (ELFAsmParser::*Handler)(StringRef, SMLoc)>
  void AddDirectiveHandler(StringRef Directive) {
    getParser().AddDirectiveHandler(this, Directive,
                                    HandleDirective<ELFAsmParser, Handler>);
  }

  bool ParseSectionSwitch(StringRef Section, unsigned Type,
                          unsigned Flags, SectionKind Kind);

public:
  ELFAsmParser() {}

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveData>(".data");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveText>(".text");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveBSS>(".bss");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveRoData>(".rodata");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveTData>(".tdata");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveTBSS>(".tbss");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveDataRel>(".data.rel");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveDataRelRo>(".data.rel.ro");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveDataRelRoLocal>(".data.rel.ro.local");
    AddDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveEhFrame>(".eh_frame");
    AddDirectiveHandler<&ELFAsmParser::ParseDirectiveSection>(".section");
    AddDirectiveHandler<&ELFAsmParser::ParseDirectiveSize>(".size");
    AddDirectiveHandler<&ELFAsmParser::ParseDirectivePrevious>(".previous");
  }

  // FIXME: Part of this logic is duplicated in the MCELFStreamer. What is
  // the best way for us to get access to it?
  bool ParseSectionDirectiveData(StringRef, SMLoc) {
    bool ret =  ParseSectionSwitch(".data", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_WRITE |MCSectionELF::SHF_ALLOC,
                              SectionKind::getDataRel());
    getStreamer().EmitCodeAlignment(4, 0);
    return ret;
  }
  bool ParseSectionDirectiveText(StringRef, SMLoc) {
    bool ret = ParseSectionSwitch(".text", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_EXECINSTR |
                              MCSectionELF::SHF_ALLOC, SectionKind::getText());
    getStreamer().EmitCodeAlignment(4, 0);
    return ret;
  }
  bool ParseSectionDirectiveBSS(StringRef, SMLoc) {
    bool ret = ParseSectionSwitch(".bss", MCSectionELF::SHT_NOBITS,
                              MCSectionELF::SHF_WRITE |
                              MCSectionELF::SHF_ALLOC, SectionKind::getBSS());
    getStreamer().EmitCodeAlignment(4, 0);
    return ret;
  }
  bool ParseSectionDirectiveRoData(StringRef, SMLoc) {
    return ParseSectionSwitch(".rodata", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_ALLOC,
                              SectionKind::getReadOnly());
  }
  bool ParseSectionDirectiveTData(StringRef, SMLoc) {
    return ParseSectionSwitch(".tdata", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_ALLOC |
                              MCSectionELF::SHF_TLS | MCSectionELF::SHF_WRITE,
                              SectionKind::getThreadData());
  }
  bool ParseSectionDirectiveTBSS(StringRef, SMLoc) {
    return ParseSectionSwitch(".tbss", MCSectionELF::SHT_NOBITS,
                              MCSectionELF::SHF_ALLOC |
                              MCSectionELF::SHF_TLS | MCSectionELF::SHF_WRITE,
                              SectionKind::getThreadBSS());
  }
  bool ParseSectionDirectiveDataRel(StringRef, SMLoc) {
    return ParseSectionSwitch(".data.rel", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_ALLOC |
                              MCSectionELF::SHF_WRITE,
                              SectionKind::getDataRel());
  }
  bool ParseSectionDirectiveDataRelRo(StringRef, SMLoc) {
    return ParseSectionSwitch(".data.rel.ro", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_ALLOC |
                              MCSectionELF::SHF_WRITE,
                              SectionKind::getReadOnlyWithRel());
  }
  bool ParseSectionDirectiveDataRelRoLocal(StringRef, SMLoc) {
    return ParseSectionSwitch(".data.rel.ro.local", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_ALLOC |
                              MCSectionELF::SHF_WRITE,
                              SectionKind::getReadOnlyWithRelLocal());
  }
  bool ParseSectionDirectiveEhFrame(StringRef, SMLoc) {
    return ParseSectionSwitch(".eh_frame", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_ALLOC |
                              MCSectionELF::SHF_WRITE,
                              SectionKind::getDataRel());
  }
  bool ParseDirectiveSection(StringRef, SMLoc);
  bool ParseDirectiveSize(StringRef, SMLoc);
  bool ParseDirectivePrevious(StringRef, SMLoc);

private:
  bool ParseSectionName(StringRef &SectionName);
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

bool ELFAsmParser::ParseDirectiveSize(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  const MCExpr *Expr;
  if (getParser().ParseExpression(Expr))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  getStreamer().EmitELFSize(Sym, Expr);
  return false;
}

bool ELFAsmParser::ParseSectionName(StringRef &SectionName) {
  // A section name can contain -, so we cannot just use
  // ParseIdentifier.
  SMLoc FirstLoc = getLexer().getLoc();
  unsigned Size = 0;

  for (;;) {
    StringRef Tmp;
    unsigned CurSize;

    SMLoc PrevLoc = getLexer().getLoc();
    if (getLexer().is(AsmToken::Minus)) {
      CurSize = 1;
      Lex(); // Consume the "-".
    } else if (!getParser().ParseIdentifier(Tmp))
      CurSize = Tmp.size();
    else
      break;

    Size += CurSize;
    SectionName = StringRef(FirstLoc.getPointer(), Size);

    // Make sure the following token is adjacent.
    if (PrevLoc.getPointer() + CurSize != getTok().getLoc().getPointer())
      break;
  }
  if (Size == 0)
    return true;

  return false;
}

// FIXME: This is a work in progress.
bool ELFAsmParser::ParseDirectiveSection(StringRef, SMLoc) {
  StringRef SectionName;

  if (ParseSectionName(SectionName))
    return TokError("expected identifier in directive");

  std::string FlagsStr;
  StringRef TypeName;
  int64_t Size = 0;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (getLexer().isNot(AsmToken::String))
      return TokError("expected string in directive");

    FlagsStr = getTok().getStringContents();
    Lex();

    AsmToken::TokenKind TypeStartToken;
    if (getContext().getAsmInfo().getCommentString()[0] == '@')
      TypeStartToken = AsmToken::Percent;
    else
      TypeStartToken = AsmToken::At;

    if (getLexer().is(AsmToken::Comma)) {
      Lex();
      if (getLexer().is(TypeStartToken)) {
        Lex();
        if (getParser().ParseIdentifier(TypeName))
          return TokError("expected identifier in directive");

        if (getLexer().is(AsmToken::Comma)) {
          Lex();

          if (getParser().ParseAbsoluteExpression(Size))
            return true;

          if (Size <= 0)
            return TokError("section size must be positive");
        }
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  unsigned Flags = 0;
  for (unsigned i = 0; i < FlagsStr.size(); i++) {
    switch (FlagsStr[i]) {
    case 'a':
      Flags |= MCSectionELF::SHF_ALLOC;
      break;
    case 'x':
      Flags |= MCSectionELF::SHF_EXECINSTR;
      break;
    case 'w':
      Flags |= MCSectionELF::SHF_WRITE;
      break;
    case 'M':
      Flags |= MCSectionELF::SHF_MERGE;
      break;
    case 'S':
      Flags |= MCSectionELF::SHF_STRINGS;
      break;
    case 'T':
      Flags |= MCSectionELF::SHF_TLS;
      break;
    case 'c':
      Flags |= MCSectionELF::XCORE_SHF_CP_SECTION;
      break;
    case 'd':
      Flags |= MCSectionELF::XCORE_SHF_DP_SECTION;
      break;
    default:
      return TokError("unknown flag");
    }
  }

  unsigned Type = MCSectionELF::SHT_NULL;
  if (!TypeName.empty()) {
    if (TypeName == "init_array")
      Type = MCSectionELF::SHT_INIT_ARRAY;
    else if (TypeName == "fini_array")
      Type = MCSectionELF::SHT_FINI_ARRAY;
    else if (TypeName == "preinit_array")
      Type = MCSectionELF::SHT_PREINIT_ARRAY;
    else if (TypeName == "nobits")
      Type = MCSectionELF::SHT_NOBITS;
    else if (TypeName == "progbits")
      Type = MCSectionELF::SHT_PROGBITS;
    else
      return TokError("unknown section type");
  }

  SectionKind Kind = (Flags & MCSectionELF::SHF_EXECINSTR)
                     ? SectionKind::getText()
                     : SectionKind::getDataRel();
  getStreamer().SwitchSection(getContext().getELFSection(SectionName, Type,
                                                         Flags, Kind, false));
  return false;
}

bool ELFAsmParser::ParseDirectivePrevious(StringRef DirName, SMLoc) {
  const MCSection *PreviousSection = getStreamer().getPreviousSection();
  if (PreviousSection != NULL)
    getStreamer().SwitchSection(PreviousSection);

  return false;
}

namespace llvm {

MCAsmParserExtension *createELFAsmParser() {
  return new ELFAsmParser;
}

}
