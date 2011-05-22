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
#include "llvm/MC/MCExpr.h"
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

    // Win64 EH directives.
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveStartProc>(
                                                                   ".seh_proc");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveEndProc>(
                                                                ".seh_endproc");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveStartChained>(
                                                           ".seh_startchained");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveEndChained>(
                                                             ".seh_endchained");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveHandler>(
                                                                ".seh_handler");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveHandlerData>(
                                                            ".seh_handlerdata");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectivePushReg>(
                                                                ".seh_pushreg");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveSetFrame>(
                                                               ".seh_setframe");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveAllocStack>(
                                                             ".seh_stackalloc");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveSaveReg>(
                                                                ".seh_savereg");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveSaveXMM>(
                                                                ".seh_savexmm");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectivePushFrame>(
                                                              ".seh_pushframe");
    AddDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveEndProlog>(
                                                            ".seh_endprologue");
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

  // Win64 EH directives.
  bool ParseSEHDirectiveStartProc(StringRef, SMLoc);
  bool ParseSEHDirectiveEndProc(StringRef, SMLoc);
  bool ParseSEHDirectiveStartChained(StringRef, SMLoc);
  bool ParseSEHDirectiveEndChained(StringRef, SMLoc);
  bool ParseSEHDirectiveHandler(StringRef, SMLoc);
  bool ParseSEHDirectiveHandlerData(StringRef, SMLoc);
  bool ParseSEHDirectivePushReg(StringRef, SMLoc L);
  bool ParseSEHDirectiveSetFrame(StringRef, SMLoc L);
  bool ParseSEHDirectiveAllocStack(StringRef, SMLoc L);
  bool ParseSEHDirectiveSaveReg(StringRef, SMLoc L);
  bool ParseSEHDirectiveSaveXMM(StringRef, SMLoc L);
  bool ParseSEHDirectivePushFrame(StringRef, SMLoc L);
  bool ParseSEHDirectiveEndProlog(StringRef, SMLoc);

  bool ParseAtUnwindOrAtExcept(bool &unwind, bool &except);
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

bool COFFAsmParser::ParseSEHDirectiveStartProc(StringRef, SMLoc) {
  const MCExpr *e;
  const MCSymbolRefExpr *funcExpr;
  SMLoc startLoc = getLexer().getLoc();
  if (getParser().ParseExpression(e))
    return true;

  if (!(funcExpr = dyn_cast<MCSymbolRefExpr>(e)))
    return Error(startLoc, "expected symbol");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().EmitWin64EHStartProc(&funcExpr->getSymbol());
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveEndProc(StringRef, SMLoc) {
  Lex();
  getStreamer().EmitWin64EHEndProc();
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveStartChained(StringRef, SMLoc) {
  Lex();
  getStreamer().EmitWin64EHStartChained();
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveEndChained(StringRef, SMLoc) {
  Lex();
  getStreamer().EmitWin64EHEndChained();
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveHandler(StringRef, SMLoc) {
  const MCExpr *e;
  const MCSymbolRefExpr *funcExpr;
  SMLoc startLoc = getLexer().getLoc();
  if (getParser().ParseExpression(e))
    return true;

  if (!(funcExpr = dyn_cast<MCSymbolRefExpr>(e)))
    return Error(startLoc, "expected symbol");

  bool unwind = false, except = false;
  if (!ParseAtUnwindOrAtExcept(unwind, except))
    return true;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    if (!ParseAtUnwindOrAtExcept(unwind, except))
      return true;
  }
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().EmitWin64EHHandler(&funcExpr->getSymbol(), unwind, except);
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveHandlerData(StringRef, SMLoc) {
  Lex();
  getStreamer().EmitWin64EHHandlerData();
  return false;
}

bool COFFAsmParser::ParseSEHDirectivePushReg(StringRef, SMLoc L) {
  return Error(L, "not implemented yet");
}

bool COFFAsmParser::ParseSEHDirectiveSetFrame(StringRef, SMLoc L) {
  return Error(L, "not implemented yet");
}

bool COFFAsmParser::ParseSEHDirectiveAllocStack(StringRef, SMLoc L) {
  return Error(L, "not implemented yet");
}

bool COFFAsmParser::ParseSEHDirectiveSaveReg(StringRef, SMLoc L) {
  return Error(L, "not implemented yet");
}

bool COFFAsmParser::ParseSEHDirectiveSaveXMM(StringRef, SMLoc L) {
  return Error(L, "not implemented yet");
}

bool COFFAsmParser::ParseSEHDirectivePushFrame(StringRef, SMLoc L) {
  return Error(L, "not implemented yet");
}

bool COFFAsmParser::ParseSEHDirectiveEndProlog(StringRef, SMLoc) {
  Lex();
  getStreamer().EmitWin64EHEndProlog();
  return false;
}

bool COFFAsmParser::ParseAtUnwindOrAtExcept(bool &unwind, bool &except) {
  StringRef identifier;
  SMLoc startLoc = getLexer().getLoc();
  if (!getParser().ParseIdentifier(identifier))
    return Error(startLoc, "expected @unwind or @except");
  if (identifier == "@unwind")
    unwind = true;
  else if (identifier == "@except")
    except = true;
  else
    return Error(startLoc, "expected @unwind or @except");
  return false;
}

namespace llvm {

MCAsmParserExtension *createCOFFAsmParser() {
  return new COFFAsmParser;
}

}
