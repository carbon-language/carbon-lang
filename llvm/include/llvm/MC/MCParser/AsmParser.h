//===- AsmParser.h - Parser for Assembly Files ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class declares the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef ASMPARSER_H
#define ASMPARSER_H

#include <vector>
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/AsmCond.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
class AsmCond;
class AsmToken;
class MCContext;
class MCExpr;
class MCInst;
class MCStreamer;
class MCAsmInfo;
class SourceMgr;
class TargetAsmParser;
class Twine;

class AsmParser : public MCAsmParser {
private:
  AsmLexer Lexer;
  MCContext &Ctx;
  MCStreamer &Out;
  SourceMgr &SrcMgr;
  TargetAsmParser *TargetParser;
  
  /// This is the current buffer index we're lexing from as managed by the
  /// SourceMgr object.
  int CurBuffer;

  AsmCond TheCondState;
  std::vector<AsmCond> TheCondStack;

  // FIXME: Figure out where this should leave, the code is a copy of that which
  // is also used by TargetLoweringObjectFile.
  mutable void *SectionUniquingMap;

  /// DirectiveMap - This is a table handlers for directives.  Each handler is
  /// invoked after the directive identifier is read and is responsible for
  /// parsing and validating the rest of the directive.  The handler is passed
  /// in the directive name and the location of the directive keyword.
  StringMap<bool(AsmParser::*)(StringRef, SMLoc)> DirectiveMap;
public:
  AsmParser(SourceMgr &SM, MCContext &Ctx, MCStreamer &Out,
            const MCAsmInfo &MAI);
  ~AsmParser();

  bool Run(bool NoInitialTextSection, bool NoFinalize = false);

  
  void AddDirectiveHandler(StringRef Directive,
                           bool (AsmParser::*Handler)(StringRef, SMLoc)) {
    DirectiveMap[Directive] = Handler;
  }
public:
  TargetAsmParser &getTargetParser() const { return *TargetParser; }
  void setTargetParser(TargetAsmParser &P) { TargetParser = &P; }

  /// @name MCAsmParser Interface
  /// {

  virtual MCAsmLexer &getLexer() { return Lexer; }
  virtual MCContext &getContext() { return Ctx; }
  virtual MCStreamer &getStreamer() { return Out; }

  virtual void Warning(SMLoc L, const Twine &Meg);
  virtual bool Error(SMLoc L, const Twine &Msg);

  const AsmToken &Lex();

  bool ParseExpression(const MCExpr *&Res);
  virtual bool ParseExpression(const MCExpr *&Res, SMLoc &EndLoc);
  virtual bool ParseParenExpression(const MCExpr *&Res, SMLoc &EndLoc);
  virtual bool ParseAbsoluteExpression(int64_t &Res);

  /// }

private:
  MCSymbol *CreateSymbol(StringRef Name);

  // FIXME: See comment on SectionUniquingMap.
  const MCSection *getMachOSection(const StringRef &Segment,
                                   const StringRef &Section,
                                   unsigned TypeAndAttributes,
                                   unsigned Reserved2,
                                   SectionKind Kind) const;

  bool ParseStatement();

  bool TokError(const char *Msg);
  
  void PrintMessage(SMLoc Loc, const std::string &Msg, const char *Type) const;
    
  /// EnterIncludeFile - Enter the specified file. This returns true on failure.
  bool EnterIncludeFile(const std::string &Filename);
  
  bool ParseConditionalAssemblyDirectives(StringRef Directive,
                                          SMLoc DirectiveLoc);
  void EatToEndOfStatement();
  
  bool ParseAssignment(const StringRef &Name);

  bool ParsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseBinOpRHS(unsigned Precedence, const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseParenExpr(const MCExpr *&Res, SMLoc &EndLoc);

  /// ParseIdentifier - Parse an identifier or string (as a quoted identifier)
  /// and set \arg Res to the identifier contents.
  bool ParseIdentifier(StringRef &Res);
  
  // Directive Parsing.
  bool ParseDirectiveDarwinSection(); // Darwin specific ".section".
  bool ParseDirectiveSectionSwitch(const char *Segment, const char *Section,
                                   unsigned TAA = 0, unsigned ImplicitAlign = 0,
                                   unsigned StubSize = 0);
  bool ParseDirectiveAscii(bool ZeroTerminated); // ".ascii", ".asciiz"
  bool ParseDirectiveValue(unsigned Size); // ".byte", ".long", ...
  bool ParseDirectiveFill(); // ".fill"
  bool ParseDirectiveSpace(); // ".space"
  bool ParseDirectiveSet(); // ".set"
  bool ParseDirectiveOrg(); // ".org"
  // ".align{,32}", ".p2align{,w,l}"
  bool ParseDirectiveAlign(bool IsPow2, unsigned ValueSize);

  /// ParseDirectiveSymbolAttribute - Parse a directive like ".globl" which
  /// accepts a single symbol (which should be a label or an external).
  bool ParseDirectiveSymbolAttribute(MCSymbolAttr Attr);
  bool ParseDirectiveDarwinSymbolDesc(); // Darwin specific ".desc"
  bool ParseDirectiveDarwinLsym(); // Darwin specific ".lsym"

  bool ParseDirectiveComm(bool IsLocal); // ".comm" and ".lcomm"
  bool ParseDirectiveDarwinZerofill(); // Darwin specific ".zerofill"

  // Darwin specific ".subsections_via_symbols"
  bool ParseDirectiveDarwinSubsectionsViaSymbols();
  // Darwin specific .dump and .load
  bool ParseDirectiveDarwinDumpOrLoad(SMLoc IDLoc, bool IsDump);

  bool ParseDirectiveAbort(); // ".abort"
  bool ParseDirectiveInclude(); // ".include"

  bool ParseDirectiveIf(SMLoc DirectiveLoc); // ".if"
  bool ParseDirectiveElseIf(SMLoc DirectiveLoc); // ".elseif"
  bool ParseDirectiveElse(SMLoc DirectiveLoc); // ".else"
  bool ParseDirectiveEndIf(SMLoc DirectiveLoc); // .endif

  bool ParseDirectiveFile(StringRef, SMLoc DirectiveLoc); // ".file"
  bool ParseDirectiveLine(StringRef, SMLoc DirectiveLoc); // ".line"
  bool ParseDirectiveLoc(StringRef, SMLoc DirectiveLoc); // ".loc"

  /// ParseEscapedString - Parse the current token as a string which may include
  /// escaped characters and return the string contents.
  bool ParseEscapedString(std::string &Data);
};

} // end namespace llvm

#endif
