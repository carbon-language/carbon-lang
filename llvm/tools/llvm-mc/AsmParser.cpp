//===- AsmParser.cpp - Parser for Assembly Files --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#include "AsmParser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;

// Mach-O section uniquing.
//
// FIXME: Figure out where this should live, it should be shared by
// TargetLoweringObjectFile.
typedef StringMap<const MCSectionMachO*> MachOUniqueMapTy;

AsmParser::AsmParser(SourceMgr &_SM, MCContext &_Ctx, MCStreamer &_Out,
                     const MCAsmInfo &_MAI) 
  : Lexer(_SM, _MAI), Ctx(_Ctx), Out(_Out), TargetParser(0),
    SectionUniquingMap(0) {
  // Debugging directives.
  AddDirectiveHandler(".file", &AsmParser::ParseDirectiveFile);
  AddDirectiveHandler(".line", &AsmParser::ParseDirectiveLine);
  AddDirectiveHandler(".loc", &AsmParser::ParseDirectiveLoc);
}



AsmParser::~AsmParser() {
  // If we have the MachO uniquing map, free it.
  delete (MachOUniqueMapTy*)SectionUniquingMap;
}

const MCSection *AsmParser::getMachOSection(const StringRef &Segment,
                                            const StringRef &Section,
                                            unsigned TypeAndAttributes,
                                            unsigned Reserved2,
                                            SectionKind Kind) const {
  // We unique sections by their segment/section pair.  The returned section
  // may not have the same flags as the requested section, if so this should be
  // diagnosed by the client as an error.
  
  // Create the map if it doesn't already exist.
  if (SectionUniquingMap == 0)
    SectionUniquingMap = new MachOUniqueMapTy();
  MachOUniqueMapTy &Map = *(MachOUniqueMapTy*)SectionUniquingMap;
  
  // Form the name to look up.
  SmallString<64> Name;
  Name += Segment;
  Name.push_back(',');
  Name += Section;

  // Do the lookup, if we have a hit, return it.
  const MCSectionMachO *&Entry = Map[Name.str()];

  // FIXME: This should validate the type and attributes.
  if (Entry) return Entry;

  // Otherwise, return a new section.
  return Entry = MCSectionMachO::Create(Segment, Section, TypeAndAttributes,
                                        Reserved2, Kind, Ctx);
}

void AsmParser::Warning(SMLoc L, const Twine &Msg) {
  Lexer.PrintMessage(L, Msg.str(), "warning");
}

bool AsmParser::Error(SMLoc L, const Twine &Msg) {
  Lexer.PrintMessage(L, Msg.str(), "error");
  return true;
}

bool AsmParser::TokError(const char *Msg) {
  Lexer.PrintMessage(Lexer.getLoc(), Msg, "error");
  return true;
}

bool AsmParser::Run() {
  // Create the initial section.
  //
  // FIXME: Support -n.
  // FIXME: Target hook & command line option for initial section.
  Out.SwitchSection(getMachOSection("__TEXT", "__text",
                                    MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                    0, SectionKind()));


  // Prime the lexer.
  Lexer.Lex();
  
  bool HadError = false;
  
  AsmCond StartingCondState = TheCondState;

  // While we have input, parse each statement.
  while (Lexer.isNot(AsmToken::Eof)) {
    // Handle conditional assembly here before calling ParseStatement()
    if (Lexer.getKind() == AsmToken::Identifier) {
      // If we have an identifier, handle it as the key symbol.
      AsmToken ID = Lexer.getTok();
      SMLoc IDLoc = ID.getLoc();
      StringRef IDVal = ID.getString();

      if (IDVal == ".if" ||
          IDVal == ".elseif" ||
          IDVal == ".else" ||
          IDVal == ".endif") {
        if (!ParseConditionalAssemblyDirectives(IDVal, IDLoc))
          continue;
	HadError = true;
	EatToEndOfStatement();
	continue;
      }
    }
    if (TheCondState.Ignore) {
      EatToEndOfStatement();
      continue;
    }

    if (!ParseStatement()) continue;
  
    // We had an error, remember it and recover by skipping to the next line.
    HadError = true;
    EatToEndOfStatement();
  }

  if (TheCondState.TheCond != StartingCondState.TheCond ||
      TheCondState.Ignore != StartingCondState.Ignore)
    return TokError("unmatched .ifs or .elses");
  
  if (!HadError)  
    Out.Finish();

  return HadError;
}

/// ParseConditionalAssemblyDirectives - parse the conditional assembly
/// directives
bool AsmParser::ParseConditionalAssemblyDirectives(StringRef Directive,
                                                   SMLoc DirectiveLoc) {
  if (Directive == ".if")
    return ParseDirectiveIf(DirectiveLoc);
  if (Directive == ".elseif")
    return ParseDirectiveElseIf(DirectiveLoc);
  if (Directive == ".else")
    return ParseDirectiveElse(DirectiveLoc);
  if (Directive == ".endif")
    return ParseDirectiveEndIf(DirectiveLoc);
  return true;
}

/// EatToEndOfStatement - Throw away the rest of the line for testing purposes.
void AsmParser::EatToEndOfStatement() {
  while (Lexer.isNot(AsmToken::EndOfStatement) &&
         Lexer.isNot(AsmToken::Eof))
    Lexer.Lex();
  
  // Eat EOL.
  if (Lexer.is(AsmToken::EndOfStatement))
    Lexer.Lex();
}


/// ParseParenExpr - Parse a paren expression and return it.
/// NOTE: This assumes the leading '(' has already been consumed.
///
/// parenexpr ::= expr)
///
bool AsmParser::ParseParenExpr(const MCExpr *&Res) {
  if (ParseExpression(Res)) return true;
  if (Lexer.isNot(AsmToken::RParen))
    return TokError("expected ')' in parentheses expression");
  Lexer.Lex();
  return false;
}

MCSymbol *AsmParser::CreateSymbol(StringRef Name) {
  if (MCSymbol *S = Ctx.LookupSymbol(Name))
    return S;

  // If the label starts with L it is an assembler temporary label.
  if (Name.startswith("L"))
    return Ctx.CreateTemporarySymbol(Name);

  return Ctx.CreateSymbol(Name);
}

/// ParsePrimaryExpr - Parse a primary expression and return it.
///  primaryexpr ::= (parenexpr
///  primaryexpr ::= symbol
///  primaryexpr ::= number
///  primaryexpr ::= ~,+,- primaryexpr
bool AsmParser::ParsePrimaryExpr(const MCExpr *&Res) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token in expression");
  case AsmToken::Exclaim:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = MCUnaryExpr::CreateLNot(Res, getContext());
    return false;
  case AsmToken::String:
  case AsmToken::Identifier:
    // This is a label, this should be parsed as part of an expression, to
    // handle things like LFOO+4.
    Res = MCSymbolRefExpr::Create(Lexer.getTok().getIdentifier(), getContext());
    Lexer.Lex(); // Eat identifier.
    return false;
  case AsmToken::Integer:
    Res = MCConstantExpr::Create(Lexer.getTok().getIntVal(), getContext());
    Lexer.Lex(); // Eat token.
    return false;
  case AsmToken::LParen:
    Lexer.Lex(); // Eat the '('.
    return ParseParenExpr(Res);
  case AsmToken::Minus:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = MCUnaryExpr::CreateMinus(Res, getContext());
    return false;
  case AsmToken::Plus:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = MCUnaryExpr::CreatePlus(Res, getContext());
    return false;
  case AsmToken::Tilde:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = MCUnaryExpr::CreateNot(Res, getContext());
    return false;
  }
}

/// ParseExpression - Parse an expression and return it.
/// 
///  expr ::= expr +,- expr          -> lowest.
///  expr ::= expr |,^,&,! expr      -> middle.
///  expr ::= expr *,/,%,<<,>> expr  -> highest.
///  expr ::= primaryexpr
///
bool AsmParser::ParseExpression(const MCExpr *&Res) {
  Res = 0;
  return ParsePrimaryExpr(Res) ||
         ParseBinOpRHS(1, Res);
}

bool AsmParser::ParseParenExpression(const MCExpr *&Res) {
  if (ParseParenExpr(Res))
    return true;

  return false;
}

bool AsmParser::ParseAbsoluteExpression(int64_t &Res) {
  const MCExpr *Expr;
  
  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Expr))
    return true;

  if (!Expr->EvaluateAsAbsolute(Ctx, Res))
    return Error(StartLoc, "expected absolute expression");

  return false;
}

static unsigned getBinOpPrecedence(AsmToken::TokenKind K, 
                                   MCBinaryExpr::Opcode &Kind) {
  switch (K) {
  default:
    return 0;    // not a binop.

    // Lowest Precedence: &&, ||
  case AsmToken::AmpAmp:
    Kind = MCBinaryExpr::LAnd;
    return 1;
  case AsmToken::PipePipe:
    Kind = MCBinaryExpr::LOr;
    return 1;

    // Low Precedence: +, -, ==, !=, <>, <, <=, >, >=
  case AsmToken::Plus:
    Kind = MCBinaryExpr::Add;
    return 2;
  case AsmToken::Minus:
    Kind = MCBinaryExpr::Sub;
    return 2;
  case AsmToken::EqualEqual:
    Kind = MCBinaryExpr::EQ;
    return 2;
  case AsmToken::ExclaimEqual:
  case AsmToken::LessGreater:
    Kind = MCBinaryExpr::NE;
    return 2;
  case AsmToken::Less:
    Kind = MCBinaryExpr::LT;
    return 2;
  case AsmToken::LessEqual:
    Kind = MCBinaryExpr::LTE;
    return 2;
  case AsmToken::Greater:
    Kind = MCBinaryExpr::GT;
    return 2;
  case AsmToken::GreaterEqual:
    Kind = MCBinaryExpr::GTE;
    return 2;

    // Intermediate Precedence: |, &, ^
    //
    // FIXME: gas seems to support '!' as an infix operator?
  case AsmToken::Pipe:
    Kind = MCBinaryExpr::Or;
    return 3;
  case AsmToken::Caret:
    Kind = MCBinaryExpr::Xor;
    return 3;
  case AsmToken::Amp:
    Kind = MCBinaryExpr::And;
    return 3;

    // Highest Precedence: *, /, %, <<, >>
  case AsmToken::Star:
    Kind = MCBinaryExpr::Mul;
    return 4;
  case AsmToken::Slash:
    Kind = MCBinaryExpr::Div;
    return 4;
  case AsmToken::Percent:
    Kind = MCBinaryExpr::Mod;
    return 4;
  case AsmToken::LessLess:
    Kind = MCBinaryExpr::Shl;
    return 4;
  case AsmToken::GreaterGreater:
    Kind = MCBinaryExpr::Shr;
    return 4;
  }
}


/// ParseBinOpRHS - Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool AsmParser::ParseBinOpRHS(unsigned Precedence, const MCExpr *&Res) {
  while (1) {
    MCBinaryExpr::Opcode Kind = MCBinaryExpr::Add;
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind(), Kind);
    
    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;
    
    Lexer.Lex();
    
    // Eat the next primary expression.
    const MCExpr *RHS;
    if (ParsePrimaryExpr(RHS)) return true;
    
    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    MCBinaryExpr::Opcode Dummy;
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind(), Dummy);
    if (TokPrec < NextTokPrec) {
      if (ParseBinOpRHS(Precedence+1, RHS)) return true;
    }

    // Merge LHS and RHS according to operator.
    Res = MCBinaryExpr::Create(Kind, Res, RHS, getContext());
  }
}

  
  
  
/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool AsmParser::ParseStatement() {
  if (Lexer.is(AsmToken::EndOfStatement)) {
    Lexer.Lex();
    return false;
  }

  // Statements always start with an identifier.
  AsmToken ID = Lexer.getTok();
  SMLoc IDLoc = ID.getLoc();
  StringRef IDVal;
  if (ParseIdentifier(IDVal))
    return TokError("unexpected token at start of statement");

  // FIXME: Recurse on local labels?

  // See what kind of statement we have.
  switch (Lexer.getKind()) {
  case AsmToken::Colon: {
    // identifier ':'   -> Label.
    Lexer.Lex();

    // Diagnose attempt to use a variable as a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: This doesn't diagnose assignment to a symbol which has been
    // implicitly marked as external.
    MCSymbol *Sym = CreateSymbol(IDVal);
    if (!Sym->isUndefined())
      return Error(IDLoc, "invalid symbol redefinition");
    
    // Emit the label.
    Out.EmitLabel(Sym);
   
    return ParseStatement();
  }

  case AsmToken::Equal:
    // identifier '=' ... -> assignment statement
    Lexer.Lex();

    return ParseAssignment(IDVal);

  default: // Normal instruction or directive.
    break;
  }
  
  // Otherwise, we have a normal instruction or directive.  
  if (IDVal[0] == '.') {
    // FIXME: This should be driven based on a hash lookup and callback.
    if (IDVal == ".section")
      return ParseDirectiveDarwinSection();
    if (IDVal == ".text")
      // FIXME: This changes behavior based on the -static flag to the
      // assembler.
      return ParseDirectiveSectionSwitch("__TEXT", "__text",
                                     MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS);
    if (IDVal == ".const")
      return ParseDirectiveSectionSwitch("__TEXT", "__const");
    if (IDVal == ".static_const")
      return ParseDirectiveSectionSwitch("__TEXT", "__static_const");
    if (IDVal == ".cstring")
      return ParseDirectiveSectionSwitch("__TEXT","__cstring", 
                                         MCSectionMachO::S_CSTRING_LITERALS);
    if (IDVal == ".literal4")
      return ParseDirectiveSectionSwitch("__TEXT", "__literal4",
                                         MCSectionMachO::S_4BYTE_LITERALS,
                                         4);
    if (IDVal == ".literal8")
      return ParseDirectiveSectionSwitch("__TEXT", "__literal8",
                                         MCSectionMachO::S_8BYTE_LITERALS,
                                         8);
    if (IDVal == ".literal16")
      return ParseDirectiveSectionSwitch("__TEXT","__literal16",
                                         MCSectionMachO::S_16BYTE_LITERALS,
                                         16);
    if (IDVal == ".constructor")
      return ParseDirectiveSectionSwitch("__TEXT","__constructor");
    if (IDVal == ".destructor")
      return ParseDirectiveSectionSwitch("__TEXT","__destructor");
    if (IDVal == ".fvmlib_init0")
      return ParseDirectiveSectionSwitch("__TEXT","__fvmlib_init0");
    if (IDVal == ".fvmlib_init1")
      return ParseDirectiveSectionSwitch("__TEXT","__fvmlib_init1");

    // FIXME: The assembler manual claims that this has the self modify code
    // flag, at least on x86-32, but that does not appear to be correct.
    if (IDVal == ".symbol_stub")
      return ParseDirectiveSectionSwitch("__TEXT","__symbol_stub",
                                         MCSectionMachO::S_SYMBOL_STUBS |
                                       MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                          // FIXME: Different on PPC and ARM.
                                         0, 16);
    // FIXME: PowerPC only?
    if (IDVal == ".picsymbol_stub")
      return ParseDirectiveSectionSwitch("__TEXT","__picsymbol_stub",
                                         MCSectionMachO::S_SYMBOL_STUBS |
                                       MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                         0, 26);
    if (IDVal == ".data")
      return ParseDirectiveSectionSwitch("__DATA", "__data");
    if (IDVal == ".static_data")
      return ParseDirectiveSectionSwitch("__DATA", "__static_data");

    // FIXME: The section names of these two are misspelled in the assembler
    // manual.
    if (IDVal == ".non_lazy_symbol_pointer")
      return ParseDirectiveSectionSwitch("__DATA", "__nl_symbol_ptr",
                                     MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS,
                                         4);
    if (IDVal == ".lazy_symbol_pointer")
      return ParseDirectiveSectionSwitch("__DATA", "__la_symbol_ptr",
                                         MCSectionMachO::S_LAZY_SYMBOL_POINTERS,
                                         4);

    if (IDVal == ".dyld")
      return ParseDirectiveSectionSwitch("__DATA", "__dyld");
    if (IDVal == ".mod_init_func")
      return ParseDirectiveSectionSwitch("__DATA", "__mod_init_func",
                                       MCSectionMachO::S_MOD_INIT_FUNC_POINTERS,
                                         4);
    if (IDVal == ".mod_term_func")
      return ParseDirectiveSectionSwitch("__DATA", "__mod_term_func",
                                       MCSectionMachO::S_MOD_TERM_FUNC_POINTERS,
                                         4);
    if (IDVal == ".const_data")
      return ParseDirectiveSectionSwitch("__DATA", "__const");
    
    
    if (IDVal == ".objc_class")
      return ParseDirectiveSectionSwitch("__OBJC", "__class", 
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_meta_class")
      return ParseDirectiveSectionSwitch("__OBJC", "__meta_class",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_cat_cls_meth")
      return ParseDirectiveSectionSwitch("__OBJC", "__cat_cls_meth",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_cat_inst_meth")
      return ParseDirectiveSectionSwitch("__OBJC", "__cat_inst_meth",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_protocol")
      return ParseDirectiveSectionSwitch("__OBJC", "__protocol",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_string_object")
      return ParseDirectiveSectionSwitch("__OBJC", "__string_object",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_cls_meth")
      return ParseDirectiveSectionSwitch("__OBJC", "__cls_meth",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_inst_meth")
      return ParseDirectiveSectionSwitch("__OBJC", "__inst_meth",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_cls_refs")
      return ParseDirectiveSectionSwitch("__OBJC", "__cls_refs",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP |
                                         MCSectionMachO::S_LITERAL_POINTERS,
                                         4);
    if (IDVal == ".objc_message_refs")
      return ParseDirectiveSectionSwitch("__OBJC", "__message_refs",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP |
                                         MCSectionMachO::S_LITERAL_POINTERS,
                                         4);
    if (IDVal == ".objc_symbols")
      return ParseDirectiveSectionSwitch("__OBJC", "__symbols",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_category")
      return ParseDirectiveSectionSwitch("__OBJC", "__category",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_class_vars")
      return ParseDirectiveSectionSwitch("__OBJC", "__class_vars",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_instance_vars")
      return ParseDirectiveSectionSwitch("__OBJC", "__instance_vars",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_module_info")
      return ParseDirectiveSectionSwitch("__OBJC", "__module_info",
                                         MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
    if (IDVal == ".objc_class_names")
      return ParseDirectiveSectionSwitch("__TEXT", "__cstring",
                                         MCSectionMachO::S_CSTRING_LITERALS);
    if (IDVal == ".objc_meth_var_types")
      return ParseDirectiveSectionSwitch("__TEXT", "__cstring",
                                         MCSectionMachO::S_CSTRING_LITERALS);
    if (IDVal == ".objc_meth_var_names")
      return ParseDirectiveSectionSwitch("__TEXT", "__cstring",
                                         MCSectionMachO::S_CSTRING_LITERALS);
    if (IDVal == ".objc_selector_strs")
      return ParseDirectiveSectionSwitch("__OBJC", "__selector_strs",
                                         MCSectionMachO::S_CSTRING_LITERALS);
    
    // Assembler features
    if (IDVal == ".set")
      return ParseDirectiveSet();

    // Data directives

    if (IDVal == ".ascii")
      return ParseDirectiveAscii(false);
    if (IDVal == ".asciz")
      return ParseDirectiveAscii(true);

    if (IDVal == ".byte")
      return ParseDirectiveValue(1);
    if (IDVal == ".short")
      return ParseDirectiveValue(2);
    if (IDVal == ".long")
      return ParseDirectiveValue(4);
    if (IDVal == ".quad")
      return ParseDirectiveValue(8);

    // FIXME: Target hooks for IsPow2.
    if (IDVal == ".align")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    if (IDVal == ".align32")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);
    if (IDVal == ".balign")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/1);
    if (IDVal == ".balignw")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/2);
    if (IDVal == ".balignl")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/4);
    if (IDVal == ".p2align")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    if (IDVal == ".p2alignw")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/2);
    if (IDVal == ".p2alignl")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);

    if (IDVal == ".org")
      return ParseDirectiveOrg();

    if (IDVal == ".fill")
      return ParseDirectiveFill();
    if (IDVal == ".space")
      return ParseDirectiveSpace();

    // Symbol attribute directives

    if (IDVal == ".globl" || IDVal == ".global")
      return ParseDirectiveSymbolAttribute(MCStreamer::Global);
    if (IDVal == ".hidden")
      return ParseDirectiveSymbolAttribute(MCStreamer::Hidden);
    if (IDVal == ".indirect_symbol")
      return ParseDirectiveSymbolAttribute(MCStreamer::IndirectSymbol);
    if (IDVal == ".internal")
      return ParseDirectiveSymbolAttribute(MCStreamer::Internal);
    if (IDVal == ".lazy_reference")
      return ParseDirectiveSymbolAttribute(MCStreamer::LazyReference);
    if (IDVal == ".no_dead_strip")
      return ParseDirectiveSymbolAttribute(MCStreamer::NoDeadStrip);
    if (IDVal == ".private_extern")
      return ParseDirectiveSymbolAttribute(MCStreamer::PrivateExtern);
    if (IDVal == ".protected")
      return ParseDirectiveSymbolAttribute(MCStreamer::Protected);
    if (IDVal == ".reference")
      return ParseDirectiveSymbolAttribute(MCStreamer::Reference);
    if (IDVal == ".weak")
      return ParseDirectiveSymbolAttribute(MCStreamer::Weak);
    if (IDVal == ".weak_definition")
      return ParseDirectiveSymbolAttribute(MCStreamer::WeakDefinition);
    if (IDVal == ".weak_reference")
      return ParseDirectiveSymbolAttribute(MCStreamer::WeakReference);

    if (IDVal == ".comm")
      return ParseDirectiveComm(/*IsLocal=*/false);
    if (IDVal == ".lcomm")
      return ParseDirectiveComm(/*IsLocal=*/true);
    if (IDVal == ".zerofill")
      return ParseDirectiveDarwinZerofill();
    if (IDVal == ".desc")
      return ParseDirectiveDarwinSymbolDesc();
    if (IDVal == ".lsym")
      return ParseDirectiveDarwinLsym();

    if (IDVal == ".subsections_via_symbols")
      return ParseDirectiveDarwinSubsectionsViaSymbols();
    if (IDVal == ".abort")
      return ParseDirectiveAbort();
    if (IDVal == ".include")
      return ParseDirectiveInclude();
    if (IDVal == ".dump")
      return ParseDirectiveDarwinDumpOrLoad(IDLoc, /*IsDump=*/true);
    if (IDVal == ".load")
      return ParseDirectiveDarwinDumpOrLoad(IDLoc, /*IsLoad=*/false);

    // Look up the handler in the handler table, 
    bool(AsmParser::*Handler)(StringRef, SMLoc) = DirectiveMap[IDVal];
    if (Handler)
      return (this->*Handler)(IDVal, IDLoc);
    
    // Target hook for parsing target specific directives.
    if (!getTargetParser().ParseDirective(ID))
      return false;

    Warning(IDLoc, "ignoring directive for now");
    EatToEndOfStatement();
    return false;
  }

  MCInst Inst;
  if (getTargetParser().ParseInstruction(IDVal, Inst))
    return true;
  
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in argument list");

  // Eat the end of statement marker.
  Lexer.Lex();
  
  // Instruction is good, process it.
  Out.EmitInstruction(Inst);
  
  // Skip to end of line for now.
  return false;
}

bool AsmParser::ParseAssignment(const StringRef &Name) {
  // FIXME: Use better location, we should use proper tokens.
  SMLoc EqualLoc = Lexer.getLoc();

  const MCExpr *Value;
  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Value))
    return true;
  
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in assignment");

  // Eat the end of statement marker.
  Lexer.Lex();

  // Diagnose assignment to a label.
  //
  // FIXME: Diagnostics. Note the location of the definition as a label.
  // FIXME: Handle '.'.
  // FIXME: Diagnose assignment to protected identifier (e.g., register name).
  MCSymbol *Sym = CreateSymbol(Name);
  if (!Sym->isUndefined() && !Sym->isAbsolute())
    return Error(EqualLoc, "symbol has already been defined");

  // Do the assignment.
  Out.EmitAssignment(Sym, Value);

  return false;
}

/// ParseIdentifier:
///   ::= identifier
///   ::= string
bool AsmParser::ParseIdentifier(StringRef &Res) {
  if (Lexer.isNot(AsmToken::Identifier) &&
      Lexer.isNot(AsmToken::String))
    return true;

  Res = Lexer.getTok().getIdentifier();

  Lexer.Lex(); // Consume the identifier token.

  return false;
}

/// ParseDirectiveSet:
///   ::= .set identifier ',' expression
bool AsmParser::ParseDirectiveSet() {
  StringRef Name;

  if (ParseIdentifier(Name))
    return TokError("expected identifier after '.set' directive");
  
  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in '.set'");
  Lexer.Lex();

  return ParseAssignment(Name);
}

/// ParseDirectiveSection:
///   ::= .section identifier (',' identifier)*
/// FIXME: This should actually parse out the segment, section, attributes and
/// sizeof_stub fields.
bool AsmParser::ParseDirectiveDarwinSection() {
  SMLoc Loc = Lexer.getLoc();

  StringRef SectionName;
  if (ParseIdentifier(SectionName))
    return Error(Loc, "expected identifier after '.section' directive");

  // Verify there is a following comma.
  if (!Lexer.is(AsmToken::Comma))
    return TokError("unexpected token in '.section' directive");

  std::string SectionSpec = SectionName;
  SectionSpec += ",";

  // Add all the tokens until the end of the line, ParseSectionSpecifier will
  // handle this.
  StringRef EOL = Lexer.LexUntilEndOfStatement();
  SectionSpec.append(EOL.begin(), EOL.end());

  Lexer.Lex();
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.section' directive");
  Lexer.Lex();


  StringRef Segment, Section;
  unsigned TAA, StubSize;
  std::string ErrorStr = 
    MCSectionMachO::ParseSectionSpecifier(SectionSpec, Segment, Section,
                                          TAA, StubSize);
  
  if (!ErrorStr.empty())
    return Error(Loc, ErrorStr.c_str());
  
  // FIXME: Arch specific.
  Out.SwitchSection(getMachOSection(Segment, Section, TAA, StubSize,
                                    SectionKind()));
  return false;
}

/// ParseDirectiveSectionSwitch - 
bool AsmParser::ParseDirectiveSectionSwitch(const char *Segment,
                                            const char *Section,
                                            unsigned TAA, unsigned Align,
                                            unsigned StubSize) {
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lexer.Lex();
  
  // FIXME: Arch specific.
  Out.SwitchSection(getMachOSection(Segment, Section, TAA, StubSize,
                                    SectionKind()));

  // Set the implicit alignment, if any.
  //
  // FIXME: This isn't really what 'as' does; I think it just uses the implicit
  // alignment on the section (e.g., if one manually inserts bytes into the
  // section, then just issueing the section switch directive will not realign
  // the section. However, this is arguably more reasonable behavior, and there
  // is no good reason for someone to intentionally emit incorrectly sized
  // values into the implicitly aligned sections.
  if (Align)
    Out.EmitValueToAlignment(Align, 0, 1, 0);

  return false;
}

bool AsmParser::ParseEscapedString(std::string &Data) {
  assert(Lexer.is(AsmToken::String) && "Unexpected current token!");

  Data = "";
  StringRef Str = Lexer.getTok().getStringContents();
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] != '\\') {
      Data += Str[i];
      continue;
    }

    // Recognize escaped characters. Note that this escape semantics currently
    // loosely follows Darwin 'as'. Notably, it doesn't support hex escapes.
    ++i;
    if (i == e)
      return TokError("unexpected backslash at end of string");

    // Recognize octal sequences.
    if ((unsigned) (Str[i] - '0') <= 7) {
      // Consume up to three octal characters.
      unsigned Value = Str[i] - '0';

      if (i + 1 != e && ((unsigned) (Str[i + 1] - '0')) <= 7) {
        ++i;
        Value = Value * 8 + (Str[i] - '0');

        if (i + 1 != e && ((unsigned) (Str[i + 1] - '0')) <= 7) {
          ++i;
          Value = Value * 8 + (Str[i] - '0');
        }
      }

      if (Value > 255)
        return TokError("invalid octal escape sequence (out of range)");

      Data += (unsigned char) Value;
      continue;
    }

    // Otherwise recognize individual escapes.
    switch (Str[i]) {
    default:
      // Just reject invalid escape sequences for now.
      return TokError("invalid escape sequence (unrecognized character)");

    case 'b': Data += '\b'; break;
    case 'f': Data += '\f'; break;
    case 'n': Data += '\n'; break;
    case 'r': Data += '\r'; break;
    case 't': Data += '\t'; break;
    case '"': Data += '"'; break;
    case '\\': Data += '\\'; break;
    }
  }

  return false;
}

/// ParseDirectiveAscii:
///   ::= ( .ascii | .asciz ) [ "string" ( , "string" )* ]
bool AsmParser::ParseDirectiveAscii(bool ZeroTerminated) {
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      if (Lexer.isNot(AsmToken::String))
        return TokError("expected string in '.ascii' or '.asciz' directive");
      
      std::string Data;
      if (ParseEscapedString(Data))
        return true;
      
      Out.EmitBytes(Data);
      if (ZeroTerminated)
        Out.EmitBytes(StringRef("\0", 1));
      
      Lexer.Lex();
      
      if (Lexer.is(AsmToken::EndOfStatement))
        break;

      if (Lexer.isNot(AsmToken::Comma))
        return TokError("unexpected token in '.ascii' or '.asciz' directive");
      Lexer.Lex();
    }
  }

  Lexer.Lex();
  return false;
}

/// ParseDirectiveValue
///  ::= (.byte | .short | ... ) [ expression (, expression)* ]
bool AsmParser::ParseDirectiveValue(unsigned Size) {
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      SMLoc StartLoc = Lexer.getLoc();
      if (ParseExpression(Value))
        return true;

      Out.EmitValue(Value, Size);

      if (Lexer.is(AsmToken::EndOfStatement))
        break;
      
      // FIXME: Improve diagnostic.
      if (Lexer.isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lexer.Lex();
    }
  }

  Lexer.Lex();
  return false;
}

/// ParseDirectiveSpace
///  ::= .space expression [ , expression ]
bool AsmParser::ParseDirectiveSpace() {
  int64_t NumBytes;
  if (ParseAbsoluteExpression(NumBytes))
    return true;

  int64_t FillExpr = 0;
  bool HasFillExpr = false;
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::Comma))
      return TokError("unexpected token in '.space' directive");
    Lexer.Lex();
    
    if (ParseAbsoluteExpression(FillExpr))
      return true;

    HasFillExpr = true;

    if (Lexer.isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.space' directive");
  }

  Lexer.Lex();

  if (NumBytes <= 0)
    return TokError("invalid number of bytes in '.space' directive");

  // FIXME: Sometimes the fill expr is 'nop' if it isn't supplied, instead of 0.
  for (uint64_t i = 0, e = NumBytes; i != e; ++i)
    Out.EmitValue(MCConstantExpr::Create(FillExpr, getContext()), 1);

  return false;
}

/// ParseDirectiveFill
///  ::= .fill expression , expression , expression
bool AsmParser::ParseDirectiveFill() {
  int64_t NumValues;
  if (ParseAbsoluteExpression(NumValues))
    return true;

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lexer.Lex();
  
  int64_t FillSize;
  if (ParseAbsoluteExpression(FillSize))
    return true;

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lexer.Lex();
  
  int64_t FillExpr;
  if (ParseAbsoluteExpression(FillExpr))
    return true;

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.fill' directive");
  
  Lexer.Lex();

  if (FillSize != 1 && FillSize != 2 && FillSize != 4 && FillSize != 8)
    return TokError("invalid '.fill' size, expected 1, 2, 4, or 8");

  for (uint64_t i = 0, e = NumValues; i != e; ++i)
    Out.EmitValue(MCConstantExpr::Create(FillExpr, getContext()), FillSize);

  return false;
}

/// ParseDirectiveOrg
///  ::= .org expression [ , expression ]
bool AsmParser::ParseDirectiveOrg() {
  const MCExpr *Offset;
  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::Comma))
      return TokError("unexpected token in '.org' directive");
    Lexer.Lex();
    
    if (ParseAbsoluteExpression(FillExpr))
      return true;

    if (Lexer.isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.org' directive");
  }

  Lexer.Lex();

  // FIXME: Only limited forms of relocatable expressions are accepted here, it
  // has to be relative to the current section.
  Out.EmitValueToOffset(Offset, FillExpr);

  return false;
}

/// ParseDirectiveAlign
///  ::= {.align, ...} expression [ , expression [ , expression ]]
bool AsmParser::ParseDirectiveAlign(bool IsPow2, unsigned ValueSize) {
  SMLoc AlignmentLoc = Lexer.getLoc();
  int64_t Alignment;
  if (ParseAbsoluteExpression(Alignment))
    return true;

  SMLoc MaxBytesLoc;
  bool HasFillExpr = false;
  int64_t FillExpr = 0;
  int64_t MaxBytesToFill = 0;
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::Comma))
      return TokError("unexpected token in directive");
    Lexer.Lex();

    // The fill expression can be omitted while specifying a maximum number of
    // alignment bytes, e.g:
    //  .align 3,,4
    if (Lexer.isNot(AsmToken::Comma)) {
      HasFillExpr = true;
      if (ParseAbsoluteExpression(FillExpr))
        return true;
    }

    if (Lexer.isNot(AsmToken::EndOfStatement)) {
      if (Lexer.isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lexer.Lex();

      MaxBytesLoc = Lexer.getLoc();
      if (ParseAbsoluteExpression(MaxBytesToFill))
        return true;
      
      if (Lexer.isNot(AsmToken::EndOfStatement))
        return TokError("unexpected token in directive");
    }
  }

  Lexer.Lex();

  if (!HasFillExpr) {
    // FIXME: Sometimes fill with nop.
    FillExpr = 0;
  }

  // Compute alignment in bytes.
  if (IsPow2) {
    // FIXME: Diagnose overflow.
    if (Alignment >= 32) {
      Error(AlignmentLoc, "invalid alignment value");
      Alignment = 31;
    }

    Alignment = 1ULL << Alignment;
  }

  // Diagnose non-sensical max bytes to align.
  if (MaxBytesLoc.isValid()) {
    if (MaxBytesToFill < 1) {
      Error(MaxBytesLoc, "alignment directive can never be satisfied in this "
            "many bytes, ignoring maximum bytes expression");
      MaxBytesToFill = 0;
    }

    if (MaxBytesToFill >= Alignment) {
      Warning(MaxBytesLoc, "maximum bytes expression exceeds alignment and "
              "has no effect");
      MaxBytesToFill = 0;
    }
  }

  // FIXME: Target specific behavior about how the "extra" bytes are filled.
  Out.EmitValueToAlignment(Alignment, FillExpr, ValueSize, MaxBytesToFill);

  return false;
}

/// ParseDirectiveSymbolAttribute
///  ::= { ".globl", ".weak", ... } [ identifier ( , identifier )* ]
bool AsmParser::ParseDirectiveSymbolAttribute(MCStreamer::SymbolAttr Attr) {
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      StringRef Name;

      if (ParseIdentifier(Name))
        return TokError("expected identifier in directive");
      
      MCSymbol *Sym = CreateSymbol(Name);

      Out.EmitSymbolAttribute(Sym, Attr);

      if (Lexer.is(AsmToken::EndOfStatement))
        break;

      if (Lexer.isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lexer.Lex();
    }
  }

  Lexer.Lex();
  return false;  
}

/// ParseDirectiveDarwinSymbolDesc
///  ::= .desc identifier , expression
bool AsmParser::ParseDirectiveDarwinSymbolDesc() {
  StringRef Name;
  if (ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = CreateSymbol(Name);

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in '.desc' directive");
  Lexer.Lex();

  SMLoc DescLoc = Lexer.getLoc();
  int64_t DescValue;
  if (ParseAbsoluteExpression(DescValue))
    return true;

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.desc' directive");
  
  Lexer.Lex();

  // Set the n_desc field of this Symbol to this DescValue
  Out.EmitSymbolDesc(Sym, DescValue);

  return false;
}

/// ParseDirectiveComm
///  ::= ( .comm | .lcomm ) identifier , size_expression [ , align_expression ]
bool AsmParser::ParseDirectiveComm(bool IsLocal) {
  SMLoc IDLoc = Lexer.getLoc();
  StringRef Name;
  if (ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = CreateSymbol(Name);

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lexer.Lex();

  int64_t Size;
  SMLoc SizeLoc = Lexer.getLoc();
  if (ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (Lexer.is(AsmToken::Comma)) {
    Lexer.Lex();
    Pow2AlignmentLoc = Lexer.getLoc();
    if (ParseAbsoluteExpression(Pow2Alignment))
      return true;
  }
  
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.comm' or '.lcomm' directive");
  
  Lexer.Lex();

  // NOTE: a size of zero for a .comm should create a undefined symbol
  // but a size of .lcomm creates a bss symbol of size zero.
  if (Size < 0)
    return Error(SizeLoc, "invalid '.comm' or '.lcomm' directive size, can't "
                 "be less than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assember
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.comm' or '.lcomm' directive "
                 "alignment, can't be less than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // '.lcomm' is equivalent to '.zerofill'.
  // Create the Symbol as a common or local common with Size and Pow2Alignment
  if (IsLocal) {
    Out.EmitZerofill(getMachOSection("__DATA", "__bss",
                                     MCSectionMachO::S_ZEROFILL, 0,
                                     SectionKind()),
                     Sym, Size, 1 << Pow2Alignment);
    return false;
  }

  Out.EmitCommonSymbol(Sym, Size, 1 << Pow2Alignment);
  return false;
}

/// ParseDirectiveDarwinZerofill
///  ::= .zerofill segname , sectname [, identifier , size_expression [
///      , align_expression ]]
bool AsmParser::ParseDirectiveDarwinZerofill() {
  // FIXME: Handle quoted names here.

  if (Lexer.isNot(AsmToken::Identifier))
    return TokError("expected segment name after '.zerofill' directive");
  StringRef Segment = Lexer.getTok().getString();
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lexer.Lex();
 
  if (Lexer.isNot(AsmToken::Identifier))
    return TokError("expected section name after comma in '.zerofill' "
                    "directive");
  StringRef Section = Lexer.getTok().getString();
  Lexer.Lex();

  // If this is the end of the line all that was wanted was to create the
  // the section but with no symbol.
  if (Lexer.is(AsmToken::EndOfStatement)) {
    // Create the zerofill section but no symbol
    Out.EmitZerofill(getMachOSection(Segment, Section,
                                     MCSectionMachO::S_ZEROFILL, 0,
                                     SectionKind()));
    return false;
  }

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::Identifier))
    return TokError("expected identifier in directive");
  
  // handle the identifier as the key symbol.
  SMLoc IDLoc = Lexer.getLoc();
  MCSymbol *Sym = CreateSymbol(Lexer.getTok().getString());
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lexer.Lex();

  int64_t Size;
  SMLoc SizeLoc = Lexer.getLoc();
  if (ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (Lexer.is(AsmToken::Comma)) {
    Lexer.Lex();
    Pow2AlignmentLoc = Lexer.getLoc();
    if (ParseAbsoluteExpression(Pow2Alignment))
      return true;
  }
  
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.zerofill' directive");
  
  Lexer.Lex();

  if (Size < 0)
    return Error(SizeLoc, "invalid '.zerofill' directive size, can't be less "
                 "than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assember
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.zerofill' directive alignment, "
                 "can't be less than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // Create the zerofill Symbol with Size and Pow2Alignment
  //
  // FIXME: Arch specific.
  Out.EmitZerofill(getMachOSection(Segment, Section,
                                 MCSectionMachO::S_ZEROFILL, 0,
                                 SectionKind()),
                   Sym, Size, 1 << Pow2Alignment);

  return false;
}

/// ParseDirectiveDarwinSubsectionsViaSymbols
///  ::= .subsections_via_symbols
bool AsmParser::ParseDirectiveDarwinSubsectionsViaSymbols() {
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.subsections_via_symbols' directive");
  
  Lexer.Lex();

  Out.EmitAssemblerFlag(MCStreamer::SubsectionsViaSymbols);

  return false;
}

/// ParseDirectiveAbort
///  ::= .abort [ "abort_string" ]
bool AsmParser::ParseDirectiveAbort() {
  // FIXME: Use loc from directive.
  SMLoc Loc = Lexer.getLoc();

  StringRef Str = "";
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::String))
      return TokError("expected string in '.abort' directive");
    
    Str = Lexer.getTok().getString();

    Lexer.Lex();
  }

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.abort' directive");
  
  Lexer.Lex();

  // FIXME: Handle here.
  if (Str.empty())
    Error(Loc, ".abort detected. Assembly stopping.");
  else
    Error(Loc, ".abort '" + Str + "' detected. Assembly stopping.");

  return false;
}

/// ParseDirectiveLsym
///  ::= .lsym identifier , expression
bool AsmParser::ParseDirectiveDarwinLsym() {
  StringRef Name;
  if (ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = CreateSymbol(Name);

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("unexpected token in '.lsym' directive");
  Lexer.Lex();

  const MCExpr *Value;
  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Value))
    return true;

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.lsym' directive");
  
  Lexer.Lex();

  // We don't currently support this directive.
  //
  // FIXME: Diagnostic location!
  (void) Sym;
  return TokError("directive '.lsym' is unsupported");
}

/// ParseDirectiveInclude
///  ::= .include "filename"
bool AsmParser::ParseDirectiveInclude() {
  if (Lexer.isNot(AsmToken::String))
    return TokError("expected string in '.include' directive");
  
  std::string Filename = Lexer.getTok().getString();
  SMLoc IncludeLoc = Lexer.getLoc();
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.include' directive");
  
  // Strip the quotes.
  Filename = Filename.substr(1, Filename.size()-2);
  
  // Attempt to switch the lexer to the included file before consuming the end
  // of statement to avoid losing it when we switch.
  if (Lexer.EnterIncludeFile(Filename)) {
    Lexer.PrintMessage(IncludeLoc,
                       "Could not find include file '" + Filename + "'",
                       "error");
    return true;
  }

  return false;
}

/// ParseDirectiveDarwinDumpOrLoad
///  ::= ( .dump | .load ) "filename"
bool AsmParser::ParseDirectiveDarwinDumpOrLoad(SMLoc IDLoc, bool IsDump) {
  if (Lexer.isNot(AsmToken::String))
    return TokError("expected string in '.dump' or '.load' directive");
  
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.dump' or '.load' directive");
  
  Lexer.Lex();

  // FIXME: If/when .dump and .load are implemented they will be done in the
  // the assembly parser and not have any need for an MCStreamer API.
  if (IsDump)
    Warning(IDLoc, "ignoring directive .dump for now");
  else
    Warning(IDLoc, "ignoring directive .load for now");

  return false;
}

/// ParseDirectiveIf
/// ::= .if expression
bool AsmParser::ParseDirectiveIf(SMLoc DirectiveLoc) {
  // Consume the identifier that was the .if directive
  Lexer.Lex();

  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;
  if(TheCondState.Ignore) {
    EatToEndOfStatement();
  }
  else {
    int64_t ExprValue;
    if (ParseAbsoluteExpression(ExprValue))
      return true;

    if (Lexer.isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.if' directive");
    
    Lexer.Lex();

    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveElseIf
/// ::= .elseif expression
bool AsmParser::ParseDirectiveElseIf(SMLoc DirectiveLoc) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
      Error(DirectiveLoc, "Encountered a .elseif that doesn't follow a .if or "
                          " an .elseif");
  TheCondState.TheCond = AsmCond::ElseIfCond;

  // Consume the identifier that was the .elseif directive
  Lexer.Lex();

  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
      LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet) {
    TheCondState.Ignore = true;
    EatToEndOfStatement();
  }
  else {
    int64_t ExprValue;
    if (ParseAbsoluteExpression(ExprValue))
      return true;

    if (Lexer.isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.elseif' directive");
    
    Lexer.Lex();
    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveElse
/// ::= .else
bool AsmParser::ParseDirectiveElse(SMLoc DirectiveLoc) {
  // Consume the identifier that was the .else directive
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.else' directive");
  
  Lexer.Lex();

  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
      Error(DirectiveLoc, "Encountered a .else that doesn't follow a .if or an "
                          ".elseif");
  TheCondState.TheCond = AsmCond::ElseCond;
  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet)
    TheCondState.Ignore = true;
  else
    TheCondState.Ignore = false;

  return false;
}

/// ParseDirectiveEndIf
/// ::= .endif
bool AsmParser::ParseDirectiveEndIf(SMLoc DirectiveLoc) {
  // Consume the identifier that was the .endif directive
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.endif' directive");
  
  Lexer.Lex();

  if ((TheCondState.TheCond == AsmCond::NoCond) ||
      TheCondStack.empty())
    Error(DirectiveLoc, "Encountered a .endif that doesn't follow a .if or "
                        ".else");
  if (!TheCondStack.empty()) {
    TheCondState = TheCondStack.back();
    TheCondStack.pop_back();
  }

  return false;
}

/// ParseDirectiveFile
/// ::= .file [number] string
bool AsmParser::ParseDirectiveFile(StringRef, SMLoc DirectiveLoc) {
  // FIXME: I'm not sure what this is.
  int64_t FileNumber = -1;
  if (Lexer.is(AsmToken::Integer)) {
    FileNumber = Lexer.getTok().getIntVal();
    Lexer.Lex();
    
    if (FileNumber < 1)
      return TokError("file number less than one");
  }

  if (Lexer.isNot(AsmToken::String))
    return TokError("unexpected token in '.file' directive");
  
  StringRef FileName = Lexer.getTok().getString();
  Lexer.Lex();

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.file' directive");

  // FIXME: Do something with the .file.

  return false;
}

/// ParseDirectiveLine
/// ::= .line [number]
bool AsmParser::ParseDirectiveLine(StringRef, SMLoc DirectiveLoc) {
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::Integer))
      return TokError("unexpected token in '.line' directive");

    int64_t LineNumber = Lexer.getTok().getIntVal();
    (void) LineNumber;
    Lexer.Lex();

    // FIXME: Do something with the .line.
  }

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.file' directive");

  return false;
}


/// ParseDirectiveLoc
/// ::= .loc number [number [number]]
bool AsmParser::ParseDirectiveLoc(StringRef, SMLoc DirectiveLoc) {
  if (Lexer.isNot(AsmToken::Integer))
    return TokError("unexpected token in '.loc' directive");

  // FIXME: What are these fields?
  int64_t FileNumber = Lexer.getTok().getIntVal();
  (void) FileNumber;
  // FIXME: Validate file.

  Lexer.Lex();
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::Integer))
      return TokError("unexpected token in '.loc' directive");

    int64_t Param2 = Lexer.getTok().getIntVal();
    (void) Param2;
    Lexer.Lex();

    if (Lexer.isNot(AsmToken::EndOfStatement)) {
      if (Lexer.isNot(AsmToken::Integer))
        return TokError("unexpected token in '.loc' directive");

      int64_t Param3 = Lexer.getTok().getIntVal();
      (void) Param3;
      Lexer.Lex();

      // FIXME: Do something with the .loc.
    }
  }

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.file' directive");

  return false;
}

