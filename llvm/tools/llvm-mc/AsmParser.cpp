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

#include "AsmExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

void AsmParser::Warning(SMLoc L, const char *Msg) {
  Lexer.PrintMessage(L, Msg, "warning");
}

bool AsmParser::Error(SMLoc L, const char *Msg) {
  Lexer.PrintMessage(L, Msg, "error");
  return true;
}

bool AsmParser::TokError(const char *Msg) {
  Lexer.PrintMessage(Lexer.getLoc(), Msg, "error");
  return true;
}

bool AsmParser::Run() {
  // Prime the lexer.
  Lexer.Lex();
  
  while (Lexer.isNot(asmtok::Eof))
    if (ParseStatement())
      return true;
  
  return false;
}

/// EatToEndOfStatement - Throw away the rest of the line for testing purposes.
void AsmParser::EatToEndOfStatement() {
  while (Lexer.isNot(asmtok::EndOfStatement) &&
         Lexer.isNot(asmtok::Eof))
    Lexer.Lex();
  
  // Eat EOL.
  if (Lexer.is(asmtok::EndOfStatement))
    Lexer.Lex();
}


/// ParseParenExpr - Parse a paren expression and return it.
/// NOTE: This assumes the leading '(' has already been consumed.
///
/// parenexpr ::= expr)
///
bool AsmParser::ParseParenExpr(AsmExpr *&Res) {
  if (ParseExpression(Res)) return true;
  if (Lexer.isNot(asmtok::RParen))
    return TokError("expected ')' in parentheses expression");
  Lexer.Lex();
  return false;
}

/// ParsePrimaryExpr - Parse a primary expression and return it.
///  primaryexpr ::= (parenexpr
///  primaryexpr ::= symbol
///  primaryexpr ::= number
///  primaryexpr ::= ~,+,- primaryexpr
bool AsmParser::ParsePrimaryExpr(AsmExpr *&Res) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token in expression");
  case asmtok::Exclaim:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = new AsmUnaryExpr(AsmUnaryExpr::LNot, Res);
    return false;
  case asmtok::Identifier: {
    // This is a label, this should be parsed as part of an expression, to
    // handle things like LFOO+4.
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Lexer.getCurStrVal());

    // If this is use of an undefined symbol then mark it external.
    if (!Sym->getSection() && !Ctx.GetSymbolValue(Sym))
      Sym->setExternal(true);
    
    Res = new AsmSymbolRefExpr(Sym);
    Lexer.Lex(); // Eat identifier.
    return false;
  }
  case asmtok::IntVal:
    Res = new AsmConstantExpr(Lexer.getCurIntVal());
    Lexer.Lex(); // Eat identifier.
    return false;
  case asmtok::LParen:
    Lexer.Lex(); // Eat the '('.
    return ParseParenExpr(Res);
  case asmtok::Minus:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = new AsmUnaryExpr(AsmUnaryExpr::Minus, Res);
    return false;
  case asmtok::Plus:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = new AsmUnaryExpr(AsmUnaryExpr::Plus, Res);
    return false;
  case asmtok::Tilde:
    Lexer.Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res))
      return true;
    Res = new AsmUnaryExpr(AsmUnaryExpr::Not, Res);
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
bool AsmParser::ParseExpression(AsmExpr *&Res) {
  Res = 0;
  return ParsePrimaryExpr(Res) ||
         ParseBinOpRHS(1, Res);
}

bool AsmParser::ParseAbsoluteExpression(int64_t &Res) {
  AsmExpr *Expr;
  
  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Expr))
    return true;

  if (!Expr->EvaluateAsAbsolute(Ctx, Res))
    return Error(StartLoc, "expected absolute expression");

  return false;
}

bool AsmParser::ParseRelocatableExpression(MCValue &Res) {
  AsmExpr *Expr;
  
  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Expr))
    return true;

  if (!Expr->EvaluateAsRelocatable(Ctx, Res))
    return Error(StartLoc, "expected relocatable expression");

  return false;
}

static unsigned getBinOpPrecedence(asmtok::TokKind K, 
                                   AsmBinaryExpr::Opcode &Kind) {
  switch (K) {
  default: return 0;    // not a binop.

    // Lowest Precedence: &&, ||
  case asmtok::AmpAmp:
    Kind = AsmBinaryExpr::LAnd;
    return 1;
  case asmtok::PipePipe:
    Kind = AsmBinaryExpr::LOr;
    return 1;

    // Low Precedence: +, -, ==, !=, <>, <, <=, >, >=
  case asmtok::Plus:
    Kind = AsmBinaryExpr::Add;
    return 2;
  case asmtok::Minus:
    Kind = AsmBinaryExpr::Sub;
    return 2;
  case asmtok::EqualEqual:
    Kind = AsmBinaryExpr::EQ;
    return 2;
  case asmtok::ExclaimEqual:
  case asmtok::LessGreater:
    Kind = AsmBinaryExpr::NE;
    return 2;
  case asmtok::Less:
    Kind = AsmBinaryExpr::LT;
    return 2;
  case asmtok::LessEqual:
    Kind = AsmBinaryExpr::LTE;
    return 2;
  case asmtok::Greater:
    Kind = AsmBinaryExpr::GT;
    return 2;
  case asmtok::GreaterEqual:
    Kind = AsmBinaryExpr::GTE;
    return 2;

    // Intermediate Precedence: |, &, ^
    //
    // FIXME: gas seems to support '!' as an infix operator?
  case asmtok::Pipe:
    Kind = AsmBinaryExpr::Or;
    return 3;
  case asmtok::Caret:
    Kind = AsmBinaryExpr::Xor;
    return 3;
  case asmtok::Amp:
    Kind = AsmBinaryExpr::And;
    return 3;

    // Highest Precedence: *, /, %, <<, >>
  case asmtok::Star:
    Kind = AsmBinaryExpr::Mul;
    return 4;
  case asmtok::Slash:
    Kind = AsmBinaryExpr::Div;
    return 4;
  case asmtok::Percent:
    Kind = AsmBinaryExpr::Mod;
    return 4;
  case asmtok::LessLess:
    Kind = AsmBinaryExpr::Shl;
    return 4;
  case asmtok::GreaterGreater:
    Kind = AsmBinaryExpr::Shr;
    return 4;
  }
}


/// ParseBinOpRHS - Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool AsmParser::ParseBinOpRHS(unsigned Precedence, AsmExpr *&Res) {
  while (1) {
    AsmBinaryExpr::Opcode Kind = AsmBinaryExpr::Add;
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind(), Kind);
    
    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;
    
    Lexer.Lex();
    
    // Eat the next primary expression.
    AsmExpr *RHS;
    if (ParsePrimaryExpr(RHS)) return true;
    
    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    AsmBinaryExpr::Opcode Dummy;
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind(), Dummy);
    if (TokPrec < NextTokPrec) {
      if (ParseBinOpRHS(Precedence+1, RHS)) return true;
    }

    // Merge LHS and RHS according to operator.
    Res = new AsmBinaryExpr(Kind, Res, RHS);
  }
}

  
  
  
/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool AsmParser::ParseStatement() {
  switch (Lexer.getKind()) {
  default:
    return TokError("unexpected token at start of statement");
  case asmtok::EndOfStatement:
    Lexer.Lex();
    return false;
  case asmtok::Identifier:
    break;
  // TODO: Recurse on local labels etc.
  }
  
  // If we have an identifier, handle it as the key symbol.
  SMLoc IDLoc = Lexer.getLoc();
  const char *IDVal = Lexer.getCurStrVal();
  
  // Consume the identifier, see what is after it.
  switch (Lexer.Lex()) {
  case asmtok::Colon: {
    // identifier ':'   -> Label.
    Lexer.Lex();

    // Diagnose attempt to use a variable as a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: This doesn't diagnose assignment to a symbol which has been
    // implicitly marked as external.
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(IDVal);
    if (Sym->getSection())
      return Error(IDLoc, "invalid symbol redefinition");
    if (Ctx.GetSymbolValue(Sym))
      return Error(IDLoc, "symbol already used as assembler variable");
    
    // Since we saw a label, create a symbol and emit it.
    // FIXME: If the label starts with L it is an assembler temporary label.
    // Why does the client of this api need to know this?
    Out.EmitLabel(Sym);
   
    return ParseStatement();
  }

  case asmtok::Equal:
    // identifier '=' ... -> assignment statement
    Lexer.Lex();

    return ParseAssignment(IDVal, false);

  default: // Normal instruction or directive.
    break;
  }
  
  // Otherwise, we have a normal instruction or directive.  
  if (IDVal[0] == '.') {
    // FIXME: This should be driven based on a hash lookup and callback.
    if (!strcmp(IDVal, ".section"))
      return ParseDirectiveDarwinSection();
    if (!strcmp(IDVal, ".text"))
      // FIXME: This changes behavior based on the -static flag to the
      // assembler.
      return ParseDirectiveSectionSwitch("__TEXT,__text",
                                         "regular,pure_instructions");
    if (!strcmp(IDVal, ".const"))
      return ParseDirectiveSectionSwitch("__TEXT,__const");
    if (!strcmp(IDVal, ".static_const"))
      return ParseDirectiveSectionSwitch("__TEXT,__static_const");
    if (!strcmp(IDVal, ".cstring"))
      return ParseDirectiveSectionSwitch("__TEXT,__cstring", 
                                         "cstring_literals");
    if (!strcmp(IDVal, ".literal4"))
      return ParseDirectiveSectionSwitch("__TEXT,__literal4", "4byte_literals");
    if (!strcmp(IDVal, ".literal8"))
      return ParseDirectiveSectionSwitch("__TEXT,__literal8", "8byte_literals");
    if (!strcmp(IDVal, ".literal16"))
      return ParseDirectiveSectionSwitch("__TEXT,__literal16",
                                         "16byte_literals");
    if (!strcmp(IDVal, ".constructor"))
      return ParseDirectiveSectionSwitch("__TEXT,__constructor");
    if (!strcmp(IDVal, ".destructor"))
      return ParseDirectiveSectionSwitch("__TEXT,__destructor");
    if (!strcmp(IDVal, ".fvmlib_init0"))
      return ParseDirectiveSectionSwitch("__TEXT,__fvmlib_init0");
    if (!strcmp(IDVal, ".fvmlib_init1"))
      return ParseDirectiveSectionSwitch("__TEXT,__fvmlib_init1");
    if (!strcmp(IDVal, ".symbol_stub")) // FIXME: Different on PPC.
      return ParseDirectiveSectionSwitch("__IMPORT,__jump_table,symbol_stubs",
                                    "self_modifying_code+pure_instructions,5");
    // FIXME: .picsymbol_stub on PPC.
    if (!strcmp(IDVal, ".data"))
      return ParseDirectiveSectionSwitch("__DATA,__data");
    if (!strcmp(IDVal, ".static_data"))
      return ParseDirectiveSectionSwitch("__DATA,__static_data");
    if (!strcmp(IDVal, ".non_lazy_symbol_pointer"))
      return ParseDirectiveSectionSwitch("__DATA,__nl_symbol_pointer",
                                         "non_lazy_symbol_pointers");
    if (!strcmp(IDVal, ".lazy_symbol_pointer"))
      return ParseDirectiveSectionSwitch("__DATA,__la_symbol_pointer",
                                         "lazy_symbol_pointers");
    if (!strcmp(IDVal, ".dyld"))
      return ParseDirectiveSectionSwitch("__DATA,__dyld");
    if (!strcmp(IDVal, ".mod_init_func"))
      return ParseDirectiveSectionSwitch("__DATA,__mod_init_func",
                                         "mod_init_funcs");
    if (!strcmp(IDVal, ".mod_term_func"))
      return ParseDirectiveSectionSwitch("__DATA,__mod_term_func",
                                         "mod_term_funcs");
    if (!strcmp(IDVal, ".const_data"))
      return ParseDirectiveSectionSwitch("__DATA,__const", "regular");
    
    
    // FIXME: Verify attributes on sections.
    if (!strcmp(IDVal, ".objc_class"))
      return ParseDirectiveSectionSwitch("__OBJC,__class");
    if (!strcmp(IDVal, ".objc_meta_class"))
      return ParseDirectiveSectionSwitch("__OBJC,__meta_class");
    if (!strcmp(IDVal, ".objc_cat_cls_meth"))
      return ParseDirectiveSectionSwitch("__OBJC,__cat_cls_meth");
    if (!strcmp(IDVal, ".objc_cat_inst_meth"))
      return ParseDirectiveSectionSwitch("__OBJC,__cat_inst_meth");
    if (!strcmp(IDVal, ".objc_protocol"))
      return ParseDirectiveSectionSwitch("__OBJC,__protocol");
    if (!strcmp(IDVal, ".objc_string_object"))
      return ParseDirectiveSectionSwitch("__OBJC,__string_object");
    if (!strcmp(IDVal, ".objc_cls_meth"))
      return ParseDirectiveSectionSwitch("__OBJC,__cls_meth");
    if (!strcmp(IDVal, ".objc_inst_meth"))
      return ParseDirectiveSectionSwitch("__OBJC,__inst_meth");
    if (!strcmp(IDVal, ".objc_cls_refs"))
      return ParseDirectiveSectionSwitch("__OBJC,__cls_refs");
    if (!strcmp(IDVal, ".objc_message_refs"))
      return ParseDirectiveSectionSwitch("__OBJC,__message_refs");
    if (!strcmp(IDVal, ".objc_symbols"))
      return ParseDirectiveSectionSwitch("__OBJC,__symbols");
    if (!strcmp(IDVal, ".objc_category"))
      return ParseDirectiveSectionSwitch("__OBJC,__category");
    if (!strcmp(IDVal, ".objc_class_vars"))
      return ParseDirectiveSectionSwitch("__OBJC,__class_vars");
    if (!strcmp(IDVal, ".objc_instance_vars"))
      return ParseDirectiveSectionSwitch("__OBJC,__instance_vars");
    if (!strcmp(IDVal, ".objc_module_info"))
      return ParseDirectiveSectionSwitch("__OBJC,__module_info");
    if (!strcmp(IDVal, ".objc_class_names"))
      return ParseDirectiveSectionSwitch("__TEXT,__cstring","cstring_literals");
    if (!strcmp(IDVal, ".objc_meth_var_types"))
      return ParseDirectiveSectionSwitch("__TEXT,__cstring","cstring_literals");
    if (!strcmp(IDVal, ".objc_meth_var_names"))
      return ParseDirectiveSectionSwitch("__TEXT,__cstring","cstring_literals");
    if (!strcmp(IDVal, ".objc_selector_strs"))
      return ParseDirectiveSectionSwitch("__OBJC,__selector_strs");
    
    // Assembler features
    if (!strcmp(IDVal, ".set"))
      return ParseDirectiveSet();

    // Data directives

    if (!strcmp(IDVal, ".ascii"))
      return ParseDirectiveAscii(false);
    if (!strcmp(IDVal, ".asciz"))
      return ParseDirectiveAscii(true);

    // FIXME: Target hooks for size? Also for "word", "hword".
    if (!strcmp(IDVal, ".byte"))
      return ParseDirectiveValue(1);
    if (!strcmp(IDVal, ".short"))
      return ParseDirectiveValue(2);
    if (!strcmp(IDVal, ".long"))
      return ParseDirectiveValue(4);
    if (!strcmp(IDVal, ".quad"))
      return ParseDirectiveValue(8);

    // FIXME: Target hooks for IsPow2.
    if (!strcmp(IDVal, ".align"))
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    if (!strcmp(IDVal, ".align32"))
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);
    if (!strcmp(IDVal, ".balign"))
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/1);
    if (!strcmp(IDVal, ".balignw"))
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/2);
    if (!strcmp(IDVal, ".balignl"))
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/4);
    if (!strcmp(IDVal, ".p2align"))
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    if (!strcmp(IDVal, ".p2alignw"))
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/2);
    if (!strcmp(IDVal, ".p2alignl"))
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);

    if (!strcmp(IDVal, ".org"))
      return ParseDirectiveOrg();

    if (!strcmp(IDVal, ".fill"))
      return ParseDirectiveFill();
    if (!strcmp(IDVal, ".space"))
      return ParseDirectiveSpace();

    // Symbol attribute directives
    if (!strcmp(IDVal, ".globl") || !strcmp(IDVal, ".global"))
      return ParseDirectiveSymbolAttribute(MCStreamer::Global);
    if (!strcmp(IDVal, ".hidden"))
      return ParseDirectiveSymbolAttribute(MCStreamer::Hidden);
    if (!strcmp(IDVal, ".indirect_symbol"))
      return ParseDirectiveSymbolAttribute(MCStreamer::IndirectSymbol);
    if (!strcmp(IDVal, ".internal"))
      return ParseDirectiveSymbolAttribute(MCStreamer::Internal);
    if (!strcmp(IDVal, ".lazy_reference"))
      return ParseDirectiveSymbolAttribute(MCStreamer::LazyReference);
    if (!strcmp(IDVal, ".no_dead_strip"))
      return ParseDirectiveSymbolAttribute(MCStreamer::NoDeadStrip);
    if (!strcmp(IDVal, ".private_extern"))
      return ParseDirectiveSymbolAttribute(MCStreamer::PrivateExtern);
    if (!strcmp(IDVal, ".protected"))
      return ParseDirectiveSymbolAttribute(MCStreamer::Protected);
    if (!strcmp(IDVal, ".reference"))
      return ParseDirectiveSymbolAttribute(MCStreamer::Reference);
    if (!strcmp(IDVal, ".weak"))
      return ParseDirectiveSymbolAttribute(MCStreamer::Weak);
    if (!strcmp(IDVal, ".weak_definition"))
      return ParseDirectiveSymbolAttribute(MCStreamer::WeakDefinition);
    if (!strcmp(IDVal, ".weak_reference"))
      return ParseDirectiveSymbolAttribute(MCStreamer::WeakReference);

    Warning(IDLoc, "ignoring directive for now");
    EatToEndOfStatement();
    return false;
  }

  MCInst Inst;
  if (ParseX86InstOperands(Inst))
    return true;
  
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in argument list");

  // Eat the end of statement marker.
  Lexer.Lex();
  
  // Instruction is good, process it.
  outs() << "Found instruction: " << IDVal << " with " << Inst.getNumOperands()
         << " operands.\n";
  
  // Skip to end of line for now.
  return false;
}

bool AsmParser::ParseAssignment(const char *Name, bool IsDotSet) {
  // FIXME: Use better location, we should use proper tokens.
  SMLoc EqualLoc = Lexer.getLoc();

  MCValue Value;
  if (ParseRelocatableExpression(Value))
    return true;
  
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in assignment");

  // Eat the end of statement marker.
  Lexer.Lex();

  // Diagnose assignment to a label.
  //
  // FIXME: Diagnostics. Note the location of the definition as a label.
  // FIXME: This doesn't diagnose assignment to a symbol which has been
  // implicitly marked as external.
  // FIXME: Handle '.'.
  // FIXME: Diagnose assignment to protected identifier (e.g., register name).
  MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name);
  if (Sym->getSection())
    return Error(EqualLoc, "invalid assignment to symbol emitted as a label");
  if (Sym->isExternal())
    return Error(EqualLoc, "invalid assignment to external symbol");

  // Do the assignment.
  Out.EmitAssignment(Sym, Value, IsDotSet);

  return false;
}

/// ParseDirectiveSet:
///   ::= .set identifier ',' expression
bool AsmParser::ParseDirectiveSet() {
  if (Lexer.isNot(asmtok::Identifier))
    return TokError("expected identifier after '.set' directive");

  const char *Name = Lexer.getCurStrVal();
  
  if (Lexer.Lex() != asmtok::Comma)
    return TokError("unexpected token in '.set'");
  Lexer.Lex();

  return ParseAssignment(Name, true);
}

/// ParseDirectiveSection:
///   ::= .section identifier (',' identifier)*
/// FIXME: This should actually parse out the segment, section, attributes and
/// sizeof_stub fields.
bool AsmParser::ParseDirectiveDarwinSection() {
  if (Lexer.isNot(asmtok::Identifier))
    return TokError("expected identifier after '.section' directive");
  
  std::string Section = Lexer.getCurStrVal();
  Lexer.Lex();
  
  // Accept a comma separated list of modifiers.
  while (Lexer.is(asmtok::Comma)) {
    Lexer.Lex();
    
    if (Lexer.isNot(asmtok::Identifier))
      return TokError("expected identifier in '.section' directive");
    Section += ',';
    Section += Lexer.getCurStrVal();
    Lexer.Lex();
  }
  
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in '.section' directive");
  Lexer.Lex();

  Out.SwitchSection(Ctx.GetSection(Section.c_str()));
  return false;
}

bool AsmParser::ParseDirectiveSectionSwitch(const char *Section,
                                            const char *Directives) {
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lexer.Lex();
  
  std::string SectionStr = Section;
  if (Directives && Directives[0]) {
    SectionStr += ","; 
    SectionStr += Directives;
  }
  
  Out.SwitchSection(Ctx.GetSection(Section));
  return false;
}

/// ParseDirectiveAscii:
///   ::= ( .ascii | .asciz ) [ "string" ( , "string" )* ]
bool AsmParser::ParseDirectiveAscii(bool ZeroTerminated) {
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    for (;;) {
      if (Lexer.isNot(asmtok::String))
        return TokError("expected string in '.ascii' or '.asciz' directive");
      
      // FIXME: This shouldn't use a const char* + strlen, the string could have
      // embedded nulls.
      // FIXME: Should have accessor for getting string contents.
      const char *Str = Lexer.getCurStrVal();
      Out.EmitBytes(Str + 1, strlen(Str) - 2);
      if (ZeroTerminated)
        Out.EmitBytes("\0", 1);
      
      Lexer.Lex();
      
      if (Lexer.is(asmtok::EndOfStatement))
        break;

      if (Lexer.isNot(asmtok::Comma))
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
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    for (;;) {
      MCValue Expr;
      if (ParseRelocatableExpression(Expr))
        return true;

      Out.EmitValue(Expr, Size);

      if (Lexer.is(asmtok::EndOfStatement))
        break;
      
      // FIXME: Improve diagnostic.
      if (Lexer.isNot(asmtok::Comma))
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
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    if (Lexer.isNot(asmtok::Comma))
      return TokError("unexpected token in '.space' directive");
    Lexer.Lex();
    
    if (ParseAbsoluteExpression(FillExpr))
      return true;

    HasFillExpr = true;

    if (Lexer.isNot(asmtok::EndOfStatement))
      return TokError("unexpected token in '.space' directive");
  }

  Lexer.Lex();

  if (NumBytes <= 0)
    return TokError("invalid number of bytes in '.space' directive");

  // FIXME: Sometimes the fill expr is 'nop' if it isn't supplied, instead of 0.
  for (uint64_t i = 0, e = NumBytes; i != e; ++i)
    Out.EmitValue(MCValue::get(FillExpr), 1);

  return false;
}

/// ParseDirectiveFill
///  ::= .fill expression , expression , expression
bool AsmParser::ParseDirectiveFill() {
  int64_t NumValues;
  if (ParseAbsoluteExpression(NumValues))
    return true;

  if (Lexer.isNot(asmtok::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lexer.Lex();
  
  int64_t FillSize;
  if (ParseAbsoluteExpression(FillSize))
    return true;

  if (Lexer.isNot(asmtok::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lexer.Lex();
  
  int64_t FillExpr;
  if (ParseAbsoluteExpression(FillExpr))
    return true;

  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in '.fill' directive");
  
  Lexer.Lex();

  if (FillSize != 1 && FillSize != 2 && FillSize != 4)
    return TokError("invalid '.fill' size, expected 1, 2, or 4");

  for (uint64_t i = 0, e = NumValues; i != e; ++i)
    Out.EmitValue(MCValue::get(FillExpr), FillSize);

  return false;
}

/// ParseDirectiveOrg
///  ::= .org expression [ , expression ]
bool AsmParser::ParseDirectiveOrg() {
  MCValue Offset;
  if (ParseRelocatableExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    if (Lexer.isNot(asmtok::Comma))
      return TokError("unexpected token in '.org' directive");
    Lexer.Lex();
    
    if (ParseAbsoluteExpression(FillExpr))
      return true;

    if (Lexer.isNot(asmtok::EndOfStatement))
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
  int64_t Alignment;
  if (ParseAbsoluteExpression(Alignment))
    return true;

  SMLoc MaxBytesLoc;
  bool HasFillExpr = false;
  int64_t FillExpr = 0;
  int64_t MaxBytesToFill = 0;
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    if (Lexer.isNot(asmtok::Comma))
      return TokError("unexpected token in directive");
    Lexer.Lex();

    // The fill expression can be omitted while specifying a maximum number of
    // alignment bytes, e.g:
    //  .align 3,,4
    if (Lexer.isNot(asmtok::Comma)) {
      HasFillExpr = true;
      if (ParseAbsoluteExpression(FillExpr))
        return true;
    }

    if (Lexer.isNot(asmtok::EndOfStatement)) {
      if (Lexer.isNot(asmtok::Comma))
        return TokError("unexpected token in directive");
      Lexer.Lex();

      MaxBytesLoc = Lexer.getLoc();
      if (ParseAbsoluteExpression(MaxBytesToFill))
        return true;
      
      if (Lexer.isNot(asmtok::EndOfStatement))
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
    Alignment = 1 << Alignment;
  }

  // Diagnose non-sensical max bytes to fill.
  if (MaxBytesLoc.isValid()) {
    if (MaxBytesToFill < 1) {
      Warning(MaxBytesLoc, "alignment directive can never be satisfied in this "
              "many bytes, ignoring");
      return false;
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
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    for (;;) {
      if (Lexer.isNot(asmtok::Identifier))
        return TokError("expected identifier in directive");
      
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(Lexer.getCurStrVal());
      Lexer.Lex();

      // If this is use of an undefined symbol then mark it external.
      if (!Sym->getSection() && !Ctx.GetSymbolValue(Sym))
        Sym->setExternal(true);

      Out.EmitSymbolAttribute(Sym, Attr);

      if (Lexer.is(asmtok::EndOfStatement))
        break;

      if (Lexer.isNot(asmtok::Comma))
        return TokError("unexpected token in directive");
      Lexer.Lex();
    }
  }

  Lexer.Lex();
  return false;  
}
