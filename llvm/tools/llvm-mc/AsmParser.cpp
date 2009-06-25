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
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool AsmParser::Error(SMLoc L, const char *Msg) {
  Lexer.PrintMessage(L, Msg);
  return true;
}

bool AsmParser::TokError(const char *Msg) {
  Lexer.PrintMessage(Lexer.getLoc(), Msg);
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
bool AsmParser::ParseParenExpr(int64_t &Res) {
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
bool AsmParser::ParsePrimaryExpr(int64_t &Res) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token in expression");
  case asmtok::Identifier:
    // This is a label, this should be parsed as part of an expression, to
    // handle things like LFOO+4
    Res = 0; // FIXME.
    Lexer.Lex(); // Eat identifier.
    return false;
  case asmtok::IntVal:
    Res = Lexer.getCurIntVal();
    Lexer.Lex(); // Eat identifier.
    return false;
  case asmtok::LParen:
    Lexer.Lex(); // Eat the '('.
    return ParseParenExpr(Res);
  case asmtok::Tilde:
  case asmtok::Plus:
  case asmtok::Minus:
    Lexer.Lex(); // Eat the operator.
    return ParsePrimaryExpr(Res);
  }
}

/// ParseExpression - Parse an expression and return it.
/// 
///  expr ::= expr +,- expr          -> lowest.
///  expr ::= expr |,^,&,! expr      -> middle.
///  expr ::= expr *,/,%,<<,>> expr  -> highest.
///  expr ::= primaryexpr
///
bool AsmParser::ParseExpression(int64_t &Res) {
  return ParsePrimaryExpr(Res) ||
         ParseBinOpRHS(1, Res);
}

static unsigned getBinOpPrecedence(asmtok::TokKind K) {
  switch (K) {
  default: return 0;    // not a binop.
  case asmtok::Plus:
  case asmtok::Minus:
    return 1;
  case asmtok::Pipe:
  case asmtok::Caret:
  case asmtok::Amp:
  case asmtok::Exclaim:
    return 2;
  case asmtok::Star:
  case asmtok::Slash:
  case asmtok::Percent:
  case asmtok::LessLess:
  case asmtok::GreaterGreater:
    return 3;
  }
}


/// ParseBinOpRHS - Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool AsmParser::ParseBinOpRHS(unsigned Precedence, int64_t &Res) {
  while (1) {
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind());
    
    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;
    
    //asmtok::TokKind BinOp = Lexer.getKind();
    Lexer.Lex();
    
    // Eat the next primary expression.
    int64_t RHS;
    if (ParsePrimaryExpr(RHS)) return true;
    
    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind());
    if (TokPrec < NextTokPrec) {
      if (ParseBinOpRHS(Precedence+1, RHS)) return true;
    }

    // Merge LHS/RHS: fixme use the right operator etc.
    Res += RHS;
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
  case asmtok::Colon:
    // identifier ':'   -> Label.
    Lexer.Lex();
    
    // Since we saw a label, create a symbol and emit it.
    // FIXME: If the label starts with L it is an assembler temporary label.
    // Why does the client of this api need to know this?
    Out.EmitLabel(Ctx.GetOrCreateSymbol(IDVal));
    
    return ParseStatement();

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
    if (!strcmp(IDVal, ".fill"))
      return ParseDirectiveFill();
    if (!strcmp(IDVal, ".org"))
      return ParseDirectiveOrg();
    if (!strcmp(IDVal, ".space"))
      return ParseDirectiveSpace();

    Lexer.PrintMessage(IDLoc, "warning: ignoring directive for now");
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
  int64_t Value;
  if (ParseExpression(Value))
    return true;
  
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in assignment");

  // Eat the end of statement marker.
  Lexer.Lex();

  // Get the symbol for this name.
  // FIXME: Handle '.'.
  MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name);
  Out.EmitAssignment(Sym, MCValue::get(Value), IsDotSet);

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
///   ::= ( .ascii | .asciiz ) [ "string" ( , "string" )* ]
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
      int64_t Expr;
      if (ParseExpression(Expr))
        return true;

      Out.EmitValue(MCValue::get(Expr), Size);

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
  if (ParseExpression(NumBytes))
    return true;

  int64_t FillExpr = 0;
  bool HasFillExpr = false;
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    if (Lexer.isNot(asmtok::Comma))
      return TokError("unexpected token in '.space' directive");
    Lexer.Lex();
    
    if (ParseExpression(FillExpr))
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
  if (ParseExpression(NumValues))
    return true;

  if (Lexer.isNot(asmtok::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lexer.Lex();
  
  int64_t FillSize;
  if (ParseExpression(FillSize))
    return true;

  if (Lexer.isNot(asmtok::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lexer.Lex();
  
  int64_t FillExpr;
  if (ParseExpression(FillExpr))
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
  int64_t Offset;
  if (ParseExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    if (Lexer.isNot(asmtok::Comma))
      return TokError("unexpected token in '.org' directive");
    Lexer.Lex();
    
    if (ParseExpression(FillExpr))
      return true;

    if (Lexer.isNot(asmtok::EndOfStatement))
      return TokError("unexpected token in '.org' directive");
  }

  Lexer.Lex();
  
  Out.EmitValueToOffset(MCValue::get(Offset), FillExpr);

  return false;
}
