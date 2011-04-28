//===--- RAIIObjectsForParser.h - RAII helpers for the parser ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines and implements the some simple RAII objects that are used
// by the parser to manage bits in recursion.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_RAII_OBJECTS_FOR_PARSER_H
#define LLVM_CLANG_PARSE_RAII_OBJECTS_FOR_PARSER_H

#include "clang/Parse/ParseDiagnostic.h"

namespace clang {
  // TODO: move ParsingDeclRAIIObject here.
  // TODO: move ParsingClassDefinition here.
  // TODO: move TentativeParsingAction here.
  
  
  /// ExtensionRAIIObject - This saves the state of extension warnings when
  /// constructed and disables them.  When destructed, it restores them back to
  /// the way they used to be.  This is used to handle __extension__ in the
  /// parser.
  class ExtensionRAIIObject {
    void operator=(const ExtensionRAIIObject &);     // DO NOT IMPLEMENT
    ExtensionRAIIObject(const ExtensionRAIIObject&); // DO NOT IMPLEMENT
    Diagnostic &Diags;
  public:
    ExtensionRAIIObject(Diagnostic &diags) : Diags(diags) {
      Diags.IncrementAllExtensionsSilenced();
    }

    ~ExtensionRAIIObject() {
      Diags.DecrementAllExtensionsSilenced();
    }
  };
  
  /// ColonProtectionRAIIObject - This sets the Parser::ColonIsSacred bool and
  /// restores it when destroyed.  This says that "foo:" should not be
  /// considered a possible typo for "foo::" for error recovery purposes.
  class ColonProtectionRAIIObject {
    Parser &P;
    bool OldVal;
  public:
    ColonProtectionRAIIObject(Parser &p, bool Value = true)
      : P(p), OldVal(P.ColonIsSacred) {
      P.ColonIsSacred = Value;
    }
    
    /// restore - This can be used to restore the state early, before the dtor
    /// is run.
    void restore() {
      P.ColonIsSacred = OldVal;
    }
    
    ~ColonProtectionRAIIObject() {
      restore();
    }
  };
  
  /// \brief RAII object that makes '>' behave either as an operator
  /// or as the closing angle bracket for a template argument list.
  class GreaterThanIsOperatorScope {
    bool &GreaterThanIsOperator;
    bool OldGreaterThanIsOperator;
  public:
    GreaterThanIsOperatorScope(bool &GTIO, bool Val)
    : GreaterThanIsOperator(GTIO), OldGreaterThanIsOperator(GTIO) {
      GreaterThanIsOperator = Val;
    }
    
    ~GreaterThanIsOperatorScope() {
      GreaterThanIsOperator = OldGreaterThanIsOperator;
    }
  };
  
  class InMessageExpressionRAIIObject {
    bool &InMessageExpression;
    bool OldValue;
    
  public:
    InMessageExpressionRAIIObject(Parser &P, bool Value)
      : InMessageExpression(P.InMessageExpression), 
        OldValue(P.InMessageExpression) {
      InMessageExpression = Value;
    }
    
    ~InMessageExpressionRAIIObject() {
      InMessageExpression = OldValue;
    }
  };
  
  /// \brief RAII object that makes sure paren/bracket/brace count is correct
  /// after declaration/statement parsing, even when there's a parsing error.
  class ParenBraceBracketBalancer {
    Parser &P;
    unsigned short ParenCount, BracketCount, BraceCount;
  public:
    ParenBraceBracketBalancer(Parser &p)
      : P(p), ParenCount(p.ParenCount), BracketCount(p.BracketCount),
        BraceCount(p.BraceCount) { }
    
    ~ParenBraceBracketBalancer() {
      P.ParenCount = ParenCount;
      P.BracketCount = BracketCount;
      P.BraceCount = BraceCount;
    }
  };

  class PoisonSEHIdentifiersRAIIObject {
    PoisonIdentifierRAIIObject Ident_AbnormalTermination;
    PoisonIdentifierRAIIObject Ident_GetExceptionCode;
    PoisonIdentifierRAIIObject Ident_GetExceptionInfo;
    PoisonIdentifierRAIIObject Ident__abnormal_termination;
    PoisonIdentifierRAIIObject Ident__exception_code;
    PoisonIdentifierRAIIObject Ident__exception_info;
    PoisonIdentifierRAIIObject Ident___abnormal_termination;
    PoisonIdentifierRAIIObject Ident___exception_code;
    PoisonIdentifierRAIIObject Ident___exception_info;
  public:
    PoisonSEHIdentifiersRAIIObject(Parser &Self, bool NewValue)
      : Ident_AbnormalTermination(Self.Ident_AbnormalTermination, NewValue),
        Ident_GetExceptionCode(Self.Ident_GetExceptionCode, NewValue),
        Ident_GetExceptionInfo(Self.Ident_GetExceptionInfo, NewValue),
        Ident__abnormal_termination(Self.Ident__abnormal_termination, NewValue),
        Ident__exception_code(Self.Ident__exception_code, NewValue),
        Ident__exception_info(Self.Ident__exception_info, NewValue),
        Ident___abnormal_termination(Self.Ident___abnormal_termination, NewValue),
        Ident___exception_code(Self.Ident___exception_code, NewValue),
        Ident___exception_info(Self.Ident___exception_info, NewValue) {
    }
  };

} // end namespace clang

#endif
