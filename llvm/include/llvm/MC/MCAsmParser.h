//===-- llvm/MC/MCAsmParser.h - Abstract Asm Parser Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMPARSER_H
#define LLVM_MC_MCASMPARSER_H

#include "llvm/System/DataTypes.h"

namespace llvm {
class MCAsmLexer;
class MCContext;
class MCExpr;
class MCStreamer;
class MCValue;
class SMLoc;
class Twine;

/// MCAsmParser - Generic assembler parser interface, for use by target specific
/// assembly parsers.
class MCAsmParser {
  MCAsmParser(const MCAsmParser &);   // DO NOT IMPLEMENT
  void operator=(const MCAsmParser &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  MCAsmParser();
 
public:
  virtual ~MCAsmParser();

  virtual MCAsmLexer &getLexer() = 0;

  virtual MCContext &getContext() = 0;

  /// getSteamer - Return the output streamer for the assembler.
  virtual MCStreamer &getStreamer() = 0;

  /// Warning - Emit a warning at the location \arg L, with the message \arg
  /// Msg.
  virtual void Warning(SMLoc L, const Twine &Msg) = 0;

  /// Warning - Emit an error at the location \arg L, with the message \arg
  /// Msg.
  ///
  /// \return The return value is always true, as an idiomatic convenience to
  /// clients.
  virtual bool Error(SMLoc L, const Twine &Msg) = 0;

  /// ParseExpression - Parse an arbitrary expression.
  ///
  /// @param Res - The value of the expression. The result is undefined
  /// on error.
  /// @result - False on success.
  virtual bool ParseExpression(const MCExpr *&Res, SMLoc &EndLoc) = 0;
  bool ParseExpression(const MCExpr *&Res);
  
  /// ParseParenExpression - Parse an arbitrary expression, assuming that an
  /// initial '(' has already been consumed.
  ///
  /// @param Res - The value of the expression. The result is undefined
  /// on error.
  /// @result - False on success.
  virtual bool ParseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) = 0;

  /// ParseAbsoluteExpression - Parse an expression which must evaluate to an
  /// absolute value.
  ///
  /// @param Res - The value of the absolute expression. The result is undefined
  /// on error.
  /// @result - False on success.
  virtual bool ParseAbsoluteExpression(int64_t &Res) = 0;
};

} // End llvm namespace

#endif
