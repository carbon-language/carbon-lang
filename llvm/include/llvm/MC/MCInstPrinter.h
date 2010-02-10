//===-- MCInstPrinter.h - Convert an MCInst to target assembly syntax -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTPRINTER_H
#define LLVM_MC_MCINSTPRINTER_H

namespace llvm {
class MCInst;
class raw_ostream;
class MCAsmInfo;

  
/// MCInstPrinter - This is an instance of a target assembly language printer
/// that converts an MCInst to valid target assembly syntax.
class MCInstPrinter {
protected:
  /// O - The main stream to emit instruction text to.
  raw_ostream &O;
  
  /// CommentStream - a stream that comments can be emitted to if desired.
  /// Each comment must end with a newline.  This will be null if verbose
  /// assembly emission is disable.
  raw_ostream *CommentStream;
  const MCAsmInfo &MAI;
public:
  MCInstPrinter(raw_ostream &o, const MCAsmInfo &mai)
    : O(o), CommentStream(0), MAI(mai) {}
  
  virtual ~MCInstPrinter();

  /// setCommentStream - Specify a stream to emit comments to.
  void setCommentStream(raw_ostream &OS) { CommentStream = &OS; }
  
  /// printInst - Print the specified MCInst to the current raw_ostream.
  ///
  virtual void printInst(const MCInst *MI) = 0;
};
  
} // namespace llvm

#endif
