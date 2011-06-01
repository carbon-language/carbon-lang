//===- TGParser.h - Parser for TableGen Files -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Parser for tablegen files.
//
//===----------------------------------------------------------------------===//

#ifndef TGPARSER_H
#define TGPARSER_H

#include "TGLexer.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include <map>

namespace llvm {
  class Record;
  class RecordVal;
  class RecordKeeper;
  struct RecTy;
  struct Init;
  struct MultiClass;
  struct SubClassReference;
  struct SubMultiClassReference;
  
  struct LetRecord {
    std::string Name;
    std::vector<unsigned> Bits;
    Init *Value;
    SMLoc Loc;
    LetRecord(const std::string &N, const std::vector<unsigned> &B, Init *V,
              SMLoc L)
      : Name(N), Bits(B), Value(V), Loc(L) {
    }
  };
  
class TGParser {
  TGLexer Lex;
  std::vector<std::vector<LetRecord> > LetStack;
  std::map<std::string, MultiClass*> MultiClasses;
  
  /// CurMultiClass - If we are parsing a 'multiclass' definition, this is the 
  /// current value.
  MultiClass *CurMultiClass;

  // Record tracker
  RecordKeeper &Records;
public:
  TGParser(SourceMgr &SrcMgr, RecordKeeper &records) : 
    Lex(SrcMgr), CurMultiClass(0), Records(records) {}
  
  /// ParseFile - Main entrypoint for parsing a tblgen file.  These parser
  /// routines return true on error, or false on success.
  bool ParseFile();
  
  bool Error(SMLoc L, const Twine &Msg) const {
    Lex.PrintError(L, Msg);
    return true;
  }
  bool TokError(const Twine &Msg) const {
    return Error(Lex.getLoc(), Msg);
  }
  const std::vector<std::string> &getDependencies() const {
    return Lex.getDependencies();
  }
private:  // Semantic analysis methods.
  bool AddValue(Record *TheRec, SMLoc Loc, const RecordVal &RV);
  bool SetValue(Record *TheRec, SMLoc Loc, const std::string &ValName, 
                const std::vector<unsigned> &BitList, Init *V);
  bool AddSubClass(Record *Rec, SubClassReference &SubClass);
  bool AddSubMultiClass(MultiClass *CurMC,
                        SubMultiClassReference &SubMultiClass);

private:  // Parser methods.
  bool ParseObjectList(MultiClass *MC = 0);
  bool ParseObject(MultiClass *MC);
  bool ParseClass();
  bool ParseMultiClass();
  bool ParseDefm(MultiClass *CurMultiClass);
  bool ParseDef(MultiClass *CurMultiClass);
  bool ParseTopLevelLet(MultiClass *CurMultiClass);
  std::vector<LetRecord> ParseLetList();

  bool ParseObjectBody(Record *CurRec);
  bool ParseBody(Record *CurRec);
  bool ParseBodyItem(Record *CurRec);

  bool ParseTemplateArgList(Record *CurRec);
  std::string ParseDeclaration(Record *CurRec, bool ParsingTemplateArgs);

  SubClassReference ParseSubClassReference(Record *CurRec, bool isDefm);
  SubMultiClassReference ParseSubMultiClassReference(MultiClass *CurMC);

  Init *ParseIDValue(Record *CurRec);
  Init *ParseIDValue(Record *CurRec, const std::string &Name, SMLoc NameLoc);
  Init *ParseSimpleValue(Record *CurRec, RecTy *ItemType = 0);
  Init *ParseValue(Record *CurRec, RecTy *ItemType = 0);
  std::vector<Init*> ParseValueList(Record *CurRec, Record *ArgsRec = 0, RecTy *EltTy = 0);
  std::vector<std::pair<llvm::Init*, std::string> > ParseDagArgList(Record *);
  bool ParseOptionalRangeList(std::vector<unsigned> &Ranges);
  bool ParseOptionalBitList(std::vector<unsigned> &Ranges);
  std::vector<unsigned> ParseRangeList();
  bool ParseRangePiece(std::vector<unsigned> &Ranges);
  RecTy *ParseType();
  Init *ParseOperation(Record *CurRec);
  RecTy *ParseOperatorType();
  std::string ParseObjectName();
  Record *ParseClassID();
  MultiClass *ParseMultiClassID();
  Record *ParseDefmID();
};
  
} // end namespace llvm

#endif
