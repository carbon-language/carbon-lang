//===- TableGenBackend.cpp - Utilities for TableGen Backends ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides useful services for TableGen backends...
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>

using namespace llvm;

const size_t MAX_LINE_LEN = 80U;

static void printLine(raw_ostream &OS, const Twine &Prefix, char Fill,
                      StringRef Suffix) {
  size_t Pos = (size_t)OS.tell();
  assert((MAX_LINE_LEN - Prefix.str().size() - Suffix.size() > 0) &&
    "header line exceeds max limit");
  OS << Prefix;
  const size_t e = MAX_LINE_LEN - Suffix.size();
  for (size_t i = (size_t)OS.tell() - Pos; i < e; ++i)
    OS << Fill;
  OS << Suffix << '\n';
}

void llvm::emitSourceFileHeader(StringRef Desc, raw_ostream &OS) {
  printLine(OS, "/*===- TableGen'erated file ", '-', "*- C++ -*-===*\\");
  printLine(OS, "|*", ' ', "*|");
  size_t Pos = 0U;
  size_t PosE;
  StringRef Prefix("|*");
  StringRef Suffix(" *|");
  do{
    size_t PSLen = Suffix.size() + Prefix.size();
    PosE = Pos + ((MAX_LINE_LEN > (Desc.size() - PSLen)) ?
      Desc.size() :
      MAX_LINE_LEN - PSLen);
    printLine(OS, Prefix + Desc.slice(Pos, PosE), ' ', Suffix);
    Pos = PosE;
  } while(Pos < Desc.size());
  printLine(OS, Prefix, ' ', Suffix);
  printLine(OS, Prefix + " Automatically generated file, do not edit!", ' ',
    Suffix);
  printLine(OS, Prefix, ' ', Suffix);
  printLine(OS, "\\*===", '-', "===*/");
  OS << '\n';
}
