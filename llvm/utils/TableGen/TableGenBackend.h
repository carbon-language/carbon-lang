//===- TableGenBackend.h - Base class for TableGen Backends -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// The TableGenBackend class is provided as a common interface for all TableGen
// backends.  It provides useful services and an standardized interface.
//
//===----------------------------------------------------------------------===//

#ifndef TABLEGENBACKEND_H
#define TABLEGENBACKEND_H

#include <string>
#include <iosfwd>

namespace llvm {

class Record;
class RecordKeeper;

struct TableGenBackend {
  virtual ~TableGenBackend() {}

  // run - All TableGen backends should implement the run method, which should
  // be the main entry point.
  virtual void run(std::ostream &OS) = 0;


public:   // Useful helper routines...
  /// EmitSourceFileHeader - Output a LLVM style file header to the specified
  /// ostream.
  void EmitSourceFileHeader(const std::string &Desc, std::ostream &OS) const;

  /// EmitSourceFileTail - Output an LLVm styelf ile tail to the specified
  /// ostream.
  void EmitSourceFileTail( std::ostream& OS ) const;

  /// getQualifiedName - Return the name of the specified record, with a
  /// namespace qualifier if the record contains one.
  std::string getQualifiedName(Record *R) const;
};

} // End llvm namespace

#endif
