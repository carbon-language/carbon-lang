//===- TableGenBackend.h - Base class for TableGen Backends -----*- C++ -*-===//
//
// The TableGenBackend class is provided as a common interface for all TableGen
// backends.  It provides useful services and an standardized interface.
//
//===----------------------------------------------------------------------===//

#ifndef TABLEGENBACKEND_H
#define TABLEGENBACKEND_H

#include <string>
#include <iosfwd>
class Record;
class RecordKeeper;

struct TableGenBackend {

  // run - All TableGen backends should implement the run method, which should
  // be the main entry point.
  virtual void run(std::ostream &OS) = 0;


public:   // Useful helper routines...
  /// EmitSourceFileHeader - Output a LLVM style file header to the specified
  /// ostream.
  void EmitSourceFileHeader(const std::string &Desc, std::ostream &OS) const;

  /// getQualifiedName - Return the name of the specified record, with a
  /// namespace qualifier if the record contains one.
  std::string getQualifiedName(Record *R) const;

  /// getTarget - Return the current instance of the Target class.
  Record *getTarget(RecordKeeper &RC) const;
};

#endif
