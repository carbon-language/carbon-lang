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

struct TableGenBackend {

  // run - All TableGen backends should implement the run method, which should
  // be the main entry point.
  virtual void run(std::ostream &OS) = 0;


public:   // Useful helper routines...
  void EmitSourceFileHeader(const std::string &Desc, std::ostream &OS) const;

};

#endif
