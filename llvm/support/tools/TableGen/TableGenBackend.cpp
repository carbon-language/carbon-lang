//===- TableGenBackend.cpp - Base class for TableGen Backends ---*- C++ -*-===//
//
// This file provides useful services for TableGen backends...
//
//===----------------------------------------------------------------------===//

#include "TableGenBackend.h"
#include <iostream>

void TableGenBackend::EmitSourceFileHeader(const std::string &Desc,
                                           std::ostream &OS) {
  OS << "//===- TableGen'erated file -------------------------------------*-"
       " C++ -*-===//\n//\n// " << Desc << "\n//\n// Automatically generate"
       "d file, do not edit!\n//\n//===------------------------------------"
       "----------------------------------===//\n\n";
}

