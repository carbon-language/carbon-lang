//===- llvm/Support/DiagnosticPrinter.h - Diagnostic Printer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the main interface for printer backend diagnostic.
//
// Clients of the backend diagnostics should overload this interface based
// on their needs.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DIAGNOSTICPRINTER_H
#define LLVM_SUPPORT_DIAGNOSTICPRINTER_H

#include <string>

namespace llvm {
// Forward declarations.
class Module;
class raw_ostream;
class StringRef;
class Twine;
class Value;

/// \brief Interface for custom diagnostic printing.
class DiagnosticPrinter {
public:
  virtual ~DiagnosticPrinter() {}

  // Simple types.
  virtual DiagnosticPrinter &operator<<(char C) = 0;
  virtual DiagnosticPrinter &operator<<(unsigned char C) = 0;
  virtual DiagnosticPrinter &operator<<(signed char C) = 0;
  virtual DiagnosticPrinter &operator<<(StringRef Str) = 0;
  virtual DiagnosticPrinter &operator<<(const char *Str) = 0;
  virtual DiagnosticPrinter &operator<<(const std::string &Str) = 0;
  virtual DiagnosticPrinter &operator<<(unsigned long N) = 0;
  virtual DiagnosticPrinter &operator<<(long N) = 0;
  virtual DiagnosticPrinter &operator<<(unsigned long long N) = 0;
  virtual DiagnosticPrinter &operator<<(long long N) = 0;
  virtual DiagnosticPrinter &operator<<(const void *P) = 0;
  virtual DiagnosticPrinter &operator<<(unsigned int N) = 0;
  virtual DiagnosticPrinter &operator<<(int N) = 0;
  virtual DiagnosticPrinter &operator<<(double N) = 0;
  virtual DiagnosticPrinter &operator<<(const Twine &Str) = 0;

  // IR related types.
  virtual DiagnosticPrinter &operator<<(const Value &V) = 0;
  virtual DiagnosticPrinter &operator<<(const Module &M) = 0;
};

/// \brief Basic diagnostic printer that uses an underlying raw_ostream.
class DiagnosticPrinterRawOStream : public DiagnosticPrinter {
protected:
  raw_ostream &Stream;

public:
  DiagnosticPrinterRawOStream(raw_ostream &Stream) : Stream(Stream) {};

  // Simple types.
  virtual DiagnosticPrinter &operator<<(char C);
  virtual DiagnosticPrinter &operator<<(unsigned char C);
  virtual DiagnosticPrinter &operator<<(signed char C);
  virtual DiagnosticPrinter &operator<<(StringRef Str);
  virtual DiagnosticPrinter &operator<<(const char *Str);
  virtual DiagnosticPrinter &operator<<(const std::string &Str);
  virtual DiagnosticPrinter &operator<<(unsigned long N);
  virtual DiagnosticPrinter &operator<<(long N);
  virtual DiagnosticPrinter &operator<<(unsigned long long N);
  virtual DiagnosticPrinter &operator<<(long long N);
  virtual DiagnosticPrinter &operator<<(const void *P);
  virtual DiagnosticPrinter &operator<<(unsigned int N);
  virtual DiagnosticPrinter &operator<<(int N);
  virtual DiagnosticPrinter &operator<<(double N);
  virtual DiagnosticPrinter &operator<<(const Twine &Str);

  // IR related types.
  virtual DiagnosticPrinter &operator<<(const Value &V);
  virtual DiagnosticPrinter &operator<<(const Module &M);
};
} // End namespace llvm

#endif
