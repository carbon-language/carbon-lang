//===--- SerializedDiagnosticPrinter.h - Serializer for diagnostics -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_SERIALIZE_DIAGNOSTIC_PRINTER_H_
#define LLVM_CLANG_FRONTEND_SERIALIZE_DIAGNOSTIC_PRINTER_H_

namespace llvm {
class raw_ostream;
}

namespace clang {
class DiagnosticConsumer;
class DiagnosticsEngine;

namespace serialized_diags {
/// \brief Returns a DiagnosticConsumer that serializes diagnostics to
///  a bitcode file.
///
/// The created DiagnosticConsumer is designed for quick and lightweight
/// transfer of of diagnostics to the enclosing build system (e.g., an IDE).
/// This allows wrapper tools for Clang to get diagnostics from Clang
/// (via libclang) without needing to parse Clang's command line output.
///
DiagnosticConsumer *create(llvm::raw_ostream *OS,
                           DiagnosticsEngine &Diags);

} // end serialized_diags namespace
} // end clang namespace

#endif
