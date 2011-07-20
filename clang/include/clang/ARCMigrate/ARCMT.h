//===-- ARCMT.h - ARC Migration Rewriter ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ARCMIGRATE_ARCMT_H
#define LLVM_CLANG_ARCMIGRATE_ARCMT_H

#include "clang/ARCMigrate/FileRemapper.h"
#include "clang/Frontend/CompilerInvocation.h"

namespace clang {
  class ASTContext;
  class DiagnosticClient;

namespace arcmt {
  class MigrationPass;

/// \brief Creates an AST with the provided CompilerInvocation but with these
/// changes:
///   -if a PCH/PTH is set, the original header is used instead
///   -Automatic Reference Counting mode is enabled
///
/// It then checks the AST and produces errors/warning for ARC migration issues
/// that the user needs to handle manually.
///
/// \param emitPremigrationARCErrors if true all ARC errors will get emitted
/// even if the migrator can fix them, but the function will still return false
/// if all ARC errors can be fixed.
///
/// \param plistOut if non-empty, it is the file path to store the plist with
/// the pre-migration ARC diagnostics.
///
/// \returns false if no error is produced, true otherwise.
bool checkForManualIssues(CompilerInvocation &CI,
                          StringRef Filename, InputKind Kind,
                          DiagnosticClient *DiagClient,
                          bool emitPremigrationARCErrors = false,
                          StringRef plistOut = StringRef());

/// \brief Works similar to checkForManualIssues but instead of checking, it
/// applies automatic modifications to source files to conform to ARC.
///
/// \returns false if no error is produced, true otherwise.
bool applyTransformations(CompilerInvocation &origCI,
                          StringRef Filename, InputKind Kind,
                          DiagnosticClient *DiagClient);

/// \brief Applies automatic modifications and produces temporary files
/// and metadata into the \arg outputDir path.
///
/// \param emitPremigrationARCErrors if true all ARC errors will get emitted
/// even if the migrator can fix them, but the function will still return false
/// if all ARC errors can be fixed.
///
/// \param plistOut if non-empty, it is the file path to store the plist with
/// the pre-migration ARC diagnostics.
///
/// \returns false if no error is produced, true otherwise.
bool migrateWithTemporaryFiles(CompilerInvocation &origCI,
                               StringRef Filename, InputKind Kind,
                               DiagnosticClient *DiagClient,
                               StringRef outputDir,
                               bool emitPremigrationARCErrors,
                               StringRef plistOut);

/// \brief Get the set of file remappings from the \arg outputDir path that
/// migrateWithTemporaryFiles produced.
///
/// \returns false if no error is produced, true otherwise.
bool getFileRemappings(std::vector<std::pair<std::string,std::string> > &remap,
                       StringRef outputDir,
                       DiagnosticClient *DiagClient);

typedef void (*TransformFn)(MigrationPass &pass);

std::vector<TransformFn> getAllTransformations();

class MigrationProcess {
  CompilerInvocation OrigCI;
  DiagnosticClient *DiagClient;
  FileRemapper Remapper;

public:
  MigrationProcess(const CompilerInvocation &CI, DiagnosticClient *diagClient,
                   StringRef outputDir = StringRef());

  class RewriteListener {
  public:
    virtual ~RewriteListener();

    virtual void start(ASTContext &Ctx) { }
    virtual void finish() { }

    virtual void insert(SourceLocation loc, StringRef text) { }
    virtual void remove(CharSourceRange range) { }
  };

  bool applyTransform(TransformFn trans, RewriteListener *listener = 0);

  FileRemapper &getRemapper() { return Remapper; }
};

} // end namespace arcmt

}  // end namespace clang

#endif
