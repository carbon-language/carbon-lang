//===--- Frontend/PCHContainerOperations.h - PCH Containers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PCH_CONTAINER_OPERATIONS_H
#define LLVM_CLANG_PCH_CONTAINER_OPERATIONS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {
class raw_pwrite_stream;
class BitstreamReader;
}

namespace clang {

class ASTConsumer;
class CodeGenOptions;
class DiagnosticsEngine;
class HeaderSearchOptions;
class LangOptions;
class PreprocessorOptions;
class TargetOptions;

struct PCHBuffer {
  bool IsComplete;
  llvm::SmallVector<char, 0> Data;
};

/// \brief This abstract interface provides operations for creating
/// and unwrapping containers for serialized ASTs (precompiled headers
/// and clang modules).
class PCHContainerOperations {
public:
  virtual ~PCHContainerOperations();
  /// \brief Return an ASTConsumer that can be chained with a
  /// PCHGenerator that produces a wrapper file format containing a
  /// serialized AST bitstream.
  virtual std::unique_ptr<ASTConsumer> CreatePCHContainerGenerator(
      DiagnosticsEngine &Diags, const HeaderSearchOptions &HSO,
      const PreprocessorOptions &PPO, const TargetOptions &TO,
      const LangOptions &LO, const std::string &MainFileName,
      const std::string &OutputFileName, llvm::raw_pwrite_stream *OS,
      std::shared_ptr<PCHBuffer> Buffer) const = 0;

  /// \brief Initialize an llvm::BitstreamReader with the serialized AST inside
  /// the PCH container Buffer.
  virtual void ExtractPCH(llvm::MemoryBufferRef Buffer,
                          llvm::BitstreamReader &StreamFile) const = 0;
};

/// \brief Implements a raw pass-through PCH container.
class RawPCHContainerOperations : public PCHContainerOperations {
  /// \brief Return an ASTConsumer that can be chained with a
  /// PCHGenerator that writes the module to a flat file.
  std::unique_ptr<ASTConsumer> CreatePCHContainerGenerator(
      DiagnosticsEngine &Diags, const HeaderSearchOptions &HSO,
      const PreprocessorOptions &PPO, const TargetOptions &TO,
      const LangOptions &LO, const std::string &MainFileName,
      const std::string &OutputFileName, llvm::raw_pwrite_stream *OS,
      std::shared_ptr<PCHBuffer> Buffer) const override;

  /// \brief Initialize an llvm::BitstreamReader with Buffer.
  void ExtractPCH(llvm::MemoryBufferRef Buffer,
                  llvm::BitstreamReader &StreamFile) const override;
};
}

#endif
