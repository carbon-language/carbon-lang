//===-- Core/ReplacementsYaml.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides functionality to serialize replacements for a
/// single file. It is used by the C++11 Migrator to store all the changes made
/// by a single transform to a particular file resulting from migrating a
/// translation unit of a particular main source file.
///
//===----------------------------------------------------------------------===//

#ifndef CPP11_MIGRATE_REPLACEMENTS_YAML_H
#define CPP11_MIGRATE_REPLACEMENTS_YAML_H

#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/YAMLTraits.h"
#include <vector>
#include <string>

/// \brief The top-level YAML document that contains the details for the
/// replacement.
struct MigratorDocument {
  std::vector<clang::tooling::Replacement> Replacements;
  std::string TargetFile;
  std::string MainSourceFile;
};

// FIXME: Put the YAML support for Replacement into clang::tooling. NOTE: The
// implementation below doesn't serialize the filename for Replacements.

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::tooling::Replacement)

namespace llvm {
namespace yaml {

/// \brief ScalarTraits to read/write std::string objects.
template <>
struct ScalarTraits<std::string> {
  static void output(const std::string &Val, void *, llvm::raw_ostream &Out) {
    // We need to put quotes around the string to make sure special characters
    // in the string is not treated as YAML tokens.
    std::string NormalizedVal = std::string("\"") + Val + std::string("\"");
    Out << NormalizedVal;
  }

  static StringRef input(StringRef Scalar, void *, std::string &Val) {
    Val = Scalar;
    return StringRef();
  }
};

/// \brief Specialized MappingTraits for Repleacements to be converted to/from
/// a YAML File.
template <>
struct MappingTraits<clang::tooling::Replacement> {
  /// \brief Normalize clang::tooling::Replacement to provide direct access to
  /// its members.
  struct NormalizedReplacement {
    NormalizedReplacement(const IO &)
        : FilePath(""), Offset(0), Length(0), ReplacementText("") {}

    NormalizedReplacement(const IO &, const clang::tooling::Replacement &R)
        : FilePath(R.getFilePath()), Offset(R.getOffset()),
          Length(R.getLength()), ReplacementText(R.getReplacementText()) {}

    clang::tooling::Replacement denormalize(const IO &) {
      return clang::tooling::Replacement(FilePath, Offset, Length,
                                         ReplacementText);
    }

    std::string FilePath;
    unsigned int Offset;
    unsigned int Length;
    std::string ReplacementText;
  };

  static void mapping(IO &Io, clang::tooling::Replacement &R) {
    MappingNormalization<NormalizedReplacement, clang::tooling::Replacement>
        Keys(Io, R);
    Io.mapRequired("Offset", Keys->Offset);
    Io.mapRequired("Length", Keys->Length);
    Io.mapRequired("ReplacementText", Keys->ReplacementText);
  }
};

/// \brief Specialized MappingTraits for MigratorDocument to be converted
/// to/from a YAML File.
template <>
struct MappingTraits<MigratorDocument> {
  static void mapping(IO &Io, MigratorDocument &TD) {
    Io.mapRequired("Replacements", TD.Replacements);
    Io.mapRequired("TargetFile", TD.TargetFile);
    Io.mapRequired("MainSourceFile", TD.MainSourceFile);
  }
};
} // end namespace yaml
} // end namespace llvm

#endif // CPP11_MIGRATE_REPLACEMENTS_YAML_H
