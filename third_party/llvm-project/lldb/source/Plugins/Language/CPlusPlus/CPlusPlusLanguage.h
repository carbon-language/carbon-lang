//===-- CPlusPlusLanguage.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CPLUSPLUSLANGUAGE_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CPLUSPLUSLANGUAGE_H

#include <set>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "Plugins/Language/ClangCommon/ClangHighlighter.h"
#include "lldb/Target/Language.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class CPlusPlusLanguage : public Language {
  ClangHighlighter m_highlighter;

public:
  class MethodName {
  public:
    MethodName()
        : m_full(), m_basename(), m_context(), m_arguments(), m_qualifiers() {}

    MethodName(ConstString s)
        : m_full(s), m_basename(), m_context(), m_arguments(), m_qualifiers(),
          m_parsed(false), m_parse_error(false) {}

    void Clear();

    bool IsValid() {
      if (!m_parsed)
        Parse();
      if (m_parse_error)
        return false;
      return (bool)m_full;
    }

    ConstString GetFullName() const { return m_full; }

    std::string GetScopeQualifiedName();

    llvm::StringRef GetBasename();

    llvm::StringRef GetContext();

    llvm::StringRef GetArguments();

    llvm::StringRef GetQualifiers();
    
    bool ContainsPath(llvm::StringRef path);

  protected:
    void Parse();
    bool TrySimplifiedParse();

    ConstString m_full; // Full name:
                        // "lldb::SBTarget::GetBreakpointAtIndex(unsigned int)
                        // const"
    llvm::StringRef m_basename;   // Basename:     "GetBreakpointAtIndex"
    llvm::StringRef m_context;    // Decl context: "lldb::SBTarget"
    llvm::StringRef m_arguments;  // Arguments:    "(unsigned int)"
    llvm::StringRef m_qualifiers; // Qualifiers:   "const"
    bool m_parsed = false;
    bool m_parse_error = false;
  };

  CPlusPlusLanguage() = default;

  ~CPlusPlusLanguage() override = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeC_plus_plus;
  }

  std::unique_ptr<TypeScavenger> GetTypeScavenger() override;
  lldb::TypeCategoryImplSP GetFormatters() override;

  HardcodedFormatters::HardcodedSummaryFinder GetHardcodedSummaries() override;

  HardcodedFormatters::HardcodedSyntheticFinder
  GetHardcodedSynthetics() override;

  bool IsNilReference(ValueObject &valobj) override;

  llvm::StringRef GetNilReferenceSummaryString() override { return "nullptr"; }

  bool IsSourceFile(llvm::StringRef file_path) const override;

  const Highlighter *GetHighlighter() const override { return &m_highlighter; }

  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() { return "cplusplus"; }

  bool SymbolNameFitsToLanguage(Mangled mangled) const override;
  
  bool DemangledNameContainsPath(llvm::StringRef path, 
                                 ConstString demangled) const override;

  ConstString
  GetDemangledFunctionNameWithoutArguments(Mangled mangled) const override;

  static bool IsCPPMangledName(llvm::StringRef name);

  // Extract C++ context and identifier from a string using heuristic matching
  // (as opposed to
  // CPlusPlusLanguage::MethodName which has to have a fully qualified C++ name
  // with parens and arguments.
  // If the name is a lone C identifier (e.g. C) or a qualified C identifier
  // (e.g. A::B::C) it will return true,
  // and identifier will be the identifier (C and C respectively) and the
  // context will be "" and "A::B" respectively.
  // If the name fails the heuristic matching for a qualified or unqualified
  // C/C++ identifier, then it will return false
  // and identifier and context will be unchanged.

  static bool ExtractContextAndIdentifier(const char *name,
                                          llvm::StringRef &context,
                                          llvm::StringRef &identifier);

  std::vector<ConstString>
  GenerateAlternateFunctionManglings(const ConstString mangled) const override;

  ConstString FindBestAlternateFunctionMangledName(
      const Mangled mangled, const SymbolContext &sym_ctx) const override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CPLUSPLUSLANGUAGE_H
