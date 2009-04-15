//===--- Warnings.cpp - C-Language Front-end ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Command line warning options handler.
//
//===----------------------------------------------------------------------===//
//
// This file is responsible for handling all warning options. This includes
// a number of -Wfoo options and their variants, which are driven by TableGen-
// generated data, and the special cases -pedantic, -pedantic-errors, -w and
// -Werror.
//
// Each warning option controls any number of actual warnings.
// Given a warning option 'foo', the following are valid:
//
// -Wfoo         -> alias of -Wfoo=warn
// -Wno-foo      -> alias of -Wfoo=ignore
// -Werror=foo   -> alias of -Wfoo=error
//
#include "clang-cc.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Lex/LexDiagnostic.h"
#include "llvm/Support/CommandLine.h"
#include <cstdio>
#include <utility>
#include <algorithm>
using namespace clang;

// This gets all -W options, including -Werror, -W[no-]system-headers, etc.  The
// driver has stripped off -Wa,foo etc.  The driver has also translated -W to
// -Wextra, so we don't need to worry about it.
static llvm::cl::list<std::string>
OptWarnings("W", llvm::cl::Prefix);

static llvm::cl::opt<bool> OptPedantic("pedantic");
static llvm::cl::opt<bool> OptPedanticErrors("pedantic-errors");
static llvm::cl::opt<bool> OptNoWarnings("w");

namespace {
  struct WarningOption {
    const char *Name;
    const diag::kind *Members;
    unsigned NumMembers;
  };
}
#define DIAGS(a) a, unsigned(sizeof(a) / sizeof(a[0]))
// These tables will be TableGenerated later.
// First the table sets describing the diagnostics controlled by each option.
static const diag::kind UnusedMacrosDiags[] = { diag::pp_macro_not_used };
static const diag::kind FloatEqualDiags[] = { diag::warn_floatingpoint_eq };
static const diag::kind ExtraTokens[] = { diag::ext_pp_extra_tokens_at_eol };
static const diag::kind ReadOnlySetterAttrsDiags[] = {
  diag::warn_objc_property_attr_mutually_exclusive
};
static const diag::kind FormatNonLiteralDiags[] = {
  diag::warn_printf_not_string_constant
};
static const diag::kind UndefDiags[] = { diag::warn_pp_undef_identifier };
static const diag::kind ImplicitFunctionDeclarationDiags[] = {
  diag::ext_implicit_function_decl, diag::warn_implicit_function_decl
};
static const diag::kind PointerSignDiags[] = {
  diag::ext_typecheck_convert_incompatible_pointer_sign
};
static const diag::kind DeprecatedDeclarations[] = { diag::warn_deprecated };
static const diag::kind MissingPrototypesDiags[] = { 
  diag::warn_missing_prototype 
};
static const diag::kind TrigraphsDiags[] = {
  diag::trigraph_ignored, diag::trigraph_ignored_block_comment,
  diag::trigraph_ends_block_comment, diag::trigraph_converted
};

// Second the table of options.  MUST be sorted by name! Binary lookup is done.
static const WarningOption OptionTable[] = {
  { "deprecated-declarations",       DIAGS(DeprecatedDeclarations) },
  { "extra-tokens",                  DIAGS(ExtraTokens) },
  { "float-equal",           DIAGS(FloatEqualDiags) },
  { "format-nonliteral",     DIAGS(FormatNonLiteralDiags) },
  { "implicit-function-declaration", DIAGS(ImplicitFunctionDeclarationDiags) },
  { "missing-prototypes", DIAGS(MissingPrototypesDiags) },
  { "pointer-sign",          DIAGS(PointerSignDiags) },
  { "readonly-setter-attrs", DIAGS(ReadOnlySetterAttrsDiags) },
  { "trigraphs",             DIAGS(TrigraphsDiags) },
  { "undef",                 DIAGS(UndefDiags) },
  { "unused-macros",         DIAGS(UnusedMacrosDiags) },
};
static const size_t OptionTableSize =
  sizeof(OptionTable) / sizeof(OptionTable[0]);

static bool WarningOptionCompare(const WarningOption &LHS,
                                 const WarningOption &RHS) {
  return strcmp(LHS.Name, RHS.Name) < 0;
}

bool clang::ProcessWarningOptions(Diagnostic &Diags) {
  Diags.setSuppressSystemWarnings(true);  // Default to -Wno-system-headers
  
  // FIXME: These should be mapped to group options.
  Diags.setIgnoreAllWarnings(OptNoWarnings);

  // Set some defaults that are currently set manually. This, too, should
  // be in the tablegen stuff later.
  Diags.setDiagnosticMapping(diag::pp_macro_not_used, diag::MAP_IGNORE);
  Diags.setDiagnosticMapping(diag::warn_floatingpoint_eq, diag::MAP_IGNORE);
  Diags.setDiagnosticMapping(diag::warn_objc_property_attr_mutually_exclusive,
                             diag::MAP_IGNORE);
  Diags.setDiagnosticMapping(diag::warn_pp_undef_identifier, diag::MAP_IGNORE);
  Diags.setDiagnosticMapping(diag::warn_implicit_function_decl,
                             diag::MAP_IGNORE);

  Diags.setDiagnosticMapping(diag::err_pp_file_not_found, diag::MAP_FATAL);
  Diags.setDiagnosticMapping(diag::err_template_recursion_depth_exceeded, 
                             diag::MAP_FATAL);
  Diags.setDiagnosticMapping(diag::warn_missing_prototype, diag::MAP_IGNORE);
  
  // FIXME: -fdiagnostics-show-option
  // FIXME: -Wfatal-errors / -Wfatal-errors=foo

  /// ControlledDiags - Keep track of the options that the user explicitly
  /// poked with -Wfoo, -Wno-foo, or -Werror=foo.
  llvm::SmallVector<unsigned short, 256> ControlledDiags;
  
  for (unsigned i = 0, e = OptWarnings.size(); i != e; ++i) {
    const std::string &Opt = OptWarnings[i];
    const char *OptStart = &Opt[0];
    const char *OptEnd = OptStart+Opt.size();
    assert(*OptEnd == 0 && "Expect null termination for lower-bound search");
    
    // Check to see if this warning starts with "no-", if so, this is a negative
    // form of the option.
    bool isPositive = true;
    if (OptEnd-OptStart > 3 && memcmp(OptStart, "no-", 3) == 0) {
      isPositive = false;
      OptStart += 3;
    }

    // Figure out how this option affects the warning.  If -Wfoo, map the
    // diagnostic to a warning, if -Wno-foo, map it to ignore.
    diag::Mapping Mapping = isPositive ? diag::MAP_WARNING : diag::MAP_IGNORE;

    // -Wsystem-headers is a special case, not driven by the option table.  It
    // cannot be controlled with -Werror.
    if (OptEnd-OptStart == 14 && memcmp(OptStart, "system-headers", 14) == 0) {
      Diags.setSuppressSystemWarnings(!isPositive);
      continue;
    }
    
    // -Werror/-Wno-error is a special case, not controlled by the option table.
    // It also has the "specifier" form of -Werror=foo.
    if (OptEnd-OptStart >= 5 && memcmp(OptStart, "error", 5) == 0) {
      const char *Specifier = 0;
      if (OptEnd-OptStart != 5) {  // Specifier must be present.
        if (OptStart[5] != '=' || OptEnd-OptStart == 6) {
          fprintf(stderr, "error: unknown warning option: -W%s\n", Opt.c_str());
          return true;
        }
        Specifier = OptStart+6;
      }
      
      if (Specifier == 0) {
        Diags.setWarningsAsErrors(true);
        continue;
      }
      
      // -Werror=foo maps foo to Error, -Wno-error=foo maps it to Warning.
      Mapping = isPositive ? diag::MAP_ERROR : diag::MAP_WARNING;
      OptStart = Specifier;
    }
    
    WarningOption Key = { OptStart, 0, 0 };
    const WarningOption *Found =
      std::lower_bound(OptionTable, OptionTable + OptionTableSize, Key,
                       WarningOptionCompare);
    if (Found == OptionTable + OptionTableSize ||
        strcmp(Found->Name, OptStart) != 0) {
      fprintf(stderr, "error: unknown warning option: -W%s\n", Opt.c_str());
      return true;
    }
    
    // Option exists, poke all the members of its diagnostic set.
    for (const diag::kind *Member = Found->Members,
         *E = Found->Members+Found->NumMembers; Member != E; ++Member) {
      Diags.setDiagnosticMapping(*Member, Mapping);
      assert(*Member < 65536 && "ControlledDiags element too small");
      ControlledDiags.push_back(*Member);
    }
  }

  // If -pedantic or -pedantic-errors was specified, then we want to map all
  // extension diagnostics onto WARNING or ERROR unless the user has futz'd
  // around with them explicitly.
  if (OptPedantic || OptPedanticErrors) {
    // Sort the array of options that has been poked at directly so we can do
    // efficient queries.
    std::sort(ControlledDiags.begin(), ControlledDiags.end());
    
    // Don't worry about iteration off the end down below.
    ControlledDiags.push_back(diag::DIAG_UPPER_LIMIT);
    
    diag::Mapping Mapping = 
      OptPedanticErrors ? diag::MAP_ERROR : diag::MAP_WARNING;

    // Loop over all of the extension diagnostics.  Unless they were explicitly
    // controlled, reset their mapping to Mapping.  We walk through the
    // ControlledDiags in parallel with this walk, which is faster than
    // repeatedly binary searching it.
    //
    llvm::SmallVectorImpl<unsigned short>::iterator ControlledDiagsIt =
      ControlledDiags.begin();
    
    // TODO: if it matters, we could make tblgen produce a list of just the
    // extension diags to avoid skipping ones that don't matter.
    for (unsigned short i = 0; i != diag::DIAG_UPPER_LIMIT; ++i) {
      // If this diagnostic was controlled, ignore it.
      if (i == *ControlledDiagsIt) {
        ++ControlledDiagsIt;
        while (i == *ControlledDiagsIt)  // ControlledDiags can have dupes.
          ++ControlledDiagsIt;
        // Do not map this diagnostic ID#.
        continue;
      }

      // Okay, the user didn't control this ID.  If it is an example, map it.
      if (Diagnostic::isBuiltinExtensionDiag(i))
        Diags.setDiagnosticMapping(i, Mapping);
    }
  }
  
  return false;
}
