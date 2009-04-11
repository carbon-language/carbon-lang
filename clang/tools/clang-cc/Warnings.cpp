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
// Warning options control the handling of the warnings that Clang emits. There
// are three possible reactions to any given warning:
// ignore: Do nothing
// warn:   Emit a message, but don't fail the compilation
// error:  Emit a message and fail the compilation
//
// Each warning option controls any number of actual warnings.
// Given a warning option 'foo', the following are valid:
// -Wfoo=ignore  -> Ignore the controlled warnings.
// -Wfoo=warn    -> Warn about the controlled warnings.
// -Wfoo=error   -> Fail on the controlled warnings.
// -Wfoo         -> alias of -Wfoo=warn
// -Wno-foo      -> alias of -Wfoo=ignore
// -Werror=foo   -> alias of -Wfoo=error
//
// Because of this complex handling of options, the default parser is replaced.

#include "clang-cc.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Lex/LexDiagnostic.h"
#include "llvm/Support/CommandLine.h"
#include <cstdio>
#include <utility>
#include <algorithm>
using namespace clang;

namespace {
  struct ParsedOption {
    std::string Name;
    diag::Mapping Mapping;

    ParsedOption() {}
    // Used by -Werror, implicitly.
    ParsedOption(const std::string& name) : Name(name), Mapping(diag::MAP_ERROR)
    {}
  };

  typedef std::vector<ParsedOption> OptionsList;

  OptionsList Options;

  struct WarningParser : public llvm::cl::basic_parser<ParsedOption> {
    diag::Mapping StrToMapping(const std::string &S) {
      if (S == "ignore")
        return diag::MAP_IGNORE;
      if (S == "warn")
        return diag::MAP_WARNING;
      if (S == "error")
        return diag::MAP_ERROR;
      return diag::MAP_DEFAULT;
    }
    bool parse(llvm::cl::Option &O, const char *ArgName,
               const std::string &ArgValue, ParsedOption &Val)
    {
      size_t Eq = ArgValue.find("=");
      if (Eq == std::string::npos) {
        // Could be -Wfoo or -Wno-foo
        if (ArgValue.compare(0, 3, "no-") == 0) {
          Val.Name = ArgValue.substr(3);
          Val.Mapping = diag::MAP_IGNORE;
        } else {
          Val.Name = ArgValue;
          Val.Mapping = diag::MAP_WARNING;
        }
      } else {
        Val.Name = ArgValue.substr(0, Eq);
        Val.Mapping = StrToMapping(ArgValue.substr(Eq+1));
        if (Val.Mapping == diag::MAP_DEFAULT) {
          fprintf(stderr, "Illegal warning option value: %s\n",
                  ArgValue.substr(Eq+1).c_str());
          return true;
        }
      }
      return false;
    }
  };
}

static llvm::cl::list<ParsedOption, OptionsList, WarningParser>
OptWarnings("W", llvm::cl::location(Options), llvm::cl::Prefix);

static llvm::cl::list<ParsedOption, OptionsList, llvm::cl::parser<std::string> >
OptWError("Werror", llvm::cl::location(Options), llvm::cl::CommaSeparated,
          llvm::cl::ValueOptional);

static llvm::cl::opt<bool> OptPedantic("pedantic");
static llvm::cl::opt<bool> OptPedanticErrors("pedantic-errors");
static llvm::cl::opt<bool> OptNoWarnings("w");
static llvm::cl::opt<bool>
OptWarnInSystemHeaders("Wsystem-headers",
           llvm::cl::desc("Do not suppress warnings issued in system headers"));

namespace {
  struct WarningOption {
    const char *Name;
    const diag::kind *Members;
    size_t NumMembers;
  };
}
#define DIAGS(a) a, (sizeof(a) / sizeof(a[0]))
// These tables will be TableGenerated later.
// First the table sets describing the diagnostics controlled by each option.
static const diag::kind UnusedMacrosDiags[] = { diag::pp_macro_not_used };
static const diag::kind FloatEqualDiags[] = { diag::warn_floatingpoint_eq };
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

// Hmm ... this option is currently actually completely ignored.
//static const diag::kind StrictSelectorMatchDiags[] = {  };
// Second the table of options.  MUST be sorted by name! Binary lookup is done.
static const WarningOption OptionTable[] = {
  { "deprecated-declarations", DIAGS(DeprecatedDeclarations) },
  { "float-equal",           DIAGS(FloatEqualDiags) },
  { "format-nonliteral",     DIAGS(FormatNonLiteralDiags) },
  { "implicit-function-declaration", DIAGS(ImplicitFunctionDeclarationDiags) },
  { "missing-prototypes", DIAGS(MissingPrototypesDiags) },
  { "pointer-sign",          DIAGS(PointerSignDiags) },
  { "readonly-setter-attrs", DIAGS(ReadOnlySetterAttrsDiags) },
  { "trigraphs",             DIAGS(TrigraphsDiags) },
  { "undef",                 DIAGS(UndefDiags) },
  { "unused-macros",         DIAGS(UnusedMacrosDiags) },
//  { "strict-selector-match", DIAGS(StrictSelectorMatchDiags) }
};
static const size_t OptionTableSize =
  sizeof(OptionTable) / sizeof(OptionTable[0]);

static bool WarningOptionCompare(const WarningOption &LHS,
                                 const WarningOption &RHS) {
  return strcmp(LHS.Name, RHS.Name) < 0;
}

bool clang::ProcessWarningOptions(Diagnostic &Diags) {
  // FIXME: These should be mapped to group options.
  Diags.setIgnoreAllWarnings(OptNoWarnings);
  Diags.setWarnOnExtensions(OptPedantic);
  Diags.setErrorOnExtensions(OptPedanticErrors);

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
  Diags.setSuppressSystemWarnings(!OptWarnInSystemHeaders);

  for (OptionsList::iterator it = Options.begin(), e = Options.end();
      it != e; ++it) {
    if (it->Name.empty()) {
      // Empty string is "everything". This way, -Werror does the right thing.
      // FIXME: These flags do not participate in proper option overriding.
      switch(it->Mapping) {
      default:
        assert(false && "Illegal mapping");
        break;

      case diag::MAP_IGNORE:
        Diags.setIgnoreAllWarnings(true);
        Diags.setWarningsAsErrors(false);
        break;

      case diag::MAP_WARNING:
        Diags.setIgnoreAllWarnings(false);
        Diags.setWarningsAsErrors(false);
        break;

      case diag::MAP_ERROR:
        Diags.setIgnoreAllWarnings(false);
        Diags.setWarningsAsErrors(true);
        break;
      }
      continue;
    }
    WarningOption Key = { it->Name.c_str(), 0, 0 };
    const WarningOption *Found =
      std::lower_bound(OptionTable, OptionTable + OptionTableSize, Key,
                       WarningOptionCompare);
    if (Found == OptionTable + OptionTableSize ||
        strcmp(Found->Name, Key.Name) != 0) {
      fprintf(stderr, "Unknown warning option: -W%s\n", Key.Name);
      return true;
    }

    // Option exists.
    for (size_t i = 0, e = Found->NumMembers; i != e; ++i)
      Diags.setDiagnosticMapping(Found->Members[i], it->Mapping);
  }
  return false;
}
