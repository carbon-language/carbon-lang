//===--- Format.cpp - Format C++ code -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements functions declared in Format.h. This will be
/// split into separate files as we go.
///
//===----------------------------------------------------------------------===//

#include "ContinuationIndenter.h"
#include "TokenAnnotator.h"
#include "UnwrappedLineParser.h"
#include "WhitespaceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
#include <queue>
#include <string>

#define DEBUG_TYPE "format-formatter"

using clang::format::FormatStyle;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::string)

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<FormatStyle::LanguageKind> {
  static void enumeration(IO &IO, FormatStyle::LanguageKind &Value) {
    IO.enumCase(Value, "Cpp", FormatStyle::LK_Cpp);
    IO.enumCase(Value, "Java", FormatStyle::LK_Java);
    IO.enumCase(Value, "JavaScript", FormatStyle::LK_JavaScript);
    IO.enumCase(Value, "Proto", FormatStyle::LK_Proto);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::LanguageStandard> {
  static void enumeration(IO &IO, FormatStyle::LanguageStandard &Value) {
    IO.enumCase(Value, "Cpp03", FormatStyle::LS_Cpp03);
    IO.enumCase(Value, "C++03", FormatStyle::LS_Cpp03);
    IO.enumCase(Value, "Cpp11", FormatStyle::LS_Cpp11);
    IO.enumCase(Value, "C++11", FormatStyle::LS_Cpp11);
    IO.enumCase(Value, "Auto", FormatStyle::LS_Auto);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::UseTabStyle> {
  static void enumeration(IO &IO, FormatStyle::UseTabStyle &Value) {
    IO.enumCase(Value, "Never", FormatStyle::UT_Never);
    IO.enumCase(Value, "false", FormatStyle::UT_Never);
    IO.enumCase(Value, "Always", FormatStyle::UT_Always);
    IO.enumCase(Value, "true", FormatStyle::UT_Always);
    IO.enumCase(Value, "ForIndentation", FormatStyle::UT_ForIndentation);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::ShortFunctionStyle> {
  static void enumeration(IO &IO, FormatStyle::ShortFunctionStyle &Value) {
    IO.enumCase(Value, "None", FormatStyle::SFS_None);
    IO.enumCase(Value, "false", FormatStyle::SFS_None);
    IO.enumCase(Value, "All", FormatStyle::SFS_All);
    IO.enumCase(Value, "true", FormatStyle::SFS_All);
    IO.enumCase(Value, "Inline", FormatStyle::SFS_Inline);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::BinaryOperatorStyle> {
  static void enumeration(IO &IO, FormatStyle::BinaryOperatorStyle &Value) {
    IO.enumCase(Value, "All", FormatStyle::BOS_All);
    IO.enumCase(Value, "true", FormatStyle::BOS_All);
    IO.enumCase(Value, "None", FormatStyle::BOS_None);
    IO.enumCase(Value, "false", FormatStyle::BOS_None);
    IO.enumCase(Value, "NonAssignment", FormatStyle::BOS_NonAssignment);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::BraceBreakingStyle> {
  static void enumeration(IO &IO, FormatStyle::BraceBreakingStyle &Value) {
    IO.enumCase(Value, "Attach", FormatStyle::BS_Attach);
    IO.enumCase(Value, "Linux", FormatStyle::BS_Linux);
    IO.enumCase(Value, "Stroustrup", FormatStyle::BS_Stroustrup);
    IO.enumCase(Value, "Allman", FormatStyle::BS_Allman);
    IO.enumCase(Value, "GNU", FormatStyle::BS_GNU);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::NamespaceIndentationKind> {
  static void enumeration(IO &IO,
                          FormatStyle::NamespaceIndentationKind &Value) {
    IO.enumCase(Value, "None", FormatStyle::NI_None);
    IO.enumCase(Value, "Inner", FormatStyle::NI_Inner);
    IO.enumCase(Value, "All", FormatStyle::NI_All);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::PointerAlignmentStyle> {
  static void enumeration(IO &IO,
                          FormatStyle::PointerAlignmentStyle &Value) {
    IO.enumCase(Value, "Middle", FormatStyle::PAS_Middle);
    IO.enumCase(Value, "Left", FormatStyle::PAS_Left);
    IO.enumCase(Value, "Right", FormatStyle::PAS_Right);

    // For backward compatibility.
    IO.enumCase(Value, "true", FormatStyle::PAS_Left);
    IO.enumCase(Value, "false", FormatStyle::PAS_Right);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::SpaceBeforeParensOptions> {
  static void enumeration(IO &IO,
                          FormatStyle::SpaceBeforeParensOptions &Value) {
    IO.enumCase(Value, "Never", FormatStyle::SBPO_Never);
    IO.enumCase(Value, "ControlStatements",
                FormatStyle::SBPO_ControlStatements);
    IO.enumCase(Value, "Always", FormatStyle::SBPO_Always);

    // For backward compatibility.
    IO.enumCase(Value, "false", FormatStyle::SBPO_Never);
    IO.enumCase(Value, "true", FormatStyle::SBPO_ControlStatements);
  }
};

template <> struct MappingTraits<FormatStyle> {
  static void mapping(IO &IO, FormatStyle &Style) {
    // When reading, read the language first, we need it for getPredefinedStyle.
    IO.mapOptional("Language", Style.Language);

    if (IO.outputting()) {
      StringRef StylesArray[] = { "LLVM",    "Google", "Chromium",
                                  "Mozilla", "WebKit", "GNU" };
      ArrayRef<StringRef> Styles(StylesArray);
      for (size_t i = 0, e = Styles.size(); i < e; ++i) {
        StringRef StyleName(Styles[i]);
        FormatStyle PredefinedStyle;
        if (getPredefinedStyle(StyleName, Style.Language, &PredefinedStyle) &&
            Style == PredefinedStyle) {
          IO.mapOptional("# BasedOnStyle", StyleName);
          break;
        }
      }
    } else {
      StringRef BasedOnStyle;
      IO.mapOptional("BasedOnStyle", BasedOnStyle);
      if (!BasedOnStyle.empty()) {
        FormatStyle::LanguageKind OldLanguage = Style.Language;
        FormatStyle::LanguageKind Language =
            ((FormatStyle *)IO.getContext())->Language;
        if (!getPredefinedStyle(BasedOnStyle, Language, &Style)) {
          IO.setError(Twine("Unknown value for BasedOnStyle: ", BasedOnStyle));
          return;
        }
        Style.Language = OldLanguage;
      }
    }

    IO.mapOptional("AccessModifierOffset", Style.AccessModifierOffset);
    IO.mapOptional("AlignEscapedNewlinesLeft", Style.AlignEscapedNewlinesLeft);
    IO.mapOptional("AlignTrailingComments", Style.AlignTrailingComments);
    IO.mapOptional("AllowAllParametersOfDeclarationOnNextLine",
                   Style.AllowAllParametersOfDeclarationOnNextLine);
    IO.mapOptional("AllowShortBlocksOnASingleLine",
                   Style.AllowShortBlocksOnASingleLine);
    IO.mapOptional("AllowShortCaseLabelsOnASingleLine",
                   Style.AllowShortCaseLabelsOnASingleLine);
    IO.mapOptional("AllowShortIfStatementsOnASingleLine",
                   Style.AllowShortIfStatementsOnASingleLine);
    IO.mapOptional("AllowShortLoopsOnASingleLine",
                   Style.AllowShortLoopsOnASingleLine);
    IO.mapOptional("AllowShortFunctionsOnASingleLine",
                   Style.AllowShortFunctionsOnASingleLine);
    IO.mapOptional("AlwaysBreakAfterDefinitionReturnType",
                   Style.AlwaysBreakAfterDefinitionReturnType);
    IO.mapOptional("AlwaysBreakTemplateDeclarations",
                   Style.AlwaysBreakTemplateDeclarations);
    IO.mapOptional("AlwaysBreakBeforeMultilineStrings",
                   Style.AlwaysBreakBeforeMultilineStrings);
    IO.mapOptional("BreakBeforeBinaryOperators",
                   Style.BreakBeforeBinaryOperators);
    IO.mapOptional("BreakBeforeTernaryOperators",
                   Style.BreakBeforeTernaryOperators);
    IO.mapOptional("BreakConstructorInitializersBeforeComma",
                   Style.BreakConstructorInitializersBeforeComma);
    IO.mapOptional("BinPackParameters", Style.BinPackParameters);
    IO.mapOptional("BinPackArguments", Style.BinPackArguments);
    IO.mapOptional("ColumnLimit", Style.ColumnLimit);
    IO.mapOptional("ConstructorInitializerAllOnOneLineOrOnePerLine",
                   Style.ConstructorInitializerAllOnOneLineOrOnePerLine);
    IO.mapOptional("ConstructorInitializerIndentWidth",
                   Style.ConstructorInitializerIndentWidth);
    IO.mapOptional("DerivePointerAlignment", Style.DerivePointerAlignment);
    IO.mapOptional("ExperimentalAutoDetectBinPacking",
                   Style.ExperimentalAutoDetectBinPacking);
    IO.mapOptional("IndentCaseLabels", Style.IndentCaseLabels);
    IO.mapOptional("IndentWrappedFunctionNames",
                   Style.IndentWrappedFunctionNames);
    IO.mapOptional("IndentFunctionDeclarationAfterType",
                   Style.IndentWrappedFunctionNames);
    IO.mapOptional("MaxEmptyLinesToKeep", Style.MaxEmptyLinesToKeep);
    IO.mapOptional("KeepEmptyLinesAtTheStartOfBlocks",
                   Style.KeepEmptyLinesAtTheStartOfBlocks);
    IO.mapOptional("NamespaceIndentation", Style.NamespaceIndentation);
    IO.mapOptional("ObjCBlockIndentWidth", Style.ObjCBlockIndentWidth);
    IO.mapOptional("ObjCSpaceAfterProperty", Style.ObjCSpaceAfterProperty);
    IO.mapOptional("ObjCSpaceBeforeProtocolList",
                   Style.ObjCSpaceBeforeProtocolList);
    IO.mapOptional("PenaltyBreakBeforeFirstCallParameter",
                   Style.PenaltyBreakBeforeFirstCallParameter);
    IO.mapOptional("PenaltyBreakComment", Style.PenaltyBreakComment);
    IO.mapOptional("PenaltyBreakString", Style.PenaltyBreakString);
    IO.mapOptional("PenaltyBreakFirstLessLess",
                   Style.PenaltyBreakFirstLessLess);
    IO.mapOptional("PenaltyExcessCharacter", Style.PenaltyExcessCharacter);
    IO.mapOptional("PenaltyReturnTypeOnItsOwnLine",
                   Style.PenaltyReturnTypeOnItsOwnLine);
    IO.mapOptional("PointerAlignment", Style.PointerAlignment);
    IO.mapOptional("SpacesBeforeTrailingComments",
                   Style.SpacesBeforeTrailingComments);
    IO.mapOptional("Cpp11BracedListStyle", Style.Cpp11BracedListStyle);
    IO.mapOptional("Standard", Style.Standard);
    IO.mapOptional("IndentWidth", Style.IndentWidth);
    IO.mapOptional("TabWidth", Style.TabWidth);
    IO.mapOptional("UseTab", Style.UseTab);
    IO.mapOptional("BreakBeforeBraces", Style.BreakBeforeBraces);
    IO.mapOptional("SpacesInParentheses", Style.SpacesInParentheses);
    IO.mapOptional("SpacesInSquareBrackets", Style.SpacesInSquareBrackets);
    IO.mapOptional("SpacesInAngles", Style.SpacesInAngles);
    IO.mapOptional("SpaceInEmptyParentheses", Style.SpaceInEmptyParentheses);
    IO.mapOptional("SpacesInCStyleCastParentheses",
                   Style.SpacesInCStyleCastParentheses);
    IO.mapOptional("SpaceAfterCStyleCast", Style.SpaceAfterCStyleCast);
    IO.mapOptional("SpacesInContainerLiterals",
                   Style.SpacesInContainerLiterals);
    IO.mapOptional("SpaceBeforeAssignmentOperators",
                   Style.SpaceBeforeAssignmentOperators);
    IO.mapOptional("ContinuationIndentWidth", Style.ContinuationIndentWidth);
    IO.mapOptional("CommentPragmas", Style.CommentPragmas);
    IO.mapOptional("ForEachMacros", Style.ForEachMacros);

    // For backward compatibility.
    if (!IO.outputting()) {
      IO.mapOptional("SpaceAfterControlStatementKeyword",
                     Style.SpaceBeforeParens);
      IO.mapOptional("PointerBindsToType", Style.PointerAlignment);
      IO.mapOptional("DerivePointerBinding", Style.DerivePointerAlignment);
    }
    IO.mapOptional("SpaceBeforeParens", Style.SpaceBeforeParens);
    IO.mapOptional("DisableFormat", Style.DisableFormat);
  }
};

// Allows to read vector<FormatStyle> while keeping default values.
// IO.getContext() should contain a pointer to the FormatStyle structure, that
// will be used to get default values for missing keys.
// If the first element has no Language specified, it will be treated as the
// default one for the following elements.
template <> struct DocumentListTraits<std::vector<FormatStyle> > {
  static size_t size(IO &IO, std::vector<FormatStyle> &Seq) {
    return Seq.size();
  }
  static FormatStyle &element(IO &IO, std::vector<FormatStyle> &Seq,
                              size_t Index) {
    if (Index >= Seq.size()) {
      assert(Index == Seq.size());
      FormatStyle Template;
      if (Seq.size() > 0 && Seq[0].Language == FormatStyle::LK_None) {
        Template = Seq[0];
      } else {
        Template = *((const FormatStyle *)IO.getContext());
        Template.Language = FormatStyle::LK_None;
      }
      Seq.resize(Index + 1, Template);
    }
    return Seq[Index];
  }
};
}
}

namespace clang {
namespace format {

const std::error_category &getParseCategory() {
  static ParseErrorCategory C;
  return C;
}
std::error_code make_error_code(ParseError e) {
  return std::error_code(static_cast<int>(e), getParseCategory());
}

const char *ParseErrorCategory::name() const LLVM_NOEXCEPT {
  return "clang-format.parse_error";
}

std::string ParseErrorCategory::message(int EV) const {
  switch (static_cast<ParseError>(EV)) {
  case ParseError::Success:
    return "Success";
  case ParseError::Error:
    return "Invalid argument";
  case ParseError::Unsuitable:
    return "Unsuitable";
  }
  llvm_unreachable("unexpected parse error");
}

FormatStyle getLLVMStyle() {
  FormatStyle LLVMStyle;
  LLVMStyle.Language = FormatStyle::LK_Cpp;
  LLVMStyle.AccessModifierOffset = -2;
  LLVMStyle.AlignEscapedNewlinesLeft = false;
  LLVMStyle.AlignTrailingComments = true;
  LLVMStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  LLVMStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  LLVMStyle.AllowShortBlocksOnASingleLine = false;
  LLVMStyle.AllowShortCaseLabelsOnASingleLine = false;
  LLVMStyle.AllowShortIfStatementsOnASingleLine = false;
  LLVMStyle.AllowShortLoopsOnASingleLine = false;
  LLVMStyle.AlwaysBreakAfterDefinitionReturnType = false;
  LLVMStyle.AlwaysBreakBeforeMultilineStrings = false;
  LLVMStyle.AlwaysBreakTemplateDeclarations = false;
  LLVMStyle.BinPackParameters = true;
  LLVMStyle.BinPackArguments = true;
  LLVMStyle.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  LLVMStyle.BreakBeforeTernaryOperators = true;
  LLVMStyle.BreakBeforeBraces = FormatStyle::BS_Attach;
  LLVMStyle.BreakConstructorInitializersBeforeComma = false;
  LLVMStyle.ColumnLimit = 80;
  LLVMStyle.CommentPragmas = "^ IWYU pragma:";
  LLVMStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = false;
  LLVMStyle.ConstructorInitializerIndentWidth = 4;
  LLVMStyle.ContinuationIndentWidth = 4;
  LLVMStyle.Cpp11BracedListStyle = true;
  LLVMStyle.DerivePointerAlignment = false;
  LLVMStyle.ExperimentalAutoDetectBinPacking = false;
  LLVMStyle.ForEachMacros.push_back("foreach");
  LLVMStyle.ForEachMacros.push_back("Q_FOREACH");
  LLVMStyle.ForEachMacros.push_back("BOOST_FOREACH");
  LLVMStyle.IndentCaseLabels = false;
  LLVMStyle.IndentWrappedFunctionNames = false;
  LLVMStyle.IndentWidth = 2;
  LLVMStyle.TabWidth = 8;
  LLVMStyle.MaxEmptyLinesToKeep = 1;
  LLVMStyle.KeepEmptyLinesAtTheStartOfBlocks = true;
  LLVMStyle.NamespaceIndentation = FormatStyle::NI_None;
  LLVMStyle.ObjCBlockIndentWidth = 2;
  LLVMStyle.ObjCSpaceAfterProperty = false;
  LLVMStyle.ObjCSpaceBeforeProtocolList = true;
  LLVMStyle.PointerAlignment = FormatStyle::PAS_Right;
  LLVMStyle.SpacesBeforeTrailingComments = 1;
  LLVMStyle.Standard = FormatStyle::LS_Cpp11;
  LLVMStyle.UseTab = FormatStyle::UT_Never;
  LLVMStyle.SpacesInParentheses = false;
  LLVMStyle.SpacesInSquareBrackets = false;
  LLVMStyle.SpaceInEmptyParentheses = false;
  LLVMStyle.SpacesInContainerLiterals = true;
  LLVMStyle.SpacesInCStyleCastParentheses = false;
  LLVMStyle.SpaceAfterCStyleCast = false;
  LLVMStyle.SpaceBeforeParens = FormatStyle::SBPO_ControlStatements;
  LLVMStyle.SpaceBeforeAssignmentOperators = true;
  LLVMStyle.SpacesInAngles = false;

  LLVMStyle.PenaltyBreakComment = 300;
  LLVMStyle.PenaltyBreakFirstLessLess = 120;
  LLVMStyle.PenaltyBreakString = 1000;
  LLVMStyle.PenaltyExcessCharacter = 1000000;
  LLVMStyle.PenaltyReturnTypeOnItsOwnLine = 60;
  LLVMStyle.PenaltyBreakBeforeFirstCallParameter = 19;

  LLVMStyle.DisableFormat = false;

  return LLVMStyle;
}

FormatStyle getGoogleStyle(FormatStyle::LanguageKind Language) {
  FormatStyle GoogleStyle = getLLVMStyle();
  GoogleStyle.Language = Language;

  GoogleStyle.AccessModifierOffset = -1;
  GoogleStyle.AlignEscapedNewlinesLeft = true;
  GoogleStyle.AllowShortIfStatementsOnASingleLine = true;
  GoogleStyle.AllowShortLoopsOnASingleLine = true;
  GoogleStyle.AlwaysBreakBeforeMultilineStrings = true;
  GoogleStyle.AlwaysBreakTemplateDeclarations = true;
  GoogleStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  GoogleStyle.DerivePointerAlignment = true;
  GoogleStyle.IndentCaseLabels = true;
  GoogleStyle.KeepEmptyLinesAtTheStartOfBlocks = false;
  GoogleStyle.ObjCSpaceAfterProperty = false;
  GoogleStyle.ObjCSpaceBeforeProtocolList = false;
  GoogleStyle.PointerAlignment = FormatStyle::PAS_Left;
  GoogleStyle.SpacesBeforeTrailingComments = 2;
  GoogleStyle.Standard = FormatStyle::LS_Auto;

  GoogleStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  GoogleStyle.PenaltyBreakBeforeFirstCallParameter = 1;

  if (Language == FormatStyle::LK_Java) {
    GoogleStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
    GoogleStyle.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
    GoogleStyle.ColumnLimit = 100;
    GoogleStyle.SpaceAfterCStyleCast = true;
  } else if (Language == FormatStyle::LK_JavaScript) {
    GoogleStyle.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
    GoogleStyle.MaxEmptyLinesToKeep = 3;
    GoogleStyle.SpacesInContainerLiterals = false;
    GoogleStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  } else if (Language == FormatStyle::LK_Proto) {
    GoogleStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
    GoogleStyle.SpacesInContainerLiterals = false;
  }

  return GoogleStyle;
}

FormatStyle getChromiumStyle(FormatStyle::LanguageKind Language) {
  FormatStyle ChromiumStyle = getGoogleStyle(Language);
  ChromiumStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  ChromiumStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  ChromiumStyle.AllowShortIfStatementsOnASingleLine = false;
  ChromiumStyle.AllowShortLoopsOnASingleLine = false;
  ChromiumStyle.BinPackParameters = false;
  ChromiumStyle.DerivePointerAlignment = false;
  return ChromiumStyle;
}

FormatStyle getMozillaStyle() {
  FormatStyle MozillaStyle = getLLVMStyle();
  MozillaStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  MozillaStyle.Cpp11BracedListStyle = false;
  MozillaStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  MozillaStyle.DerivePointerAlignment = true;
  MozillaStyle.IndentCaseLabels = true;
  MozillaStyle.ObjCSpaceAfterProperty = true;
  MozillaStyle.ObjCSpaceBeforeProtocolList = false;
  MozillaStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  MozillaStyle.PointerAlignment = FormatStyle::PAS_Left;
  MozillaStyle.Standard = FormatStyle::LS_Cpp03;
  return MozillaStyle;
}

FormatStyle getWebKitStyle() {
  FormatStyle Style = getLLVMStyle();
  Style.AccessModifierOffset = -4;
  Style.AlignTrailingComments = false;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  Style.BreakConstructorInitializersBeforeComma = true;
  Style.Cpp11BracedListStyle = false;
  Style.ColumnLimit = 0;
  Style.IndentWidth = 4;
  Style.NamespaceIndentation = FormatStyle::NI_Inner;
  Style.ObjCBlockIndentWidth = 4;
  Style.ObjCSpaceAfterProperty = true;
  Style.PointerAlignment = FormatStyle::PAS_Left;
  Style.Standard = FormatStyle::LS_Cpp03;
  return Style;
}

FormatStyle getGNUStyle() {
  FormatStyle Style = getLLVMStyle();
  Style.AlwaysBreakAfterDefinitionReturnType = true;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  Style.BreakBeforeBraces = FormatStyle::BS_GNU;
  Style.BreakBeforeTernaryOperators = true;
  Style.Cpp11BracedListStyle = false;
  Style.ColumnLimit = 79;
  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;
  Style.Standard = FormatStyle::LS_Cpp03;
  return Style;
}

FormatStyle getNoStyle() {
  FormatStyle NoStyle = getLLVMStyle();
  NoStyle.DisableFormat = true;
  return NoStyle;
}

bool getPredefinedStyle(StringRef Name, FormatStyle::LanguageKind Language,
                        FormatStyle *Style) {
  if (Name.equals_lower("llvm")) {
    *Style = getLLVMStyle();
  } else if (Name.equals_lower("chromium")) {
    *Style = getChromiumStyle(Language);
  } else if (Name.equals_lower("mozilla")) {
    *Style = getMozillaStyle();
  } else if (Name.equals_lower("google")) {
    *Style = getGoogleStyle(Language);
  } else if (Name.equals_lower("webkit")) {
    *Style = getWebKitStyle();
  } else if (Name.equals_lower("gnu")) {
    *Style = getGNUStyle();
  } else if (Name.equals_lower("none")) {
    *Style = getNoStyle();
  } else {
    return false;
  }

  Style->Language = Language;
  return true;
}

std::error_code parseConfiguration(StringRef Text, FormatStyle *Style) {
  assert(Style);
  FormatStyle::LanguageKind Language = Style->Language;
  assert(Language != FormatStyle::LK_None);
  if (Text.trim().empty())
    return make_error_code(ParseError::Error);

  std::vector<FormatStyle> Styles;
  llvm::yaml::Input Input(Text);
  // DocumentListTraits<vector<FormatStyle>> uses the context to get default
  // values for the fields, keys for which are missing from the configuration.
  // Mapping also uses the context to get the language to find the correct
  // base style.
  Input.setContext(Style);
  Input >> Styles;
  if (Input.error())
    return Input.error();

  for (unsigned i = 0; i < Styles.size(); ++i) {
    // Ensures that only the first configuration can skip the Language option.
    if (Styles[i].Language == FormatStyle::LK_None && i != 0)
      return make_error_code(ParseError::Error);
    // Ensure that each language is configured at most once.
    for (unsigned j = 0; j < i; ++j) {
      if (Styles[i].Language == Styles[j].Language) {
        DEBUG(llvm::dbgs()
              << "Duplicate languages in the config file on positions " << j
              << " and " << i << "\n");
        return make_error_code(ParseError::Error);
      }
    }
  }
  // Look for a suitable configuration starting from the end, so we can
  // find the configuration for the specific language first, and the default
  // configuration (which can only be at slot 0) after it.
  for (int i = Styles.size() - 1; i >= 0; --i) {
    if (Styles[i].Language == Language ||
        Styles[i].Language == FormatStyle::LK_None) {
      *Style = Styles[i];
      Style->Language = Language;
      return make_error_code(ParseError::Success);
    }
  }
  return make_error_code(ParseError::Unsuitable);
}

std::string configurationAsText(const FormatStyle &Style) {
  std::string Text;
  llvm::raw_string_ostream Stream(Text);
  llvm::yaml::Output Output(Stream);
  // We use the same mapping method for input and output, so we need a non-const
  // reference here.
  FormatStyle NonConstStyle = Style;
  Output << NonConstStyle;
  return Stream.str();
}

namespace {

class NoColumnLimitFormatter {
public:
  NoColumnLimitFormatter(ContinuationIndenter *Indenter) : Indenter(Indenter) {}

  /// \brief Formats the line starting at \p State, simply keeping all of the
  /// input's line breaking decisions.
  void format(unsigned FirstIndent, const AnnotatedLine *Line) {
    LineState State =
        Indenter->getInitialState(FirstIndent, Line, /*DryRun=*/false);
    while (State.NextToken) {
      bool Newline =
          Indenter->mustBreak(State) ||
          (Indenter->canBreak(State) && State.NextToken->NewlinesBefore > 0);
      Indenter->addTokenToState(State, Newline, /*DryRun=*/false);
    }
  }

private:
  ContinuationIndenter *Indenter;
};

class LineJoiner {
public:
  LineJoiner(const FormatStyle &Style) : Style(Style) {}

  /// \brief Calculates how many lines can be merged into 1 starting at \p I.
  unsigned
  tryFitMultipleLinesInOne(unsigned Indent,
                           SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                           SmallVectorImpl<AnnotatedLine *>::const_iterator E) {
    // We can never merge stuff if there are trailing line comments.
    const AnnotatedLine *TheLine = *I;
    if (TheLine->Last->Type == TT_LineComment)
      return 0;

    if (Style.ColumnLimit > 0 && Indent > Style.ColumnLimit)
      return 0;

    unsigned Limit =
        Style.ColumnLimit == 0 ? UINT_MAX : Style.ColumnLimit - Indent;
    // If we already exceed the column limit, we set 'Limit' to 0. The different
    // tryMerge..() functions can then decide whether to still do merging.
    Limit = TheLine->Last->TotalLength > Limit
                ? 0
                : Limit - TheLine->Last->TotalLength;

    if (I + 1 == E || I[1]->Type == LT_Invalid || I[1]->First->MustBreakBefore)
      return 0;

    // FIXME: TheLine->Level != 0 might or might not be the right check to do.
    // If necessary, change to something smarter.
    bool MergeShortFunctions =
        Style.AllowShortFunctionsOnASingleLine == FormatStyle::SFS_All ||
        (Style.AllowShortFunctionsOnASingleLine == FormatStyle::SFS_Inline &&
         TheLine->Level != 0);

    if (TheLine->Last->Type == TT_FunctionLBrace &&
        TheLine->First != TheLine->Last) {
      return MergeShortFunctions ? tryMergeSimpleBlock(I, E, Limit) : 0;
    }
    if (TheLine->Last->is(tok::l_brace)) {
      return Style.BreakBeforeBraces == FormatStyle::BS_Attach
                 ? tryMergeSimpleBlock(I, E, Limit)
                 : 0;
    }
    if (I[1]->First->Type == TT_FunctionLBrace &&
        Style.BreakBeforeBraces != FormatStyle::BS_Attach) {
      // Check for Limit <= 2 to account for the " {".
      if (Limit <= 2 || (Style.ColumnLimit == 0 && containsMustBreak(TheLine)))
        return 0;
      Limit -= 2;

      unsigned MergedLines = 0;
      if (MergeShortFunctions) {
        MergedLines = tryMergeSimpleBlock(I + 1, E, Limit);
        // If we managed to merge the block, count the function header, which is
        // on a separate line.
        if (MergedLines > 0)
          ++MergedLines;
      }
      return MergedLines;
    }
    if (TheLine->First->is(tok::kw_if)) {
      return Style.AllowShortIfStatementsOnASingleLine
                 ? tryMergeSimpleControlStatement(I, E, Limit)
                 : 0;
    }
    if (TheLine->First->isOneOf(tok::kw_for, tok::kw_while)) {
      return Style.AllowShortLoopsOnASingleLine
                 ? tryMergeSimpleControlStatement(I, E, Limit)
                 : 0;
    }
    if (TheLine->First->isOneOf(tok::kw_case, tok::kw_default)) {
      return Style.AllowShortCaseLabelsOnASingleLine
                 ? tryMergeShortCaseLabels(I, E, Limit)
                 : 0;
    }
    if (TheLine->InPPDirective &&
        (TheLine->First->HasUnescapedNewline || TheLine->First->IsFirst)) {
      return tryMergeSimplePPDirective(I, E, Limit);
    }
    return 0;
  }

private:
  unsigned
  tryMergeSimplePPDirective(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                            SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                            unsigned Limit) {
    if (Limit == 0)
      return 0;
    if (!I[1]->InPPDirective || I[1]->First->HasUnescapedNewline)
      return 0;
    if (I + 2 != E && I[2]->InPPDirective && !I[2]->First->HasUnescapedNewline)
      return 0;
    if (1 + I[1]->Last->TotalLength > Limit)
      return 0;
    return 1;
  }

  unsigned tryMergeSimpleControlStatement(
      SmallVectorImpl<AnnotatedLine *>::const_iterator I,
      SmallVectorImpl<AnnotatedLine *>::const_iterator E, unsigned Limit) {
    if (Limit == 0)
      return 0;
    if ((Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
         Style.BreakBeforeBraces == FormatStyle::BS_GNU) &&
        (I[1]->First->is(tok::l_brace) && !Style.AllowShortBlocksOnASingleLine))
      return 0;
    if (I[1]->InPPDirective != (*I)->InPPDirective ||
        (I[1]->InPPDirective && I[1]->First->HasUnescapedNewline))
      return 0;
    Limit = limitConsideringMacros(I + 1, E, Limit);
    AnnotatedLine &Line = **I;
    if (Line.Last->isNot(tok::r_paren))
      return 0;
    if (1 + I[1]->Last->TotalLength > Limit)
      return 0;
    if (I[1]->First->isOneOf(tok::semi, tok::kw_if, tok::kw_for,
                             tok::kw_while) ||
        I[1]->First->Type == TT_LineComment)
      return 0;
    // Only inline simple if's (no nested if or else).
    if (I + 2 != E && Line.First->is(tok::kw_if) &&
        I[2]->First->is(tok::kw_else))
      return 0;
    return 1;
  }

  unsigned tryMergeShortCaseLabels(
      SmallVectorImpl<AnnotatedLine *>::const_iterator I,
      SmallVectorImpl<AnnotatedLine *>::const_iterator E, unsigned Limit) {
    if (Limit == 0 || I + 1 == E ||
        I[1]->First->isOneOf(tok::kw_case, tok::kw_default))
      return 0;
    unsigned NumStmts = 0;
    unsigned Length = 0;
    for (; NumStmts < 3; ++NumStmts) {
      if (I + 1 + NumStmts == E)
        break;
      const AnnotatedLine *Line = I[1 + NumStmts];
      if (Line->First->isOneOf(tok::kw_case, tok::kw_default, tok::r_brace))
        break;
      if (Line->First->isOneOf(tok::kw_if, tok::kw_for, tok::kw_switch,
                               tok::kw_while))
        return 0;
      Length += I[1 + NumStmts]->Last->TotalLength + 1; // 1 for the space.
    }
    if (NumStmts == 0 || NumStmts == 3 || Length > Limit)
      return 0;
    return NumStmts;
  }

  unsigned
  tryMergeSimpleBlock(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                      SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                      unsigned Limit) {
    AnnotatedLine &Line = **I;

    // Don't merge ObjC @ keywords and methods.
    if (Line.First->isOneOf(tok::at, tok::minus, tok::plus))
      return 0;

    // Check that the current line allows merging. This depends on whether we
    // are in a control flow statements as well as several style flags.
    if (Line.First->isOneOf(tok::kw_else, tok::kw_case))
      return 0;
    if (Line.First->isOneOf(tok::kw_if, tok::kw_while, tok::kw_do, tok::kw_try,
                            tok::kw_catch, tok::kw_for, tok::r_brace)) {
      if (!Style.AllowShortBlocksOnASingleLine)
        return 0;
      if (!Style.AllowShortIfStatementsOnASingleLine &&
          Line.First->is(tok::kw_if))
        return 0;
      if (!Style.AllowShortLoopsOnASingleLine &&
          Line.First->isOneOf(tok::kw_while, tok::kw_do, tok::kw_for))
        return 0;
      // FIXME: Consider an option to allow short exception handling clauses on
      // a single line.
      if (Line.First->isOneOf(tok::kw_try, tok::kw_catch))
        return 0;
    }

    FormatToken *Tok = I[1]->First;
    if (Tok->is(tok::r_brace) && !Tok->MustBreakBefore &&
        (Tok->getNextNonComment() == nullptr ||
         Tok->getNextNonComment()->is(tok::semi))) {
      // We merge empty blocks even if the line exceeds the column limit.
      Tok->SpacesRequiredBefore = 0;
      Tok->CanBreakBefore = true;
      return 1;
    } else if (Limit != 0 && Line.First->isNot(tok::kw_namespace)) {
      // We don't merge short records.
      if (Line.First->isOneOf(tok::kw_class, tok::kw_union, tok::kw_struct))
        return 0;

      // Check that we still have three lines and they fit into the limit.
      if (I + 2 == E || I[2]->Type == LT_Invalid)
        return 0;
      Limit = limitConsideringMacros(I + 2, E, Limit);

      if (!nextTwoLinesFitInto(I, Limit))
        return 0;

      // Second, check that the next line does not contain any braces - if it
      // does, readability declines when putting it into a single line.
      if (I[1]->Last->Type == TT_LineComment)
        return 0;
      do {
        if (Tok->is(tok::l_brace) && Tok->BlockKind != BK_BracedInit)
          return 0;
        Tok = Tok->Next;
      } while (Tok);

      // Last, check that the third line starts with a closing brace.
      Tok = I[2]->First;
      if (Tok->isNot(tok::r_brace))
        return 0;

      return 2;
    }
    return 0;
  }

  /// Returns the modified column limit for \p I if it is inside a macro and
  /// needs a trailing '\'.
  unsigned
  limitConsideringMacros(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                         SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                         unsigned Limit) {
    if (I[0]->InPPDirective && I + 1 != E &&
        !I[1]->First->HasUnescapedNewline && !I[1]->First->is(tok::eof)) {
      return Limit < 2 ? 0 : Limit - 2;
    }
    return Limit;
  }

  bool nextTwoLinesFitInto(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                           unsigned Limit) {
    if (I[1]->First->MustBreakBefore || I[2]->First->MustBreakBefore)
      return false;
    return 1 + I[1]->Last->TotalLength + 1 + I[2]->Last->TotalLength <= Limit;
  }

  bool containsMustBreak(const AnnotatedLine *Line) {
    for (const FormatToken *Tok = Line->First; Tok; Tok = Tok->Next) {
      if (Tok->MustBreakBefore)
        return true;
    }
    return false;
  }

  const FormatStyle &Style;
};

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(ContinuationIndenter *Indenter,
                         WhitespaceManager *Whitespaces,
                         const FormatStyle &Style)
      : Indenter(Indenter), Whitespaces(Whitespaces), Style(Style),
        Joiner(Style) {}

  unsigned format(const SmallVectorImpl<AnnotatedLine *> &Lines, bool DryRun,
                  int AdditionalIndent = 0, bool FixBadIndentation = false) {
    // Try to look up already computed penalty in DryRun-mode.
    std::pair<const SmallVectorImpl<AnnotatedLine *> *, unsigned> CacheKey(
        &Lines, AdditionalIndent);
    auto CacheIt = PenaltyCache.find(CacheKey);
    if (DryRun && CacheIt != PenaltyCache.end())
      return CacheIt->second;

    assert(!Lines.empty());
    unsigned Penalty = 0;
    std::vector<int> IndentForLevel;
    for (unsigned i = 0, e = Lines[0]->Level; i != e; ++i)
      IndentForLevel.push_back(Style.IndentWidth * i + AdditionalIndent);
    const AnnotatedLine *PreviousLine = nullptr;
    for (SmallVectorImpl<AnnotatedLine *>::const_iterator I = Lines.begin(),
                                                          E = Lines.end();
         I != E; ++I) {
      const AnnotatedLine &TheLine = **I;
      const FormatToken *FirstTok = TheLine.First;
      int Offset = getIndentOffset(*FirstTok);

      // Determine indent and try to merge multiple unwrapped lines.
      unsigned Indent;
      if (TheLine.InPPDirective) {
        Indent = TheLine.Level * Style.IndentWidth;
      } else {
        while (IndentForLevel.size() <= TheLine.Level)
          IndentForLevel.push_back(-1);
        IndentForLevel.resize(TheLine.Level + 1);
        Indent = getIndent(IndentForLevel, TheLine.Level);
      }
      unsigned LevelIndent = Indent;
      if (static_cast<int>(Indent) + Offset >= 0)
        Indent += Offset;

      // Merge multiple lines if possible.
      unsigned MergedLines = Joiner.tryFitMultipleLinesInOne(Indent, I, E);
      if (MergedLines > 0 && Style.ColumnLimit == 0) {
        // Disallow line merging if there is a break at the start of one of the
        // input lines.
        for (unsigned i = 0; i < MergedLines; ++i) {
          if (I[i + 1]->First->NewlinesBefore > 0)
            MergedLines = 0;
        }
      }
      if (!DryRun) {
        for (unsigned i = 0; i < MergedLines; ++i) {
          join(*I[i], *I[i + 1]);
        }
      }
      I += MergedLines;

      bool FixIndentation =
          FixBadIndentation && (LevelIndent != FirstTok->OriginalColumn);
      if (TheLine.First->is(tok::eof)) {
        if (PreviousLine && PreviousLine->Affected && !DryRun) {
          // Remove the file's trailing whitespace.
          unsigned Newlines = std::min(FirstTok->NewlinesBefore, 1u);
          Whitespaces->replaceWhitespace(*TheLine.First, Newlines,
                                         /*IndentLevel=*/0, /*Spaces=*/0,
                                         /*TargetColumn=*/0);
        }
      } else if (TheLine.Type != LT_Invalid &&
                 (TheLine.Affected || FixIndentation)) {
        if (FirstTok->WhitespaceRange.isValid()) {
          if (!DryRun)
            formatFirstToken(*TheLine.First, PreviousLine, TheLine.Level,
                             Indent, TheLine.InPPDirective);
        } else {
          Indent = LevelIndent = FirstTok->OriginalColumn;
        }

        // If everything fits on a single line, just put it there.
        unsigned ColumnLimit = Style.ColumnLimit;
        if (I + 1 != E) {
          AnnotatedLine *NextLine = I[1];
          if (NextLine->InPPDirective && !NextLine->First->HasUnescapedNewline)
            ColumnLimit = getColumnLimit(TheLine.InPPDirective);
        }

        if (TheLine.Last->TotalLength + Indent <= ColumnLimit) {
          LineState State = Indenter->getInitialState(Indent, &TheLine, DryRun);
          while (State.NextToken) {
            formatChildren(State, /*Newline=*/false, /*DryRun=*/false, Penalty);
            Indenter->addTokenToState(State, /*Newline=*/false, DryRun);
          }
        } else if (Style.ColumnLimit == 0) {
          // FIXME: Implement nested blocks for ColumnLimit = 0.
          NoColumnLimitFormatter Formatter(Indenter);
          if (!DryRun)
            Formatter.format(Indent, &TheLine);
        } else {
          Penalty += format(TheLine, Indent, DryRun);
        }

        if (!TheLine.InPPDirective)
          IndentForLevel[TheLine.Level] = LevelIndent;
      } else if (TheLine.ChildrenAffected) {
        format(TheLine.Children, DryRun);
      } else {
        // Format the first token if necessary, and notify the WhitespaceManager
        // about the unchanged whitespace.
        for (FormatToken *Tok = TheLine.First; Tok; Tok = Tok->Next) {
          if (Tok == TheLine.First &&
              (Tok->NewlinesBefore > 0 || Tok->IsFirst)) {
            unsigned LevelIndent = Tok->OriginalColumn;
            if (!DryRun) {
              // Remove trailing whitespace of the previous line.
              if ((PreviousLine && PreviousLine->Affected) ||
                  TheLine.LeadingEmptyLinesAffected) {
                formatFirstToken(*Tok, PreviousLine, TheLine.Level, LevelIndent,
                                 TheLine.InPPDirective);
              } else {
                Whitespaces->addUntouchableToken(*Tok, TheLine.InPPDirective);
              }
            }

            if (static_cast<int>(LevelIndent) - Offset >= 0)
              LevelIndent -= Offset;
            if (Tok->isNot(tok::comment) && !TheLine.InPPDirective)
              IndentForLevel[TheLine.Level] = LevelIndent;
          } else if (!DryRun) {
            Whitespaces->addUntouchableToken(*Tok, TheLine.InPPDirective);
          }
        }
      }
      if (!DryRun) {
        for (FormatToken *Tok = TheLine.First; Tok; Tok = Tok->Next) {
          Tok->Finalized = true;
        }
      }
      PreviousLine = *I;
    }
    PenaltyCache[CacheKey] = Penalty;
    return Penalty;
  }

private:
  /// \brief Formats an \c AnnotatedLine and returns the penalty.
  ///
  /// If \p DryRun is \c false, directly applies the changes.
  unsigned format(const AnnotatedLine &Line, unsigned FirstIndent,
                  bool DryRun) {
    LineState State = Indenter->getInitialState(FirstIndent, &Line, DryRun);

    // If the ObjC method declaration does not fit on a line, we should format
    // it with one arg per line.
    if (State.Line->Type == LT_ObjCMethodDecl)
      State.Stack.back().BreakBeforeParameter = true;

    // Find best solution in solution space.
    return analyzeSolutionSpace(State, DryRun);
  }

  /// \brief An edge in the solution space from \c Previous->State to \c State,
  /// inserting a newline dependent on the \c NewLine.
  struct StateNode {
    StateNode(const LineState &State, bool NewLine, StateNode *Previous)
        : State(State), NewLine(NewLine), Previous(Previous) {}
    LineState State;
    bool NewLine;
    StateNode *Previous;
  };

  /// \brief A pair of <penalty, count> that is used to prioritize the BFS on.
  ///
  /// In case of equal penalties, we want to prefer states that were inserted
  /// first. During state generation we make sure that we insert states first
  /// that break the line as late as possible.
  typedef std::pair<unsigned, unsigned> OrderedPenalty;

  /// \brief An item in the prioritized BFS search queue. The \c StateNode's
  /// \c State has the given \c OrderedPenalty.
  typedef std::pair<OrderedPenalty, StateNode *> QueueItem;

  /// \brief The BFS queue type.
  typedef std::priority_queue<QueueItem, std::vector<QueueItem>,
                              std::greater<QueueItem> > QueueType;

  /// \brief Get the offset of the line relatively to the level.
  ///
  /// For example, 'public:' labels in classes are offset by 1 or 2
  /// characters to the left from their level.
  int getIndentOffset(const FormatToken &RootToken) {
    if (Style.Language == FormatStyle::LK_Java)
      return 0;
    if (RootToken.isAccessSpecifier(false) || RootToken.isObjCAccessSpecifier())
      return Style.AccessModifierOffset;
    return 0;
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  void formatFirstToken(FormatToken &RootToken,
                        const AnnotatedLine *PreviousLine, unsigned IndentLevel,
                        unsigned Indent, bool InPPDirective) {
    unsigned Newlines =
        std::min(RootToken.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
    // Remove empty lines before "}" where applicable.
    if (RootToken.is(tok::r_brace) &&
        (!RootToken.Next ||
         (RootToken.Next->is(tok::semi) && !RootToken.Next->Next)))
      Newlines = std::min(Newlines, 1u);
    if (Newlines == 0 && !RootToken.IsFirst)
      Newlines = 1;
    if (RootToken.IsFirst && !RootToken.HasUnescapedNewline)
      Newlines = 0;

    // Remove empty lines after "{".
    if (!Style.KeepEmptyLinesAtTheStartOfBlocks && PreviousLine &&
        PreviousLine->Last->is(tok::l_brace) &&
        PreviousLine->First->isNot(tok::kw_namespace))
      Newlines = 1;

    // Insert extra new line before access specifiers.
    if (PreviousLine && PreviousLine->Last->isOneOf(tok::semi, tok::r_brace) &&
        RootToken.isAccessSpecifier() && RootToken.NewlinesBefore == 1)
      ++Newlines;

    // Remove empty lines after access specifiers.
    if (PreviousLine && PreviousLine->First->isAccessSpecifier())
      Newlines = std::min(1u, Newlines);

    Whitespaces->replaceWhitespace(RootToken, Newlines, IndentLevel, Indent,
                                   Indent, InPPDirective &&
                                               !RootToken.HasUnescapedNewline);
  }

  /// \brief Get the indent of \p Level from \p IndentForLevel.
  ///
  /// \p IndentForLevel must contain the indent for the level \c l
  /// at \p IndentForLevel[l], or a value < 0 if the indent for
  /// that level is unknown.
  unsigned getIndent(ArrayRef<int> IndentForLevel, unsigned Level) {
    if (IndentForLevel[Level] != -1)
      return IndentForLevel[Level];
    if (Level == 0)
      return 0;
    return getIndent(IndentForLevel, Level - 1) + Style.IndentWidth;
  }

  void join(AnnotatedLine &A, const AnnotatedLine &B) {
    assert(!A.Last->Next);
    assert(!B.First->Previous);
    if (B.Affected)
      A.Affected = true;
    A.Last->Next = B.First;
    B.First->Previous = A.Last;
    B.First->CanBreakBefore = true;
    unsigned LengthA = A.Last->TotalLength + B.First->SpacesRequiredBefore;
    for (FormatToken *Tok = B.First; Tok; Tok = Tok->Next) {
      Tok->TotalLength += LengthA;
      A.Last = Tok;
    }
  }

  unsigned getColumnLimit(bool InPPDirective) const {
    // In preprocessor directives reserve two chars for trailing " \"
    return Style.ColumnLimit - (InPPDirective ? 2 : 0);
  }

  struct CompareLineStatePointers {
    bool operator()(LineState *obj1, LineState *obj2) const {
      return *obj1 < *obj2;
    }
  };

  /// \brief Analyze the entire solution space starting from \p InitialState.
  ///
  /// This implements a variant of Dijkstra's algorithm on the graph that spans
  /// the solution space (\c LineStates are the nodes). The algorithm tries to
  /// find the shortest path (the one with lowest penalty) from \p InitialState
  /// to a state where all tokens are placed. Returns the penalty.
  ///
  /// If \p DryRun is \c false, directly applies the changes.
  unsigned analyzeSolutionSpace(LineState &InitialState, bool DryRun = false) {
    std::set<LineState *, CompareLineStatePointers> Seen;

    // Increasing count of \c StateNode items we have created. This is used to
    // create a deterministic order independent of the container.
    unsigned Count = 0;
    QueueType Queue;

    // Insert start element into queue.
    StateNode *Node =
        new (Allocator.Allocate()) StateNode(InitialState, false, nullptr);
    Queue.push(QueueItem(OrderedPenalty(0, Count), Node));
    ++Count;

    unsigned Penalty = 0;

    // While not empty, take first element and follow edges.
    while (!Queue.empty()) {
      Penalty = Queue.top().first.first;
      StateNode *Node = Queue.top().second;
      if (!Node->State.NextToken) {
        DEBUG(llvm::dbgs() << "\n---\nPenalty for line: " << Penalty << "\n");
        break;
      }
      Queue.pop();

      // Cut off the analysis of certain solutions if the analysis gets too
      // complex. See description of IgnoreStackForComparison.
      if (Count > 10000)
        Node->State.IgnoreStackForComparison = true;

      if (!Seen.insert(&Node->State).second)
        // State already examined with lower penalty.
        continue;

      FormatDecision LastFormat = Node->State.NextToken->Decision;
      if (LastFormat == FD_Unformatted || LastFormat == FD_Continue)
        addNextStateToQueue(Penalty, Node, /*NewLine=*/false, &Count, &Queue);
      if (LastFormat == FD_Unformatted || LastFormat == FD_Break)
        addNextStateToQueue(Penalty, Node, /*NewLine=*/true, &Count, &Queue);
    }

    if (Queue.empty()) {
      // We were unable to find a solution, do nothing.
      // FIXME: Add diagnostic?
      DEBUG(llvm::dbgs() << "Could not find a solution.\n");
      return 0;
    }

    // Reconstruct the solution.
    if (!DryRun)
      reconstructPath(InitialState, Queue.top().second);

    DEBUG(llvm::dbgs() << "Total number of analyzed states: " << Count << "\n");
    DEBUG(llvm::dbgs() << "---\n");

    return Penalty;
  }

  void reconstructPath(LineState &State, StateNode *Current) {
    std::deque<StateNode *> Path;
    // We do not need a break before the initial token.
    while (Current->Previous) {
      Path.push_front(Current);
      Current = Current->Previous;
    }
    for (std::deque<StateNode *>::iterator I = Path.begin(), E = Path.end();
         I != E; ++I) {
      unsigned Penalty = 0;
      formatChildren(State, (*I)->NewLine, /*DryRun=*/false, Penalty);
      Penalty += Indenter->addTokenToState(State, (*I)->NewLine, false);

      DEBUG({
        if ((*I)->NewLine) {
          llvm::dbgs() << "Penalty for placing "
                       << (*I)->Previous->State.NextToken->Tok.getName() << ": "
                       << Penalty << "\n";
        }
      });
    }
  }

  /// \brief Add the following state to the analysis queue \c Queue.
  ///
  /// Assume the current state is \p PreviousNode and has been reached with a
  /// penalty of \p Penalty. Insert a line break if \p NewLine is \c true.
  void addNextStateToQueue(unsigned Penalty, StateNode *PreviousNode,
                           bool NewLine, unsigned *Count, QueueType *Queue) {
    if (NewLine && !Indenter->canBreak(PreviousNode->State))
      return;
    if (!NewLine && Indenter->mustBreak(PreviousNode->State))
      return;

    StateNode *Node = new (Allocator.Allocate())
        StateNode(PreviousNode->State, NewLine, PreviousNode);
    if (!formatChildren(Node->State, NewLine, /*DryRun=*/true, Penalty))
      return;

    Penalty += Indenter->addTokenToState(Node->State, NewLine, true);

    Queue->push(QueueItem(OrderedPenalty(Penalty, *Count), Node));
    ++(*Count);
  }

  /// \brief If the \p State's next token is an r_brace closing a nested block,
  /// format the nested block before it.
  ///
  /// Returns \c true if all children could be placed successfully and adapts
  /// \p Penalty as well as \p State. If \p DryRun is false, also directly
  /// creates changes using \c Whitespaces.
  ///
  /// The crucial idea here is that children always get formatted upon
  /// encountering the closing brace right after the nested block. Now, if we
  /// are currently trying to keep the "}" on the same line (i.e. \p NewLine is
  /// \c false), the entire block has to be kept on the same line (which is only
  /// possible if it fits on the line, only contains a single statement, etc.
  ///
  /// If \p NewLine is true, we format the nested block on separate lines, i.e.
  /// break after the "{", format all lines with correct indentation and the put
  /// the closing "}" on yet another new line.
  ///
  /// This enables us to keep the simple structure of the
  /// \c UnwrappedLineFormatter, where we only have two options for each token:
  /// break or don't break.
  bool formatChildren(LineState &State, bool NewLine, bool DryRun,
                      unsigned &Penalty) {
    FormatToken &Previous = *State.NextToken->Previous;
    const FormatToken *LBrace = State.NextToken->getPreviousNonComment();
    if (!LBrace || LBrace->isNot(tok::l_brace) ||
        LBrace->BlockKind != BK_Block || Previous.Children.size() == 0)
      // The previous token does not open a block. Nothing to do. We don't
      // assert so that we can simply call this function for all tokens.
      return true;

    if (NewLine) {
      int AdditionalIndent =
          State.FirstIndent - State.Line->Level * Style.IndentWidth;
      if (State.Stack.size() < 2 ||
          !State.Stack[State.Stack.size() - 2].JSFunctionInlined) {
        AdditionalIndent = State.Stack.back().Indent -
                           Previous.Children[0]->Level * Style.IndentWidth;
      }

      Penalty += format(Previous.Children, DryRun, AdditionalIndent,
                        /*FixBadIndentation=*/true);
      return true;
    }

    if (Previous.Children[0]->First->MustBreakBefore)
      return false;

    // Cannot merge multiple statements into a single line.
    if (Previous.Children.size() > 1)
      return false;

    // Cannot merge into one line if this line ends on a comment.
    if (Previous.is(tok::comment))
      return false;

    // We can't put the closing "}" on a line with a trailing comment.
    if (Previous.Children[0]->Last->isTrailingComment())
      return false;

    // If the child line exceeds the column limit, we wouldn't want to merge it.
    // We add +2 for the trailing " }".
    if (Style.ColumnLimit > 0 &&
        Previous.Children[0]->Last->TotalLength + State.Column + 2 >
            Style.ColumnLimit)
      return false;

    if (!DryRun) {
      Whitespaces->replaceWhitespace(
          *Previous.Children[0]->First,
          /*Newlines=*/0, /*IndentLevel=*/0, /*Spaces=*/1,
          /*StartOfTokenColumn=*/State.Column, State.Line->InPPDirective);
    }
    Penalty += format(*Previous.Children[0], State.Column + 1, DryRun);

    State.Column += 1 + Previous.Children[0]->Last->TotalLength;
    return true;
  }

  ContinuationIndenter *Indenter;
  WhitespaceManager *Whitespaces;
  FormatStyle Style;
  LineJoiner Joiner;

  llvm::SpecificBumpPtrAllocator<StateNode> Allocator;

  // Cache to store the penalty of formatting a vector of AnnotatedLines
  // starting from a specific additional offset. Improves performance if there
  // are many nested blocks.
  std::map<std::pair<const SmallVectorImpl<AnnotatedLine *> *, unsigned>,
           unsigned> PenaltyCache;
};

class FormatTokenLexer {
public:
  FormatTokenLexer(SourceManager &SourceMgr, FileID ID, FormatStyle &Style,
                   encoding::Encoding Encoding)
      : FormatTok(nullptr), IsFirstToken(true), GreaterStashed(false),
        Column(0), TrailingWhitespace(0),
        SourceMgr(SourceMgr), ID(ID), Style(Style),
        IdentTable(getFormattingLangOpts(Style)), Encoding(Encoding),
        FirstInLineIndex(0), FormattingDisabled(false) {
    Lex.reset(new Lexer(ID, SourceMgr.getBuffer(ID), SourceMgr,
                        getFormattingLangOpts(Style)));
    Lex->SetKeepWhitespaceMode(true);

    for (const std::string &ForEachMacro : Style.ForEachMacros)
      ForEachMacros.push_back(&IdentTable.get(ForEachMacro));
    std::sort(ForEachMacros.begin(), ForEachMacros.end());
  }

  ArrayRef<FormatToken *> lex() {
    assert(Tokens.empty());
    assert(FirstInLineIndex == 0);
    do {
      Tokens.push_back(getNextToken());
      tryMergePreviousTokens();
      if (Tokens.back()->NewlinesBefore > 0)
        FirstInLineIndex = Tokens.size() - 1;
    } while (Tokens.back()->Tok.isNot(tok::eof));
    return Tokens;
  }

  IdentifierTable &getIdentTable() { return IdentTable; }

private:
  void tryMergePreviousTokens() {
    if (tryMerge_TMacro())
      return;
    if (tryMergeConflictMarkers())
      return;

    if (Style.Language == FormatStyle::LK_JavaScript) {
      if (tryMergeJSRegexLiteral())
        return;
      if (tryMergeEscapeSequence())
        return;

      static tok::TokenKind JSIdentity[] = { tok::equalequal, tok::equal };
      static tok::TokenKind JSNotIdentity[] = { tok::exclaimequal, tok::equal };
      static tok::TokenKind JSShiftEqual[] = { tok::greater, tok::greater,
                                               tok::greaterequal };
      static tok::TokenKind JSRightArrow[] = { tok::equal, tok::greater };
      // FIXME: We probably need to change token type to mimic operator with the
      // correct priority.
      if (tryMergeTokens(JSIdentity))
        return;
      if (tryMergeTokens(JSNotIdentity))
        return;
      if (tryMergeTokens(JSShiftEqual))
        return;
      if (tryMergeTokens(JSRightArrow))
        return;
    }
  }

  bool tryMergeTokens(ArrayRef<tok::TokenKind> Kinds) {
    if (Tokens.size() < Kinds.size())
      return false;

    SmallVectorImpl<FormatToken *>::const_iterator First =
        Tokens.end() - Kinds.size();
    if (!First[0]->is(Kinds[0]))
      return false;
    unsigned AddLength = 0;
    for (unsigned i = 1; i < Kinds.size(); ++i) {
      if (!First[i]->is(Kinds[i]) || First[i]->WhitespaceRange.getBegin() !=
                                         First[i]->WhitespaceRange.getEnd())
        return false;
      AddLength += First[i]->TokenText.size();
    }
    Tokens.resize(Tokens.size() - Kinds.size() + 1);
    First[0]->TokenText = StringRef(First[0]->TokenText.data(),
                                    First[0]->TokenText.size() + AddLength);
    First[0]->ColumnWidth += AddLength;
    return true;
  }

  // Tries to merge an escape sequence, i.e. a "\\" and the following
  // character. Use e.g. inside JavaScript regex literals.
  bool tryMergeEscapeSequence() {
    if (Tokens.size() < 2)
      return false;
    FormatToken *Previous = Tokens[Tokens.size() - 2];
    if (Previous->isNot(tok::unknown) || Previous->TokenText != "\\")
      return false;
    ++Previous->ColumnWidth;
    StringRef Text = Previous->TokenText;
    Previous->TokenText = StringRef(Text.data(), Text.size() + 1);
    resetLexer(SourceMgr.getFileOffset(Tokens.back()->Tok.getLocation()) + 1);
    Tokens.resize(Tokens.size() - 1);
    Column = Previous->OriginalColumn + Previous->ColumnWidth;
    return true;
  }

  // Try to determine whether the current token ends a JavaScript regex literal.
  // We heuristically assume that this is a regex literal if we find two
  // unescaped slashes on a line and the token before the first slash is one of
  // "(;,{}![:?", a binary operator or 'return', as those cannot be followed by
  // a division.
  bool tryMergeJSRegexLiteral() {
    if (Tokens.size() < 2)
      return false;
    // If a regex literal ends in "\//", this gets represented by an unknown
    // token "\" and a comment.
    bool MightEndWithEscapedSlash =
        Tokens.back()->is(tok::comment) &&
        Tokens.back()->TokenText.startswith("//") &&
        Tokens[Tokens.size() - 2]->TokenText == "\\";
    if (!MightEndWithEscapedSlash &&
        (Tokens.back()->isNot(tok::slash) ||
         (Tokens[Tokens.size() - 2]->is(tok::unknown) &&
          Tokens[Tokens.size() - 2]->TokenText == "\\")))
      return false;
    unsigned TokenCount = 0;
    unsigned LastColumn = Tokens.back()->OriginalColumn;
    for (auto I = Tokens.rbegin() + 1, E = Tokens.rend(); I != E; ++I) {
      ++TokenCount;
      if (I[0]->is(tok::slash) && I + 1 != E &&
          (I[1]->isOneOf(tok::l_paren, tok::semi, tok::l_brace, tok::r_brace,
                         tok::exclaim, tok::l_square, tok::colon, tok::comma,
                         tok::question, tok::kw_return) ||
           I[1]->isBinaryOperator())) {
        if (MightEndWithEscapedSlash) {
          // This regex literal ends in '\//'. Skip past the '//' of the last
          // token and re-start lexing from there.
          SourceLocation Loc = Tokens.back()->Tok.getLocation();
          resetLexer(SourceMgr.getFileOffset(Loc) + 2);
        }
        Tokens.resize(Tokens.size() - TokenCount);
        Tokens.back()->Tok.setKind(tok::unknown);
        Tokens.back()->Type = TT_RegexLiteral;
        Tokens.back()->ColumnWidth += LastColumn - I[0]->OriginalColumn;
        return true;
      }

      // There can't be a newline inside a regex literal.
      if (I[0]->NewlinesBefore > 0)
        return false;
    }
    return false;
  }

  bool tryMerge_TMacro() {
    if (Tokens.size() < 4)
      return false;
    FormatToken *Last = Tokens.back();
    if (!Last->is(tok::r_paren))
      return false;

    FormatToken *String = Tokens[Tokens.size() - 2];
    if (!String->is(tok::string_literal) || String->IsMultiline)
      return false;

    if (!Tokens[Tokens.size() - 3]->is(tok::l_paren))
      return false;

    FormatToken *Macro = Tokens[Tokens.size() - 4];
    if (Macro->TokenText != "_T")
      return false;

    const char *Start = Macro->TokenText.data();
    const char *End = Last->TokenText.data() + Last->TokenText.size();
    String->TokenText = StringRef(Start, End - Start);
    String->IsFirst = Macro->IsFirst;
    String->LastNewlineOffset = Macro->LastNewlineOffset;
    String->WhitespaceRange = Macro->WhitespaceRange;
    String->OriginalColumn = Macro->OriginalColumn;
    String->ColumnWidth = encoding::columnWidthWithTabs(
        String->TokenText, String->OriginalColumn, Style.TabWidth, Encoding);

    Tokens.pop_back();
    Tokens.pop_back();
    Tokens.pop_back();
    Tokens.back() = String;
    return true;
  }

  bool tryMergeConflictMarkers() {
    if (Tokens.back()->NewlinesBefore == 0 && Tokens.back()->isNot(tok::eof))
      return false;

    // Conflict lines look like:
    // <marker> <text from the vcs>
    // For example:
    // >>>>>>> /file/in/file/system at revision 1234
    //
    // We merge all tokens in a line that starts with a conflict marker
    // into a single token with a special token type that the unwrapped line
    // parser will use to correctly rebuild the underlying code.

    FileID ID;
    // Get the position of the first token in the line.
    unsigned FirstInLineOffset;
    std::tie(ID, FirstInLineOffset) = SourceMgr.getDecomposedLoc(
        Tokens[FirstInLineIndex]->getStartOfNonWhitespace());
    StringRef Buffer = SourceMgr.getBuffer(ID)->getBuffer();
    // Calculate the offset of the start of the current line.
    auto LineOffset = Buffer.rfind('\n', FirstInLineOffset);
    if (LineOffset == StringRef::npos) {
      LineOffset = 0;
    } else {
      ++LineOffset;
    }

    auto FirstSpace = Buffer.find_first_of(" \n", LineOffset);
    StringRef LineStart;
    if (FirstSpace == StringRef::npos) {
      LineStart = Buffer.substr(LineOffset);
    } else {
      LineStart = Buffer.substr(LineOffset, FirstSpace - LineOffset);
    }

    TokenType Type = TT_Unknown;
    if (LineStart == "<<<<<<<" || LineStart == ">>>>") {
      Type = TT_ConflictStart;
    } else if (LineStart == "|||||||" || LineStart == "=======" ||
               LineStart == "====") {
      Type = TT_ConflictAlternative;
    } else if (LineStart == ">>>>>>>" || LineStart == "<<<<") {
      Type = TT_ConflictEnd;
    }

    if (Type != TT_Unknown) {
      FormatToken *Next = Tokens.back();

      Tokens.resize(FirstInLineIndex + 1);
      // We do not need to build a complete token here, as we will skip it
      // during parsing anyway (as we must not touch whitespace around conflict
      // markers).
      Tokens.back()->Type = Type;
      Tokens.back()->Tok.setKind(tok::kw___unknown_anytype);

      Tokens.push_back(Next);
      return true;
    }

    return false;
  }

  FormatToken *getNextToken() {
    if (GreaterStashed) {
      // Create a synthesized second '>' token.
      // FIXME: Increment Column and set OriginalColumn.
      Token Greater = FormatTok->Tok;
      FormatTok = new (Allocator.Allocate()) FormatToken;
      FormatTok->Tok = Greater;
      SourceLocation GreaterLocation =
          FormatTok->Tok.getLocation().getLocWithOffset(1);
      FormatTok->WhitespaceRange =
          SourceRange(GreaterLocation, GreaterLocation);
      FormatTok->TokenText = ">";
      FormatTok->ColumnWidth = 1;
      GreaterStashed = false;
      return FormatTok;
    }

    FormatTok = new (Allocator.Allocate()) FormatToken;
    readRawToken(*FormatTok);
    SourceLocation WhitespaceStart =
        FormatTok->Tok.getLocation().getLocWithOffset(-TrailingWhitespace);
    FormatTok->IsFirst = IsFirstToken;
    IsFirstToken = false;

    // Consume and record whitespace until we find a significant token.
    unsigned WhitespaceLength = TrailingWhitespace;
    while (FormatTok->Tok.is(tok::unknown)) {
      for (int i = 0, e = FormatTok->TokenText.size(); i != e; ++i) {
        switch (FormatTok->TokenText[i]) {
        case '\n':
          ++FormatTok->NewlinesBefore;
          // FIXME: This is technically incorrect, as it could also
          // be a literal backslash at the end of the line.
          if (i == 0 || (FormatTok->TokenText[i - 1] != '\\' &&
                         (FormatTok->TokenText[i - 1] != '\r' || i == 1 ||
                          FormatTok->TokenText[i - 2] != '\\')))
            FormatTok->HasUnescapedNewline = true;
          FormatTok->LastNewlineOffset = WhitespaceLength + i + 1;
          Column = 0;
          break;
        case '\r':
        case '\f':
        case '\v':
          Column = 0;
          break;
        case ' ':
          ++Column;
          break;
        case '\t':
          Column += Style.TabWidth - Column % Style.TabWidth;
          break;
        case '\\':
          ++Column;
          if (i + 1 == e || (FormatTok->TokenText[i + 1] != '\r' &&
                             FormatTok->TokenText[i + 1] != '\n'))
            FormatTok->Type = TT_ImplicitStringLiteral;
          break;
        default:
          FormatTok->Type = TT_ImplicitStringLiteral;
          ++Column;
          break;
        }
      }

      if (FormatTok->Type == TT_ImplicitStringLiteral)
        break;
      WhitespaceLength += FormatTok->Tok.getLength();

      readRawToken(*FormatTok);
    }

    // In case the token starts with escaped newlines, we want to
    // take them into account as whitespace - this pattern is quite frequent
    // in macro definitions.
    // FIXME: Add a more explicit test.
    while (FormatTok->TokenText.size() > 1 && FormatTok->TokenText[0] == '\\' &&
           FormatTok->TokenText[1] == '\n') {
      ++FormatTok->NewlinesBefore;
      WhitespaceLength += 2;
      Column = 0;
      FormatTok->TokenText = FormatTok->TokenText.substr(2);
    }

    FormatTok->WhitespaceRange = SourceRange(
        WhitespaceStart, WhitespaceStart.getLocWithOffset(WhitespaceLength));

    FormatTok->OriginalColumn = Column;

    TrailingWhitespace = 0;
    if (FormatTok->Tok.is(tok::comment)) {
      // FIXME: Add the trimmed whitespace to Column.
      StringRef UntrimmedText = FormatTok->TokenText;
      FormatTok->TokenText = FormatTok->TokenText.rtrim(" \t\v\f");
      TrailingWhitespace = UntrimmedText.size() - FormatTok->TokenText.size();
    } else if (FormatTok->Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = IdentTable.get(FormatTok->TokenText);
      FormatTok->Tok.setIdentifierInfo(&Info);
      FormatTok->Tok.setKind(Info.getTokenID());
    } else if (FormatTok->Tok.is(tok::greatergreater)) {
      FormatTok->Tok.setKind(tok::greater);
      FormatTok->TokenText = FormatTok->TokenText.substr(0, 1);
      GreaterStashed = true;
    }

    // Now FormatTok is the next non-whitespace token.

    StringRef Text = FormatTok->TokenText;
    size_t FirstNewlinePos = Text.find('\n');
    if (FirstNewlinePos == StringRef::npos) {
      // FIXME: ColumnWidth actually depends on the start column, we need to
      // take this into account when the token is moved.
      FormatTok->ColumnWidth =
          encoding::columnWidthWithTabs(Text, Column, Style.TabWidth, Encoding);
      Column += FormatTok->ColumnWidth;
    } else {
      FormatTok->IsMultiline = true;
      // FIXME: ColumnWidth actually depends on the start column, we need to
      // take this into account when the token is moved.
      FormatTok->ColumnWidth = encoding::columnWidthWithTabs(
          Text.substr(0, FirstNewlinePos), Column, Style.TabWidth, Encoding);

      // The last line of the token always starts in column 0.
      // Thus, the length can be precomputed even in the presence of tabs.
      FormatTok->LastLineColumnWidth = encoding::columnWidthWithTabs(
          Text.substr(Text.find_last_of('\n') + 1), 0, Style.TabWidth,
          Encoding);
      Column = FormatTok->LastLineColumnWidth;
    }

    FormatTok->IsForEachMacro =
        std::binary_search(ForEachMacros.begin(), ForEachMacros.end(),
                           FormatTok->Tok.getIdentifierInfo());

    return FormatTok;
  }

  FormatToken *FormatTok;
  bool IsFirstToken;
  bool GreaterStashed;
  unsigned Column;
  unsigned TrailingWhitespace;
  std::unique_ptr<Lexer> Lex;
  SourceManager &SourceMgr;
  FileID ID;
  FormatStyle &Style;
  IdentifierTable IdentTable;
  encoding::Encoding Encoding;
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  // Index (in 'Tokens') of the last token that starts a new line.
  unsigned FirstInLineIndex;
  SmallVector<FormatToken *, 16> Tokens;
  SmallVector<IdentifierInfo *, 8> ForEachMacros;

  bool FormattingDisabled;

  void readRawToken(FormatToken &Tok) {
    Lex->LexFromRawLexer(Tok.Tok);
    Tok.TokenText = StringRef(SourceMgr.getCharacterData(Tok.Tok.getLocation()),
                              Tok.Tok.getLength());
    // For formatting, treat unterminated string literals like normal string
    // literals.
    if (Tok.is(tok::unknown)) {
      if (!Tok.TokenText.empty() && Tok.TokenText[0] == '"') {
        Tok.Tok.setKind(tok::string_literal);
        Tok.IsUnterminatedLiteral = true;
      } else if (Style.Language == FormatStyle::LK_JavaScript &&
                 Tok.TokenText == "''") {
        Tok.Tok.setKind(tok::char_constant);
      }
    }

    if (Tok.is(tok::comment) && (Tok.TokenText == "// clang-format on" ||
                                 Tok.TokenText == "/* clang-format on */")) {
      FormattingDisabled = false;
    }

    Tok.Finalized = FormattingDisabled;

    if (Tok.is(tok::comment) && (Tok.TokenText == "// clang-format off" ||
                                 Tok.TokenText == "/* clang-format off */")) {
      FormattingDisabled = true;
    }
  }

  void resetLexer(unsigned Offset) {
    StringRef Buffer = SourceMgr.getBufferData(ID);
    Lex.reset(new Lexer(SourceMgr.getLocForStartOfFile(ID),
                        getFormattingLangOpts(Style), Buffer.begin(),
                        Buffer.begin() + Offset, Buffer.end()));
    Lex->SetKeepWhitespaceMode(true);
  }
};

static StringRef getLanguageName(FormatStyle::LanguageKind Language) {
  switch (Language) {
  case FormatStyle::LK_Cpp:
    return "C++";
  case FormatStyle::LK_Java:
    return "Java";
  case FormatStyle::LK_JavaScript:
    return "JavaScript";
  case FormatStyle::LK_Proto:
    return "Proto";
  default:
    return "Unknown";
  }
}

class Formatter : public UnwrappedLineConsumer {
public:
  Formatter(const FormatStyle &Style, SourceManager &SourceMgr, FileID ID,
            ArrayRef<CharSourceRange> Ranges)
      : Style(Style), ID(ID), SourceMgr(SourceMgr),
        Whitespaces(SourceMgr, Style,
                    inputUsesCRLF(SourceMgr.getBufferData(ID))),
        Ranges(Ranges.begin(), Ranges.end()), UnwrappedLines(1),
        Encoding(encoding::detectEncoding(SourceMgr.getBufferData(ID))) {
    DEBUG(llvm::dbgs() << "File encoding: "
                       << (Encoding == encoding::Encoding_UTF8 ? "UTF8"
                                                               : "unknown")
                       << "\n");
    DEBUG(llvm::dbgs() << "Language: " << getLanguageName(Style.Language)
                       << "\n");
  }

  tooling::Replacements format() {
    tooling::Replacements Result;
    FormatTokenLexer Tokens(SourceMgr, ID, Style, Encoding);

    UnwrappedLineParser Parser(Style, Tokens.lex(), *this);
    bool StructuralError = Parser.parse();
    assert(UnwrappedLines.rbegin()->empty());
    for (unsigned Run = 0, RunE = UnwrappedLines.size(); Run + 1 != RunE;
         ++Run) {
      DEBUG(llvm::dbgs() << "Run " << Run << "...\n");
      SmallVector<AnnotatedLine *, 16> AnnotatedLines;
      for (unsigned i = 0, e = UnwrappedLines[Run].size(); i != e; ++i) {
        AnnotatedLines.push_back(new AnnotatedLine(UnwrappedLines[Run][i]));
      }
      tooling::Replacements RunResult =
          format(AnnotatedLines, StructuralError, Tokens);
      DEBUG({
        llvm::dbgs() << "Replacements for run " << Run << ":\n";
        for (tooling::Replacements::iterator I = RunResult.begin(),
                                             E = RunResult.end();
             I != E; ++I) {
          llvm::dbgs() << I->toString() << "\n";
        }
      });
      for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
        delete AnnotatedLines[i];
      }
      Result.insert(RunResult.begin(), RunResult.end());
      Whitespaces.reset();
    }
    return Result;
  }

  tooling::Replacements format(SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
                               bool StructuralError, FormatTokenLexer &Tokens) {
    TokenAnnotator Annotator(Style, Tokens.getIdentTable().get("in"));
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.annotate(*AnnotatedLines[i]);
    }
    deriveLocalStyle(AnnotatedLines);
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.calculateFormattingInformation(*AnnotatedLines[i]);
    }
    computeAffectedLines(AnnotatedLines.begin(), AnnotatedLines.end());

    Annotator.setCommentLineLevels(AnnotatedLines);
    ContinuationIndenter Indenter(Style, SourceMgr, Whitespaces, Encoding,
                                  BinPackInconclusiveFunctions);
    UnwrappedLineFormatter Formatter(&Indenter, &Whitespaces, Style);
    Formatter.format(AnnotatedLines, /*DryRun=*/false);
    return Whitespaces.generateReplacements();
  }

private:
  // Determines which lines are affected by the SourceRanges given as input.
  // Returns \c true if at least one line between I and E or one of their
  // children is affected.
  bool computeAffectedLines(SmallVectorImpl<AnnotatedLine *>::iterator I,
                            SmallVectorImpl<AnnotatedLine *>::iterator E) {
    bool SomeLineAffected = false;
    const AnnotatedLine *PreviousLine = nullptr;
    while (I != E) {
      AnnotatedLine *Line = *I;
      Line->LeadingEmptyLinesAffected = affectsLeadingEmptyLines(*Line->First);

      // If a line is part of a preprocessor directive, it needs to be formatted
      // if any token within the directive is affected.
      if (Line->InPPDirective) {
        FormatToken *Last = Line->Last;
        SmallVectorImpl<AnnotatedLine *>::iterator PPEnd = I + 1;
        while (PPEnd != E && !(*PPEnd)->First->HasUnescapedNewline) {
          Last = (*PPEnd)->Last;
          ++PPEnd;
        }

        if (affectsTokenRange(*Line->First, *Last,
                              /*IncludeLeadingNewlines=*/false)) {
          SomeLineAffected = true;
          markAllAsAffected(I, PPEnd);
        }
        I = PPEnd;
        continue;
      }

      if (nonPPLineAffected(Line, PreviousLine))
        SomeLineAffected = true;

      PreviousLine = Line;
      ++I;
    }
    return SomeLineAffected;
  }

  // Determines whether 'Line' is affected by the SourceRanges given as input.
  // Returns \c true if line or one if its children is affected.
  bool nonPPLineAffected(AnnotatedLine *Line,
                         const AnnotatedLine *PreviousLine) {
    bool SomeLineAffected = false;
    Line->ChildrenAffected =
        computeAffectedLines(Line->Children.begin(), Line->Children.end());
    if (Line->ChildrenAffected)
      SomeLineAffected = true;

    // Stores whether one of the line's tokens is directly affected.
    bool SomeTokenAffected = false;
    // Stores whether we need to look at the leading newlines of the next token
    // in order to determine whether it was affected.
    bool IncludeLeadingNewlines = false;

    // Stores whether the first child line of any of this line's tokens is
    // affected.
    bool SomeFirstChildAffected = false;

    for (FormatToken *Tok = Line->First; Tok; Tok = Tok->Next) {
      // Determine whether 'Tok' was affected.
      if (affectsTokenRange(*Tok, *Tok, IncludeLeadingNewlines))
        SomeTokenAffected = true;

      // Determine whether the first child of 'Tok' was affected.
      if (!Tok->Children.empty() && Tok->Children.front()->Affected)
        SomeFirstChildAffected = true;

      IncludeLeadingNewlines = Tok->Children.empty();
    }

    // Was this line moved, i.e. has it previously been on the same line as an
    // affected line?
    bool LineMoved = PreviousLine && PreviousLine->Affected &&
                     Line->First->NewlinesBefore == 0;

    bool IsContinuedComment =
        Line->First->is(tok::comment) && Line->First->Next == nullptr &&
        Line->First->NewlinesBefore < 2 && PreviousLine &&
        PreviousLine->Affected && PreviousLine->Last->is(tok::comment);

    if (SomeTokenAffected || SomeFirstChildAffected || LineMoved ||
        IsContinuedComment) {
      Line->Affected = true;
      SomeLineAffected = true;
    }
    return SomeLineAffected;
  }

  // Marks all lines between I and E as well as all their children as affected.
  void markAllAsAffected(SmallVectorImpl<AnnotatedLine *>::iterator I,
                         SmallVectorImpl<AnnotatedLine *>::iterator E) {
    while (I != E) {
      (*I)->Affected = true;
      markAllAsAffected((*I)->Children.begin(), (*I)->Children.end());
      ++I;
    }
  }

  // Returns true if the range from 'First' to 'Last' intersects with one of the
  // input ranges.
  bool affectsTokenRange(const FormatToken &First, const FormatToken &Last,
                         bool IncludeLeadingNewlines) {
    SourceLocation Start = First.WhitespaceRange.getBegin();
    if (!IncludeLeadingNewlines)
      Start = Start.getLocWithOffset(First.LastNewlineOffset);
    SourceLocation End = Last.getStartOfNonWhitespace();
    End = End.getLocWithOffset(Last.TokenText.size());
    CharSourceRange Range = CharSourceRange::getCharRange(Start, End);
    return affectsCharSourceRange(Range);
  }

  // Returns true if one of the input ranges intersect the leading empty lines
  // before 'Tok'.
  bool affectsLeadingEmptyLines(const FormatToken &Tok) {
    CharSourceRange EmptyLineRange = CharSourceRange::getCharRange(
        Tok.WhitespaceRange.getBegin(),
        Tok.WhitespaceRange.getBegin().getLocWithOffset(Tok.LastNewlineOffset));
    return affectsCharSourceRange(EmptyLineRange);
  }

  // Returns true if 'Range' intersects with one of the input ranges.
  bool affectsCharSourceRange(const CharSourceRange &Range) {
    for (SmallVectorImpl<CharSourceRange>::const_iterator I = Ranges.begin(),
                                                          E = Ranges.end();
         I != E; ++I) {
      if (!SourceMgr.isBeforeInTranslationUnit(Range.getEnd(), I->getBegin()) &&
          !SourceMgr.isBeforeInTranslationUnit(I->getEnd(), Range.getBegin()))
        return true;
    }
    return false;
  }

  static bool inputUsesCRLF(StringRef Text) {
    return Text.count('\r') * 2 > Text.count('\n');
  }

  void
  deriveLocalStyle(const SmallVectorImpl<AnnotatedLine *> &AnnotatedLines) {
    unsigned CountBoundToVariable = 0;
    unsigned CountBoundToType = 0;
    bool HasCpp03IncompatibleFormat = false;
    bool HasBinPackedFunction = false;
    bool HasOnePerLineFunction = false;
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      if (!AnnotatedLines[i]->First->Next)
        continue;
      FormatToken *Tok = AnnotatedLines[i]->First->Next;
      while (Tok->Next) {
        if (Tok->Type == TT_PointerOrReference) {
          bool SpacesBefore =
              Tok->WhitespaceRange.getBegin() != Tok->WhitespaceRange.getEnd();
          bool SpacesAfter = Tok->Next->WhitespaceRange.getBegin() !=
                             Tok->Next->WhitespaceRange.getEnd();
          if (SpacesBefore && !SpacesAfter)
            ++CountBoundToVariable;
          else if (!SpacesBefore && SpacesAfter)
            ++CountBoundToType;
        }

        if (Tok->WhitespaceRange.getBegin() == Tok->WhitespaceRange.getEnd()) {
          if (Tok->is(tok::coloncolon) &&
              Tok->Previous->Type == TT_TemplateOpener)
            HasCpp03IncompatibleFormat = true;
          if (Tok->Type == TT_TemplateCloser &&
              Tok->Previous->Type == TT_TemplateCloser)
            HasCpp03IncompatibleFormat = true;
        }

        if (Tok->PackingKind == PPK_BinPacked)
          HasBinPackedFunction = true;
        if (Tok->PackingKind == PPK_OnePerLine)
          HasOnePerLineFunction = true;

        Tok = Tok->Next;
      }
    }
    if (Style.DerivePointerAlignment) {
      if (CountBoundToType > CountBoundToVariable)
        Style.PointerAlignment = FormatStyle::PAS_Left;
      else if (CountBoundToType < CountBoundToVariable)
        Style.PointerAlignment = FormatStyle::PAS_Right;
    }
    if (Style.Standard == FormatStyle::LS_Auto) {
      Style.Standard = HasCpp03IncompatibleFormat ? FormatStyle::LS_Cpp11
                                                  : FormatStyle::LS_Cpp03;
    }
    BinPackInconclusiveFunctions =
        HasBinPackedFunction || !HasOnePerLineFunction;
  }

  void consumeUnwrappedLine(const UnwrappedLine &TheLine) override {
    assert(!UnwrappedLines.empty());
    UnwrappedLines.back().push_back(TheLine);
  }

  void finishRun() override {
    UnwrappedLines.push_back(SmallVector<UnwrappedLine, 16>());
  }

  FormatStyle Style;
  FileID ID;
  SourceManager &SourceMgr;
  WhitespaceManager Whitespaces;
  SmallVector<CharSourceRange, 8> Ranges;
  SmallVector<SmallVector<UnwrappedLine, 16>, 2> UnwrappedLines;

  encoding::Encoding Encoding;
  bool BinPackInconclusiveFunctions;
};

} // end anonymous namespace

tooling::Replacements reformat(const FormatStyle &Style, Lexer &Lex,
                               SourceManager &SourceMgr,
                               ArrayRef<CharSourceRange> Ranges) {
  if (Style.DisableFormat)
    return tooling::Replacements();
  return reformat(Style, SourceMgr,
                  SourceMgr.getFileID(Lex.getSourceLocation()), Ranges);
}

tooling::Replacements reformat(const FormatStyle &Style,
                               SourceManager &SourceMgr, FileID ID,
                               ArrayRef<CharSourceRange> Ranges) {
  if (Style.DisableFormat)
    return tooling::Replacements();
  Formatter formatter(Style, SourceMgr, ID, Ranges);
  return formatter.format();
}

tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
                               ArrayRef<tooling::Range> Ranges,
                               StringRef FileName) {
  if (Style.DisableFormat)
    return tooling::Replacements();

  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager SourceMgr(Diagnostics, Files);
  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Code, FileName);
  const clang::FileEntry *Entry =
      Files.getVirtualFile(FileName, Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(Entry, std::move(Buf));
  FileID ID =
      SourceMgr.createFileID(Entry, SourceLocation(), clang::SrcMgr::C_User);
  SourceLocation StartOfFile = SourceMgr.getLocForStartOfFile(ID);
  std::vector<CharSourceRange> CharRanges;
  for (const tooling::Range &Range : Ranges) {
    SourceLocation Start = StartOfFile.getLocWithOffset(Range.getOffset());
    SourceLocation End = Start.getLocWithOffset(Range.getLength());
    CharRanges.push_back(CharSourceRange::getCharRange(Start, End));
  }
  return reformat(Style, SourceMgr, ID, CharRanges);
}

LangOptions getFormattingLangOpts(const FormatStyle &Style) {
  LangOptions LangOpts;
  LangOpts.CPlusPlus = 1;
  LangOpts.CPlusPlus11 = Style.Standard == FormatStyle::LS_Cpp03 ? 0 : 1;
  LangOpts.CPlusPlus14 = Style.Standard == FormatStyle::LS_Cpp03 ? 0 : 1;
  LangOpts.LineComment = 1;
  LangOpts.CXXOperatorNames =
      Style.Language != FormatStyle::LK_JavaScript ? 1 : 0;
  LangOpts.Bool = 1;
  LangOpts.ObjC1 = 1;
  LangOpts.ObjC2 = 1;
  return LangOpts;
}

const char *StyleOptionHelpDescription =
    "Coding style, currently supports:\n"
    "  LLVM, Google, Chromium, Mozilla, WebKit.\n"
    "Use -style=file to load style configuration from\n"
    ".clang-format file located in one of the parent\n"
    "directories of the source file (or current\n"
    "directory for stdin).\n"
    "Use -style=\"{key: value, ...}\" to set specific\n"
    "parameters, e.g.:\n"
    "  -style=\"{BasedOnStyle: llvm, IndentWidth: 8}\"";

static FormatStyle::LanguageKind getLanguageByFileName(StringRef FileName) {
  if (FileName.endswith(".java")) {
    return FormatStyle::LK_Java;
  } else if (FileName.endswith_lower(".js")) {
    return FormatStyle::LK_JavaScript;
  } else if (FileName.endswith_lower(".proto") ||
             FileName.endswith_lower(".protodevel")) {
    return FormatStyle::LK_Proto;
  }
  return FormatStyle::LK_Cpp;
}

FormatStyle getStyle(StringRef StyleName, StringRef FileName,
                     StringRef FallbackStyle) {
  FormatStyle Style = getLLVMStyle();
  Style.Language = getLanguageByFileName(FileName);
  if (!getPredefinedStyle(FallbackStyle, Style.Language, &Style)) {
    llvm::errs() << "Invalid fallback style \"" << FallbackStyle
                 << "\" using LLVM style\n";
    return Style;
  }

  if (StyleName.startswith("{")) {
    // Parse YAML/JSON style from the command line.
    if (std::error_code ec = parseConfiguration(StyleName, &Style)) {
      llvm::errs() << "Error parsing -style: " << ec.message() << ", using "
                   << FallbackStyle << " style\n";
    }
    return Style;
  }

  if (!StyleName.equals_lower("file")) {
    if (!getPredefinedStyle(StyleName, Style.Language, &Style))
      llvm::errs() << "Invalid value for -style, using " << FallbackStyle
                   << " style\n";
    return Style;
  }

  // Look for .clang-format/_clang-format file in the file's parent directories.
  SmallString<128> UnsuitableConfigFiles;
  SmallString<128> Path(FileName);
  llvm::sys::fs::make_absolute(Path);
  for (StringRef Directory = Path; !Directory.empty();
       Directory = llvm::sys::path::parent_path(Directory)) {
    if (!llvm::sys::fs::is_directory(Directory))
      continue;
    SmallString<128> ConfigFile(Directory);

    llvm::sys::path::append(ConfigFile, ".clang-format");
    DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");
    bool IsFile = false;
    // Ignore errors from is_regular_file: we only need to know if we can read
    // the file or not.
    llvm::sys::fs::is_regular_file(Twine(ConfigFile), IsFile);

    if (!IsFile) {
      // Try _clang-format too, since dotfiles are not commonly used on Windows.
      ConfigFile = Directory;
      llvm::sys::path::append(ConfigFile, "_clang-format");
      DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");
      llvm::sys::fs::is_regular_file(Twine(ConfigFile), IsFile);
    }

    if (IsFile) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
          llvm::MemoryBuffer::getFile(ConfigFile.c_str());
      if (std::error_code EC = Text.getError()) {
        llvm::errs() << EC.message() << "\n";
        break;
      }
      if (std::error_code ec =
              parseConfiguration(Text.get()->getBuffer(), &Style)) {
        if (ec == ParseError::Unsuitable) {
          if (!UnsuitableConfigFiles.empty())
            UnsuitableConfigFiles.append(", ");
          UnsuitableConfigFiles.append(ConfigFile);
          continue;
        }
        llvm::errs() << "Error reading " << ConfigFile << ": " << ec.message()
                     << "\n";
        break;
      }
      DEBUG(llvm::dbgs() << "Using configuration file " << ConfigFile << "\n");
      return Style;
    }
  }
  llvm::errs() << "Can't find usable .clang-format, using " << FallbackStyle
               << " style\n";
  if (!UnsuitableConfigFiles.empty()) {
    llvm::errs() << "Configuration file(s) do(es) not support "
                 << getLanguageName(Style.Language) << ": "
                 << UnsuitableConfigFiles << "\n";
  }
  return Style;
}

} // namespace format
} // namespace clang
