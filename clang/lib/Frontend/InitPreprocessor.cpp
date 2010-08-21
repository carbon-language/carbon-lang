//===--- InitPreprocessor.cpp - PP initialization code. ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the clang::InitializePreprocessor function.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "clang/Frontend/Utils.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Path.h"
using namespace clang;

// Append a #define line to Buf for Macro.  Macro should be of the form XXX,
// in which case we emit "#define XXX 1" or "XXX=Y z W" in which case we emit
// "#define XXX Y z W".  To get a #define with no value, use "XXX=".
static void DefineBuiltinMacro(MacroBuilder &Builder, llvm::StringRef Macro,
                               Diagnostic &Diags) {
  std::pair<llvm::StringRef, llvm::StringRef> MacroPair = Macro.split('=');
  llvm::StringRef MacroName = MacroPair.first;
  llvm::StringRef MacroBody = MacroPair.second;
  if (MacroName.size() != Macro.size()) {
    // Per GCC -D semantics, the macro ends at \n if it exists.
    llvm::StringRef::size_type End = MacroBody.find_first_of("\n\r");
    if (End != llvm::StringRef::npos)
      Diags.Report(diag::warn_fe_macro_contains_embedded_newline)
        << MacroName;
    Builder.defineMacro(MacroName, MacroBody.substr(0, End));
  } else {
    // Push "macroname 1".
    Builder.defineMacro(Macro);
  }
}

std::string clang::NormalizeDashIncludePath(llvm::StringRef File) {
  // Implicit include paths should be resolved relative to the current
  // working directory first, and then use the regular header search
  // mechanism. The proper way to handle this is to have the
  // predefines buffer located at the current working directory, but
  // it has not file entry. For now, workaround this by using an
  // absolute path if we find the file here, and otherwise letting
  // header search handle it.
  llvm::sys::Path Path(File);
  Path.makeAbsolute();
  if (!Path.exists())
    Path = File;

  return Lexer::Stringify(Path.str());
}

/// AddImplicitInclude - Add an implicit #include of the specified file to the
/// predefines buffer.
static void AddImplicitInclude(MacroBuilder &Builder, llvm::StringRef File) {
  Builder.append("#include \"" +
                 llvm::Twine(NormalizeDashIncludePath(File)) + "\"");
}

static void AddImplicitIncludeMacros(MacroBuilder &Builder,
                                     llvm::StringRef File) {
  Builder.append("#__include_macros \"" +
                 llvm::Twine(NormalizeDashIncludePath(File)) + "\"");
  // Marker token to stop the __include_macros fetch loop.
  Builder.append("##"); // ##?
}

/// AddImplicitIncludePTH - Add an implicit #include using the original file
///  used to generate a PTH cache.
static void AddImplicitIncludePTH(MacroBuilder &Builder, Preprocessor &PP,
                                  llvm::StringRef ImplicitIncludePTH) {
  PTHManager *P = PP.getPTHManager();
  // Null check 'P' in the corner case where it couldn't be created.
  const char *OriginalFile = P ? P->getOriginalSourceFile() : 0;

  if (!OriginalFile) {
    PP.getDiagnostics().Report(diag::err_fe_pth_file_has_no_source_header)
      << ImplicitIncludePTH;
    return;
  }

  AddImplicitInclude(Builder, OriginalFile);
}

/// PickFP - This is used to pick a value based on the FP semantics of the
/// specified FP model.
template <typename T>
static T PickFP(const llvm::fltSemantics *Sem, T IEEESingleVal,
                T IEEEDoubleVal, T X87DoubleExtendedVal, T PPCDoubleDoubleVal,
                T IEEEQuadVal) {
  if (Sem == (const llvm::fltSemantics*)&llvm::APFloat::IEEEsingle)
    return IEEESingleVal;
  if (Sem == (const llvm::fltSemantics*)&llvm::APFloat::IEEEdouble)
    return IEEEDoubleVal;
  if (Sem == (const llvm::fltSemantics*)&llvm::APFloat::x87DoubleExtended)
    return X87DoubleExtendedVal;
  if (Sem == (const llvm::fltSemantics*)&llvm::APFloat::PPCDoubleDouble)
    return PPCDoubleDoubleVal;
  assert(Sem == (const llvm::fltSemantics*)&llvm::APFloat::IEEEquad);
  return IEEEQuadVal;
}

static void DefineFloatMacros(MacroBuilder &Builder, llvm::StringRef Prefix,
                              const llvm::fltSemantics *Sem) {
  const char *DenormMin, *Epsilon, *Max, *Min;
  DenormMin = PickFP(Sem, "1.40129846e-45F", "4.9406564584124654e-324",
                     "3.64519953188247460253e-4951L",
                     "4.94065645841246544176568792868221e-324L",
                     "6.47517511943802511092443895822764655e-4966L");
  int Digits = PickFP(Sem, 6, 15, 18, 31, 33);
  Epsilon = PickFP(Sem, "1.19209290e-7F", "2.2204460492503131e-16",
                   "1.08420217248550443401e-19L",
                   "4.94065645841246544176568792868221e-324L",
                   "1.92592994438723585305597794258492732e-34L");
  int MantissaDigits = PickFP(Sem, 24, 53, 64, 106, 113);
  int Min10Exp = PickFP(Sem, -37, -307, -4931, -291, -4931);
  int Max10Exp = PickFP(Sem, 38, 308, 4932, 308, 4932);
  int MinExp = PickFP(Sem, -125, -1021, -16381, -968, -16381);
  int MaxExp = PickFP(Sem, 128, 1024, 16384, 1024, 16384);
  Min = PickFP(Sem, "1.17549435e-38F", "2.2250738585072014e-308",
               "3.36210314311209350626e-4932L",
               "2.00416836000897277799610805135016e-292L",
               "3.36210314311209350626267781732175260e-4932L");
  Max = PickFP(Sem, "3.40282347e+38F", "1.7976931348623157e+308",
               "1.18973149535723176502e+4932L",
               "1.79769313486231580793728971405301e+308L",
               "1.18973149535723176508575932662800702e+4932L");

  llvm::SmallString<32> DefPrefix;
  DefPrefix = "__";
  DefPrefix += Prefix;
  DefPrefix += "_";

  Builder.defineMacro(DefPrefix + "DENORM_MIN__", DenormMin);
  Builder.defineMacro(DefPrefix + "HAS_DENORM__");
  Builder.defineMacro(DefPrefix + "DIG__", llvm::Twine(Digits));
  Builder.defineMacro(DefPrefix + "EPSILON__", llvm::Twine(Epsilon));
  Builder.defineMacro(DefPrefix + "HAS_INFINITY__");
  Builder.defineMacro(DefPrefix + "HAS_QUIET_NAN__");
  Builder.defineMacro(DefPrefix + "MANT_DIG__", llvm::Twine(MantissaDigits));

  Builder.defineMacro(DefPrefix + "MAX_10_EXP__", llvm::Twine(Max10Exp));
  Builder.defineMacro(DefPrefix + "MAX_EXP__", llvm::Twine(MaxExp));
  Builder.defineMacro(DefPrefix + "MAX__", llvm::Twine(Max));

  Builder.defineMacro(DefPrefix + "MIN_10_EXP__","("+llvm::Twine(Min10Exp)+")");
  Builder.defineMacro(DefPrefix + "MIN_EXP__", "("+llvm::Twine(MinExp)+")");
  Builder.defineMacro(DefPrefix + "MIN__", llvm::Twine(Min));
}


/// DefineTypeSize - Emit a macro to the predefines buffer that declares a macro
/// named MacroName with the max value for a type with width 'TypeWidth' a
/// signedness of 'isSigned' and with a value suffix of 'ValSuffix' (e.g. LL).
static void DefineTypeSize(llvm::StringRef MacroName, unsigned TypeWidth,
                           llvm::StringRef ValSuffix, bool isSigned,
                           MacroBuilder& Builder) {
  long long MaxVal;
  if (isSigned) {
    assert(TypeWidth != 1);
    MaxVal = ~0ULL >> (65-TypeWidth);
  } else
    MaxVal = ~0ULL >> (64-TypeWidth);

  Builder.defineMacro(MacroName, llvm::Twine(MaxVal) + ValSuffix);
}

/// DefineTypeSize - An overloaded helper that uses TargetInfo to determine
/// the width, suffix, and signedness of the given type
static void DefineTypeSize(llvm::StringRef MacroName, TargetInfo::IntType Ty,
                           const TargetInfo &TI, MacroBuilder &Builder) {
  DefineTypeSize(MacroName, TI.getTypeWidth(Ty), TI.getTypeConstantSuffix(Ty), 
                 TI.isTypeSigned(Ty), Builder);
}

static void DefineType(const llvm::Twine &MacroName, TargetInfo::IntType Ty,
                       MacroBuilder &Builder) {
  Builder.defineMacro(MacroName, TargetInfo::getTypeName(Ty));
}

static void DefineTypeWidth(llvm::StringRef MacroName, TargetInfo::IntType Ty,
                            const TargetInfo &TI, MacroBuilder &Builder) {
  Builder.defineMacro(MacroName, llvm::Twine(TI.getTypeWidth(Ty)));
}

static void DefineTypeSizeof(llvm::StringRef MacroName, unsigned BitWidth,
                             const TargetInfo &TI, MacroBuilder &Builder) {
  Builder.defineMacro(MacroName,
                      llvm::Twine(BitWidth / TI.getCharWidth()));
}

static void DefineExactWidthIntType(TargetInfo::IntType Ty, 
                               const TargetInfo &TI, MacroBuilder &Builder) {
  int TypeWidth = TI.getTypeWidth(Ty);

  // Use the target specified int64 type, when appropriate, so that [u]int64_t
  // ends up being defined in terms of the correct type.
  if (TypeWidth == 64)
    Ty = TI.getInt64Type();

  DefineType("__INT" + llvm::Twine(TypeWidth) + "_TYPE__", Ty, Builder);

  llvm::StringRef ConstSuffix(TargetInfo::getTypeConstantSuffix(Ty));
  if (!ConstSuffix.empty())
    Builder.defineMacro("__INT" + llvm::Twine(TypeWidth) + "_C_SUFFIX__",
                        ConstSuffix);
}

static void InitializePredefinedMacros(const TargetInfo &TI,
                                       const LangOptions &LangOpts,
                                       const FrontendOptions &FEOpts,
                                       MacroBuilder &Builder) {
  // Compiler version introspection macros.
  Builder.defineMacro("__llvm__");  // LLVM Backend
  Builder.defineMacro("__clang__"); // Clang Frontend
#define TOSTR2(X) #X
#define TOSTR(X) TOSTR2(X)
  Builder.defineMacro("__clang_major__", TOSTR(CLANG_VERSION_MAJOR));
  Builder.defineMacro("__clang_minor__", TOSTR(CLANG_VERSION_MINOR));
#ifdef CLANG_VERSION_PATCHLEVEL
  Builder.defineMacro("__clang_patchlevel__", TOSTR(CLANG_VERSION_PATCHLEVEL));
#else
  Builder.defineMacro("__clang_patchlevel__", "0");
#endif
  Builder.defineMacro("__clang_version__", 
                      "\"" CLANG_VERSION_STRING " ("
                      + getClangFullRepositoryVersion() + ")\"");
#undef TOSTR
#undef TOSTR2
  // Currently claim to be compatible with GCC 4.2.1-5621.
  Builder.defineMacro("__GNUC_MINOR__", "2");
  Builder.defineMacro("__GNUC_PATCHLEVEL__", "1");
  Builder.defineMacro("__GNUC__", "4");
  Builder.defineMacro("__GXX_ABI_VERSION", "1002");
  Builder.defineMacro("__VERSION__", "\"4.2.1 Compatible Clang Compiler\"");

  // Initialize language-specific preprocessor defines.

  // These should all be defined in the preprocessor according to the
  // current language configuration.
  if (!LangOpts.Microsoft)
    Builder.defineMacro("__STDC__");
  if (LangOpts.AsmPreprocessor)
    Builder.defineMacro("__ASSEMBLER__");

  if (!LangOpts.CPlusPlus) {
    if (LangOpts.C99)
      Builder.defineMacro("__STDC_VERSION__", "199901L");
    else if (!LangOpts.GNUMode && LangOpts.Digraphs)
      Builder.defineMacro("__STDC_VERSION__", "199409L");
  }

  // Standard conforming mode?
  if (!LangOpts.GNUMode)
    Builder.defineMacro("__STRICT_ANSI__");

  if (LangOpts.CPlusPlus0x)
    Builder.defineMacro("__GXX_EXPERIMENTAL_CXX0X__");

  if (LangOpts.Freestanding)
    Builder.defineMacro("__STDC_HOSTED__", "0");
  else
    Builder.defineMacro("__STDC_HOSTED__");

  if (LangOpts.ObjC1) {
    Builder.defineMacro("__OBJC__");
    if (LangOpts.ObjCNonFragileABI) {
      Builder.defineMacro("__OBJC2__");
      Builder.defineMacro("OBJC_ZEROCOST_EXCEPTIONS");
    }

    if (LangOpts.getGCMode() != LangOptions::NonGC)
      Builder.defineMacro("__OBJC_GC__");

    if (LangOpts.NeXTRuntime)
      Builder.defineMacro("__NEXT_RUNTIME__");
  }

  // darwin_constant_cfstrings controls this. This is also dependent
  // on other things like the runtime I believe.  This is set even for C code.
  Builder.defineMacro("__CONSTANT_CFSTRINGS__");

  if (LangOpts.ObjC2)
    Builder.defineMacro("OBJC_NEW_PROPERTIES");

  if (LangOpts.PascalStrings)
    Builder.defineMacro("__PASCAL_STRINGS__");

  if (LangOpts.Blocks) {
    Builder.defineMacro("__block", "__attribute__((__blocks__(byref)))");
    Builder.defineMacro("__BLOCKS__");
  }

  if (LangOpts.Exceptions)
    Builder.defineMacro("__EXCEPTIONS");
  if (LangOpts.RTTI)
    Builder.defineMacro("__GXX_RTTI");
  if (LangOpts.SjLjExceptions)
    Builder.defineMacro("__USING_SJLJ_EXCEPTIONS__");

  if (LangOpts.CPlusPlus) {
    Builder.defineMacro("__DEPRECATED");
    Builder.defineMacro("__GNUG__", "4");
    Builder.defineMacro("__GXX_WEAK__");
    if (LangOpts.GNUMode)
      Builder.defineMacro("__cplusplus");
    else
      // C++ [cpp.predefined]p1:
      //   The name_ _cplusplusis defined to the value 199711L when compiling a
      //   C++ translation unit.
      Builder.defineMacro("__cplusplus", "199711L");
    Builder.defineMacro("__private_extern__", "extern");
  }

  if (LangOpts.Microsoft) {
    // Filter out some microsoft extensions when trying to parse in ms-compat
    // mode.
    Builder.defineMacro("__int8", "__INT8_TYPE__");
    Builder.defineMacro("__int16", "__INT16_TYPE__");
    Builder.defineMacro("__int32", "__INT32_TYPE__");
    Builder.defineMacro("__int64", "__INT64_TYPE__");
    // Both __PRETTY_FUNCTION__ and __FUNCTION__ are GCC extensions, however
    // VC++ appears to only like __FUNCTION__.
    Builder.defineMacro("__PRETTY_FUNCTION__", "__FUNCTION__");
    // Work around some issues with Visual C++ headerws.
    if (LangOpts.CPlusPlus) {
      // Since we define wchar_t in C++ mode.
      Builder.defineMacro("_WCHAR_T_DEFINED");
      Builder.defineMacro("_NATIVE_WCHAR_T_DEFINED");
      // FIXME:  This should be temporary until we have a __pragma
      // solution, to avoid some errors flagged in VC++ headers.
      Builder.defineMacro("_CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES", "0");
    }
  }

  if (LangOpts.Optimize)
    Builder.defineMacro("__OPTIMIZE__");
  if (LangOpts.OptimizeSize)
    Builder.defineMacro("__OPTIMIZE_SIZE__");

  // Initialize target-specific preprocessor defines.

  // Define type sizing macros based on the target properties.
  assert(TI.getCharWidth() == 8 && "Only support 8-bit char so far");
  Builder.defineMacro("__CHAR_BIT__", "8");

  DefineTypeSize("__SCHAR_MAX__", TI.getCharWidth(), "", true, Builder);
  DefineTypeSize("__SHRT_MAX__", TargetInfo::SignedShort, TI, Builder);
  DefineTypeSize("__INT_MAX__", TargetInfo::SignedInt, TI, Builder);
  DefineTypeSize("__LONG_MAX__", TargetInfo::SignedLong, TI, Builder);
  DefineTypeSize("__LONG_LONG_MAX__", TargetInfo::SignedLongLong, TI, Builder);
  DefineTypeSize("__WCHAR_MAX__", TI.getWCharType(), TI, Builder);
  DefineTypeSize("__INTMAX_MAX__", TI.getIntMaxType(), TI, Builder);

  DefineTypeSizeof("__SIZEOF_DOUBLE__", TI.getDoubleWidth(), TI, Builder);
  DefineTypeSizeof("__SIZEOF_FLOAT__", TI.getFloatWidth(), TI, Builder);
  DefineTypeSizeof("__SIZEOF_INT__", TI.getIntWidth(), TI, Builder);
  DefineTypeSizeof("__SIZEOF_LONG__", TI.getLongWidth(), TI, Builder);
  DefineTypeSizeof("__SIZEOF_LONG_DOUBLE__",TI.getLongDoubleWidth(),TI,Builder);
  DefineTypeSizeof("__SIZEOF_LONG_LONG__", TI.getLongLongWidth(), TI, Builder);
  DefineTypeSizeof("__SIZEOF_POINTER__", TI.getPointerWidth(0), TI, Builder);
  DefineTypeSizeof("__SIZEOF_SHORT__", TI.getShortWidth(), TI, Builder);
  DefineTypeSizeof("__SIZEOF_PTRDIFF_T__",
                   TI.getTypeWidth(TI.getPtrDiffType(0)), TI, Builder);
  DefineTypeSizeof("__SIZEOF_SIZE_T__",
                   TI.getTypeWidth(TI.getSizeType()), TI, Builder);
  DefineTypeSizeof("__SIZEOF_WCHAR_T__",
                   TI.getTypeWidth(TI.getWCharType()), TI, Builder);
  DefineTypeSizeof("__SIZEOF_WINT_T__",
                   TI.getTypeWidth(TI.getWIntType()), TI, Builder);

  DefineType("__INTMAX_TYPE__", TI.getIntMaxType(), Builder);
  DefineType("__UINTMAX_TYPE__", TI.getUIntMaxType(), Builder);
  DefineTypeWidth("__INTMAX_WIDTH__",  TI.getIntMaxType(), TI, Builder);
  DefineType("__PTRDIFF_TYPE__", TI.getPtrDiffType(0), Builder);
  DefineTypeWidth("__PTRDIFF_WIDTH__", TI.getPtrDiffType(0), TI, Builder);
  DefineType("__INTPTR_TYPE__", TI.getIntPtrType(), Builder);
  DefineTypeWidth("__INTPTR_WIDTH__", TI.getIntPtrType(), TI, Builder);
  DefineType("__SIZE_TYPE__", TI.getSizeType(), Builder);
  DefineTypeWidth("__SIZE_WIDTH__", TI.getSizeType(), TI, Builder);
  DefineType("__WCHAR_TYPE__", TI.getWCharType(), Builder);
  DefineTypeWidth("__WCHAR_WIDTH__", TI.getWCharType(), TI, Builder);
  DefineType("__WINT_TYPE__", TI.getWIntType(), Builder);
  DefineTypeWidth("__WINT_WIDTH__", TI.getWIntType(), TI, Builder);
  DefineTypeWidth("__SIG_ATOMIC_WIDTH__", TI.getSigAtomicType(), TI, Builder);
  DefineType("__CHAR16_TYPE__", TI.getChar16Type(), Builder);
  DefineType("__CHAR32_TYPE__", TI.getChar32Type(), Builder);

  DefineFloatMacros(Builder, "FLT", &TI.getFloatFormat());
  DefineFloatMacros(Builder, "DBL", &TI.getDoubleFormat());
  DefineFloatMacros(Builder, "LDBL", &TI.getLongDoubleFormat());

  // Define a __POINTER_WIDTH__ macro for stdint.h.
  Builder.defineMacro("__POINTER_WIDTH__",
                      llvm::Twine((int)TI.getPointerWidth(0)));

  if (!LangOpts.CharIsSigned)
    Builder.defineMacro("__CHAR_UNSIGNED__");

  // Define exact-width integer types for stdint.h
  Builder.defineMacro("__INT" + llvm::Twine(TI.getCharWidth()) + "_TYPE__",
                      "char");

  if (TI.getShortWidth() > TI.getCharWidth())
    DefineExactWidthIntType(TargetInfo::SignedShort, TI, Builder);

  if (TI.getIntWidth() > TI.getShortWidth())
    DefineExactWidthIntType(TargetInfo::SignedInt, TI, Builder);

  if (TI.getLongWidth() > TI.getIntWidth())
    DefineExactWidthIntType(TargetInfo::SignedLong, TI, Builder);

  if (TI.getLongLongWidth() > TI.getLongWidth())
    DefineExactWidthIntType(TargetInfo::SignedLongLong, TI, Builder);

  // Add __builtin_va_list typedef.
  Builder.append(TI.getVAListDeclaration());

  if (const char *Prefix = TI.getUserLabelPrefix())
    Builder.defineMacro("__USER_LABEL_PREFIX__", Prefix);

  // Build configuration options.  FIXME: these should be controlled by
  // command line options or something.
  Builder.defineMacro("__FINITE_MATH_ONLY__", "0");

  if (LangOpts.GNUInline)
    Builder.defineMacro("__GNUC_GNU_INLINE__");
  else
    Builder.defineMacro("__GNUC_STDC_INLINE__");

  if (LangOpts.NoInline)
    Builder.defineMacro("__NO_INLINE__");

  if (unsigned PICLevel = LangOpts.PICLevel) {
    Builder.defineMacro("__PIC__", llvm::Twine(PICLevel));
    Builder.defineMacro("__pic__", llvm::Twine(PICLevel));
  }

  // Macros to control C99 numerics and <float.h>
  Builder.defineMacro("__FLT_EVAL_METHOD__", "0");
  Builder.defineMacro("__FLT_RADIX__", "2");
  int Dig = PickFP(&TI.getLongDoubleFormat(), -1/*FIXME*/, 17, 21, 33, 36);
  Builder.defineMacro("__DECIMAL_DIG__", llvm::Twine(Dig));

  if (LangOpts.getStackProtectorMode() == LangOptions::SSPOn)
    Builder.defineMacro("__SSP__");
  else if (LangOpts.getStackProtectorMode() == LangOptions::SSPReq)
    Builder.defineMacro("__SSP_ALL__", "2");

  if (FEOpts.ProgramAction == frontend::RewriteObjC)
    Builder.defineMacro("__weak", "__attribute__((objc_gc(weak)))");

  // Define a macro that exists only when using the static analyzer.
  if (FEOpts.ProgramAction == frontend::RunAnalysis)
    Builder.defineMacro("__clang_analyzer__");

  // Get other target #defines.
  TI.getTargetDefines(LangOpts, Builder);
}

// Initialize the remapping of files to alternative contents, e.g.,
// those specified through other files.
static void InitializeFileRemapping(Diagnostic &Diags,
                                    SourceManager &SourceMgr,
                                    FileManager &FileMgr,
                                    const PreprocessorOptions &InitOpts) {
  // Remap files in the source manager (with buffers).
  for (PreprocessorOptions::const_remapped_file_buffer_iterator
         Remap = InitOpts.remapped_file_buffer_begin(),
         RemapEnd = InitOpts.remapped_file_buffer_end();
       Remap != RemapEnd;
       ++Remap) {
    // Create the file entry for the file that we're mapping from.
    const FileEntry *FromFile = FileMgr.getVirtualFile(Remap->first,
                                                Remap->second->getBufferSize(),
                                                       0);
    if (!FromFile) {
      Diags.Report(diag::err_fe_remap_missing_from_file)
        << Remap->first;
      if (!InitOpts.RetainRemappedFileBuffers)
        delete Remap->second;
      continue;
    }

    // Override the contents of the "from" file with the contents of
    // the "to" file.
    SourceMgr.overrideFileContents(FromFile, Remap->second,
                                   InitOpts.RetainRemappedFileBuffers);
  }

  // Remap files in the source manager (with other files).
  for (PreprocessorOptions::const_remapped_file_iterator
         Remap = InitOpts.remapped_file_begin(),
         RemapEnd = InitOpts.remapped_file_end();
       Remap != RemapEnd;
       ++Remap) {
    // Find the file that we're mapping to.
    const FileEntry *ToFile = FileMgr.getFile(Remap->second);
    if (!ToFile) {
      Diags.Report(diag::err_fe_remap_missing_to_file)
      << Remap->first << Remap->second;
      continue;
    }
    
    // Create the file entry for the file that we're mapping from.
    const FileEntry *FromFile = FileMgr.getVirtualFile(Remap->first,
                                                       ToFile->getSize(),
                                                       0);
    if (!FromFile) {
      Diags.Report(diag::err_fe_remap_missing_from_file)
      << Remap->first;
      continue;
    }
    
    // Load the contents of the file we're mapping to.
    std::string ErrorStr;
    const llvm::MemoryBuffer *Buffer
    = llvm::MemoryBuffer::getFile(ToFile->getName(), &ErrorStr);
    if (!Buffer) {
      Diags.Report(diag::err_fe_error_opening)
        << Remap->second << ErrorStr;
      continue;
    }
    
    // Override the contents of the "from" file with the contents of
    // the "to" file.
    SourceMgr.overrideFileContents(FromFile, Buffer);
  }
}

/// InitializePreprocessor - Initialize the preprocessor getting it and the
/// environment ready to process a single file. This returns true on error.
///
void clang::InitializePreprocessor(Preprocessor &PP,
                                   const PreprocessorOptions &InitOpts,
                                   const HeaderSearchOptions &HSOpts,
                                   const FrontendOptions &FEOpts) {
  std::string PredefineBuffer;
  PredefineBuffer.reserve(4080);
  llvm::raw_string_ostream Predefines(PredefineBuffer);
  MacroBuilder Builder(Predefines);

  InitializeFileRemapping(PP.getDiagnostics(), PP.getSourceManager(),
                          PP.getFileManager(), InitOpts);

  // Emit line markers for various builtin sections of the file.  We don't do
  // this in asm preprocessor mode, because "# 4" is not a line marker directive
  // in this mode.
  if (!PP.getLangOptions().AsmPreprocessor)
    Builder.append("# 1 \"<built-in>\" 3");

  // Install things like __POWERPC__, __GNUC__, etc into the macro table.
  if (InitOpts.UsePredefines)
    InitializePredefinedMacros(PP.getTargetInfo(), PP.getLangOptions(),
                               FEOpts, Builder);

  // Add on the predefines from the driver.  Wrap in a #line directive to report
  // that they come from the command line.
  if (!PP.getLangOptions().AsmPreprocessor)
    Builder.append("# 1 \"<command line>\" 1");

  // Process #define's and #undef's in the order they are given.
  for (unsigned i = 0, e = InitOpts.Macros.size(); i != e; ++i) {
    if (InitOpts.Macros[i].second)  // isUndef
      Builder.undefineMacro(InitOpts.Macros[i].first);
    else
      DefineBuiltinMacro(Builder, InitOpts.Macros[i].first,
                         PP.getDiagnostics());
  }

  // If -imacros are specified, include them now.  These are processed before
  // any -include directives.
  for (unsigned i = 0, e = InitOpts.MacroIncludes.size(); i != e; ++i)
    AddImplicitIncludeMacros(Builder, InitOpts.MacroIncludes[i]);

  // Process -include directives.
  for (unsigned i = 0, e = InitOpts.Includes.size(); i != e; ++i) {
    const std::string &Path = InitOpts.Includes[i];
    if (Path == InitOpts.ImplicitPTHInclude)
      AddImplicitIncludePTH(Builder, PP, Path);
    else
      AddImplicitInclude(Builder, Path);
  }

  // Exit the command line and go back to <built-in> (2 is LC_LEAVE).
  if (!PP.getLangOptions().AsmPreprocessor)
    Builder.append("# 1 \"<built-in>\" 2");

  // Instruct the preprocessor to skip the preamble.
  PP.setSkipMainFilePreamble(InitOpts.PrecompiledPreambleBytes.first,
                             InitOpts.PrecompiledPreambleBytes.second);
                          
  // Copy PredefinedBuffer into the Preprocessor.
  PP.setPredefines(Predefines.str());

  // Initialize the header search object.
  ApplyHeaderSearchOptions(PP.getHeaderSearchInfo(), HSOpts,
                           PP.getLangOptions(),
                           PP.getTargetInfo().getTriple());
}
