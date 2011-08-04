//===--- ASTReader.cpp - AST File Reader ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTReader class, which reads AST files.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "ASTCommon.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/Utils.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Scope.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/VersionTuple.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <sys/stat.h>
#include <iostream>

using namespace clang;
using namespace clang::serialization;

//===----------------------------------------------------------------------===//
// PCH validator implementation
//===----------------------------------------------------------------------===//

ASTReaderListener::~ASTReaderListener() {}

bool
PCHValidator::ReadLanguageOptions(const LangOptions &LangOpts) {
  const LangOptions &PPLangOpts = PP.getLangOptions();
#define PARSE_LANGOPT_BENIGN(Option)
#define PARSE_LANGOPT_IMPORTANT(Option, DiagID)                    \
  if (PPLangOpts.Option != LangOpts.Option) {                      \
    Reader.Diag(DiagID) << LangOpts.Option << PPLangOpts.Option;   \
    return true;                                                   \
  }

  PARSE_LANGOPT_BENIGN(Trigraphs);
  PARSE_LANGOPT_BENIGN(BCPLComment);
  PARSE_LANGOPT_BENIGN(DollarIdents);
  PARSE_LANGOPT_BENIGN(AsmPreprocessor);
  PARSE_LANGOPT_IMPORTANT(GNUMode, diag::warn_pch_gnu_extensions);
  PARSE_LANGOPT_IMPORTANT(GNUKeywords, diag::warn_pch_gnu_keywords);
  PARSE_LANGOPT_BENIGN(ImplicitInt);
  PARSE_LANGOPT_BENIGN(Digraphs);
  PARSE_LANGOPT_BENIGN(HexFloats);
  PARSE_LANGOPT_IMPORTANT(C99, diag::warn_pch_c99);
  PARSE_LANGOPT_IMPORTANT(C1X, diag::warn_pch_c1x);
  PARSE_LANGOPT_IMPORTANT(Microsoft, diag::warn_pch_microsoft_extensions);
  PARSE_LANGOPT_BENIGN(MSCVersion);
  PARSE_LANGOPT_IMPORTANT(CPlusPlus, diag::warn_pch_cplusplus);
  PARSE_LANGOPT_IMPORTANT(CPlusPlus0x, diag::warn_pch_cplusplus0x);
  PARSE_LANGOPT_BENIGN(CXXOperatorName);
  PARSE_LANGOPT_IMPORTANT(ObjC1, diag::warn_pch_objective_c);
  PARSE_LANGOPT_IMPORTANT(ObjC2, diag::warn_pch_objective_c2);
  PARSE_LANGOPT_IMPORTANT(ObjCNonFragileABI, diag::warn_pch_nonfragile_abi);
  PARSE_LANGOPT_IMPORTANT(ObjCNonFragileABI2, diag::warn_pch_nonfragile_abi2);
  PARSE_LANGOPT_IMPORTANT(AppleKext, diag::warn_pch_apple_kext);
  PARSE_LANGOPT_IMPORTANT(ObjCDefaultSynthProperties,
                          diag::warn_pch_objc_auto_properties);
  PARSE_LANGOPT_BENIGN(ObjCInferRelatedResultType)
  PARSE_LANGOPT_IMPORTANT(NoConstantCFStrings,
                          diag::warn_pch_no_constant_cfstrings);
  PARSE_LANGOPT_BENIGN(PascalStrings);
  PARSE_LANGOPT_BENIGN(WritableStrings);
  PARSE_LANGOPT_IMPORTANT(LaxVectorConversions,
                          diag::warn_pch_lax_vector_conversions);
  PARSE_LANGOPT_IMPORTANT(AltiVec, diag::warn_pch_altivec);
  PARSE_LANGOPT_IMPORTANT(Exceptions, diag::warn_pch_exceptions);
  PARSE_LANGOPT_IMPORTANT(ObjCExceptions, diag::warn_pch_objc_exceptions);
  PARSE_LANGOPT_IMPORTANT(CXXExceptions, diag::warn_pch_cxx_exceptions);
  PARSE_LANGOPT_IMPORTANT(SjLjExceptions, diag::warn_pch_sjlj_exceptions);
  PARSE_LANGOPT_IMPORTANT(MSBitfields, diag::warn_pch_ms_bitfields);
  PARSE_LANGOPT_IMPORTANT(NeXTRuntime, diag::warn_pch_objc_runtime);
  PARSE_LANGOPT_IMPORTANT(Freestanding, diag::warn_pch_freestanding);
  PARSE_LANGOPT_IMPORTANT(NoBuiltin, diag::warn_pch_builtins);
  PARSE_LANGOPT_IMPORTANT(ThreadsafeStatics,
                          diag::warn_pch_thread_safe_statics);
  PARSE_LANGOPT_IMPORTANT(POSIXThreads, diag::warn_pch_posix_threads);
  PARSE_LANGOPT_IMPORTANT(Blocks, diag::warn_pch_blocks);
  PARSE_LANGOPT_BENIGN(EmitAllDecls);
  PARSE_LANGOPT_IMPORTANT(MathErrno, diag::warn_pch_math_errno);
  PARSE_LANGOPT_BENIGN(getSignedOverflowBehavior());
  PARSE_LANGOPT_IMPORTANT(HeinousExtensions,
                          diag::warn_pch_heinous_extensions);
  // FIXME: Most of the options below are benign if the macro wasn't
  // used. Unfortunately, this means that a PCH compiled without
  // optimization can't be used with optimization turned on, even
  // though the only thing that changes is whether __OPTIMIZE__ was
  // defined... but if __OPTIMIZE__ never showed up in the header, it
  // doesn't matter. We could consider making this some special kind
  // of check.
  PARSE_LANGOPT_IMPORTANT(Optimize, diag::warn_pch_optimize);
  PARSE_LANGOPT_IMPORTANT(OptimizeSize, diag::warn_pch_optimize_size);
  PARSE_LANGOPT_IMPORTANT(Static, diag::warn_pch_static);
  PARSE_LANGOPT_IMPORTANT(PICLevel, diag::warn_pch_pic_level);
  PARSE_LANGOPT_IMPORTANT(GNUInline, diag::warn_pch_gnu_inline);
  PARSE_LANGOPT_IMPORTANT(NoInline, diag::warn_pch_no_inline);
  PARSE_LANGOPT_IMPORTANT(Deprecated, diag::warn_pch_deprecated);
  PARSE_LANGOPT_IMPORTANT(AccessControl, diag::warn_pch_access_control);
  PARSE_LANGOPT_IMPORTANT(CharIsSigned, diag::warn_pch_char_signed);
  PARSE_LANGOPT_IMPORTANT(ShortWChar, diag::warn_pch_short_wchar);
  PARSE_LANGOPT_IMPORTANT(ShortEnums, diag::warn_pch_short_enums);
  if ((PPLangOpts.getGCMode() != 0) != (LangOpts.getGCMode() != 0)) {
    Reader.Diag(diag::warn_pch_gc_mode)
      << LangOpts.getGCMode() << PPLangOpts.getGCMode();
    return true;
  }
  PARSE_LANGOPT_BENIGN(getVisibilityMode());
  PARSE_LANGOPT_IMPORTANT(getStackProtectorMode(),
                          diag::warn_pch_stack_protector);
  PARSE_LANGOPT_BENIGN(InstantiationDepth);
  PARSE_LANGOPT_IMPORTANT(OpenCL, diag::warn_pch_opencl);
  PARSE_LANGOPT_IMPORTANT(CUDA, diag::warn_pch_cuda);
  PARSE_LANGOPT_BENIGN(CatchUndefined);
  PARSE_LANGOPT_BENIGN(DefaultFPContract);
  PARSE_LANGOPT_IMPORTANT(ElideConstructors, diag::warn_pch_elide_constructors);
  PARSE_LANGOPT_BENIGN(SpellChecking);
  PARSE_LANGOPT_IMPORTANT(ObjCAutoRefCount, diag::warn_pch_auto_ref_count);
  PARSE_LANGOPT_BENIGN(ObjCInferRelatedReturnType);
#undef PARSE_LANGOPT_IMPORTANT
#undef PARSE_LANGOPT_BENIGN

  return false;
}

bool PCHValidator::ReadTargetTriple(StringRef Triple) {
  if (Triple == PP.getTargetInfo().getTriple().str())
    return false;

  Reader.Diag(diag::warn_pch_target_triple)
    << Triple << PP.getTargetInfo().getTriple().str();
  return true;
}

namespace {
  struct EmptyStringRef {
    bool operator ()(StringRef r) const { return r.empty(); }
  };
  struct EmptyBlock {
    bool operator ()(const PCHPredefinesBlock &r) const {return r.Data.empty();}
  };
}

static bool EqualConcatenations(SmallVector<StringRef, 2> L,
                                PCHPredefinesBlocks R) {
  // First, sum up the lengths.
  unsigned LL = 0, RL = 0;
  for (unsigned I = 0, N = L.size(); I != N; ++I) {
    LL += L[I].size();
  }
  for (unsigned I = 0, N = R.size(); I != N; ++I) {
    RL += R[I].Data.size();
  }
  if (LL != RL)
    return false;
  if (LL == 0 && RL == 0)
    return true;

  // Kick out empty parts, they confuse the algorithm below.
  L.erase(std::remove_if(L.begin(), L.end(), EmptyStringRef()), L.end());
  R.erase(std::remove_if(R.begin(), R.end(), EmptyBlock()), R.end());

  // Do it the hard way. At this point, both vectors must be non-empty.
  StringRef LR = L[0], RR = R[0].Data;
  unsigned LI = 0, RI = 0, LN = L.size(), RN = R.size();
  (void) RN;
  for (;;) {
    // Compare the current pieces.
    if (LR.size() == RR.size()) {
      // If they're the same length, it's pretty easy.
      if (LR != RR)
        return false;
      // Both pieces are done, advance.
      ++LI;
      ++RI;
      // If either string is done, they're both done, since they're the same
      // length.
      if (LI == LN) {
        assert(RI == RN && "Strings not the same length after all?");
        return true;
      }
      LR = L[LI];
      RR = R[RI].Data;
    } else if (LR.size() < RR.size()) {
      // Right piece is longer.
      if (!RR.startswith(LR))
        return false;
      ++LI;
      assert(LI != LN && "Strings not the same length after all?");
      RR = RR.substr(LR.size());
      LR = L[LI];
    } else {
      // Left piece is longer.
      if (!LR.startswith(RR))
        return false;
      ++RI;
      assert(RI != RN && "Strings not the same length after all?");
      LR = LR.substr(RR.size());
      RR = R[RI].Data;
    }
  }
}

static std::pair<FileID, StringRef::size_type>
FindMacro(const PCHPredefinesBlocks &Buffers, StringRef MacroDef) {
  std::pair<FileID, StringRef::size_type> Res;
  for (unsigned I = 0, N = Buffers.size(); I != N; ++I) {
    Res.second = Buffers[I].Data.find(MacroDef);
    if (Res.second != StringRef::npos) {
      Res.first = Buffers[I].BufferID;
      break;
    }
  }
  return Res;
}

bool PCHValidator::ReadPredefinesBuffer(const PCHPredefinesBlocks &Buffers,
                                        StringRef OriginalFileName,
                                        std::string &SuggestedPredefines,
                                        FileManager &FileMgr) {
  // We are in the context of an implicit include, so the predefines buffer will
  // have a #include entry for the PCH file itself (as normalized by the
  // preprocessor initialization). Find it and skip over it in the checking
  // below.
  llvm::SmallString<256> PCHInclude;
  PCHInclude += "#include \"";
  PCHInclude += NormalizeDashIncludePath(OriginalFileName, FileMgr);
  PCHInclude += "\"\n";
  std::pair<StringRef,StringRef> Split =
    StringRef(PP.getPredefines()).split(PCHInclude.str());
  StringRef Left =  Split.first, Right = Split.second;
  if (Left == PP.getPredefines()) {
    Error("Missing PCH include entry!");
    return true;
  }

  // If the concatenation of all the PCH buffers is equal to the adjusted
  // command line, we're done.
  SmallVector<StringRef, 2> CommandLine;
  CommandLine.push_back(Left);
  CommandLine.push_back(Right);
  if (EqualConcatenations(CommandLine, Buffers))
    return false;

  SourceManager &SourceMgr = PP.getSourceManager();

  // The predefines buffers are different. Determine what the differences are,
  // and whether they require us to reject the PCH file.
  SmallVector<StringRef, 8> PCHLines;
  for (unsigned I = 0, N = Buffers.size(); I != N; ++I)
    Buffers[I].Data.split(PCHLines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  SmallVector<StringRef, 8> CmdLineLines;
  Left.split(CmdLineLines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  // Pick out implicit #includes after the PCH and don't consider them for
  // validation; we will insert them into SuggestedPredefines so that the
  // preprocessor includes them.
  std::string IncludesAfterPCH;
  SmallVector<StringRef, 8> AfterPCHLines;
  Right.split(AfterPCHLines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (unsigned i = 0, e = AfterPCHLines.size(); i != e; ++i) {
    if (AfterPCHLines[i].startswith("#include ")) {
      IncludesAfterPCH += AfterPCHLines[i];
      IncludesAfterPCH += '\n';
    } else {
      CmdLineLines.push_back(AfterPCHLines[i]);
    }
  }

  // Make sure we add the includes last into SuggestedPredefines before we
  // exit this function.
  struct AddIncludesRAII {
    std::string &SuggestedPredefines;
    std::string &IncludesAfterPCH;

    AddIncludesRAII(std::string &SuggestedPredefines,
                    std::string &IncludesAfterPCH)
      : SuggestedPredefines(SuggestedPredefines),
        IncludesAfterPCH(IncludesAfterPCH) { }
    ~AddIncludesRAII() {
      SuggestedPredefines += IncludesAfterPCH;
    }
  } AddIncludes(SuggestedPredefines, IncludesAfterPCH);

  // Sort both sets of predefined buffer lines, since we allow some extra
  // definitions and they may appear at any point in the output.
  std::sort(CmdLineLines.begin(), CmdLineLines.end());
  std::sort(PCHLines.begin(), PCHLines.end());

  // Determine which predefines that were used to build the PCH file are missing
  // from the command line.
  std::vector<StringRef> MissingPredefines;
  std::set_difference(PCHLines.begin(), PCHLines.end(),
                      CmdLineLines.begin(), CmdLineLines.end(),
                      std::back_inserter(MissingPredefines));

  bool MissingDefines = false;
  bool ConflictingDefines = false;
  for (unsigned I = 0, N = MissingPredefines.size(); I != N; ++I) {
    StringRef Missing = MissingPredefines[I];
    if (Missing.startswith("#include ")) {
      // An -include was specified when generating the PCH; it is included in
      // the PCH, just ignore it.
      continue;
    }
    if (!Missing.startswith("#define ")) {
      Reader.Diag(diag::warn_pch_compiler_options_mismatch);
      return true;
    }

    // This is a macro definition. Determine the name of the macro we're
    // defining.
    std::string::size_type StartOfMacroName = strlen("#define ");
    std::string::size_type EndOfMacroName
      = Missing.find_first_of("( \n\r", StartOfMacroName);
    assert(EndOfMacroName != std::string::npos &&
           "Couldn't find the end of the macro name");
    StringRef MacroName = Missing.slice(StartOfMacroName, EndOfMacroName);

    // Determine whether this macro was given a different definition on the
    // command line.
    std::string MacroDefStart = "#define " + MacroName.str();
    std::string::size_type MacroDefLen = MacroDefStart.size();
    SmallVector<StringRef, 8>::iterator ConflictPos
      = std::lower_bound(CmdLineLines.begin(), CmdLineLines.end(),
                         MacroDefStart);
    for (; ConflictPos != CmdLineLines.end(); ++ConflictPos) {
      if (!ConflictPos->startswith(MacroDefStart)) {
        // Different macro; we're done.
        ConflictPos = CmdLineLines.end();
        break;
      }

      assert(ConflictPos->size() > MacroDefLen &&
             "Invalid #define in predefines buffer?");
      if ((*ConflictPos)[MacroDefLen] != ' ' &&
          (*ConflictPos)[MacroDefLen] != '(')
        continue; // Longer macro name; keep trying.

      // We found a conflicting macro definition.
      break;
    }

    if (ConflictPos != CmdLineLines.end()) {
      Reader.Diag(diag::warn_cmdline_conflicting_macro_def)
          << MacroName;

      // Show the definition of this macro within the PCH file.
      std::pair<FileID, StringRef::size_type> MacroLoc =
          FindMacro(Buffers, Missing);
      assert(MacroLoc.second!=StringRef::npos && "Unable to find macro!");
      SourceLocation PCHMissingLoc =
          SourceMgr.getLocForStartOfFile(MacroLoc.first)
            .getFileLocWithOffset(MacroLoc.second);
      Reader.Diag(PCHMissingLoc, diag::note_pch_macro_defined_as) << MacroName;

      ConflictingDefines = true;
      continue;
    }

    // If the macro doesn't conflict, then we'll just pick up the macro
    // definition from the PCH file. Warn the user that they made a mistake.
    if (ConflictingDefines)
      continue; // Don't complain if there are already conflicting defs

    if (!MissingDefines) {
      Reader.Diag(diag::warn_cmdline_missing_macro_defs);
      MissingDefines = true;
    }

    // Show the definition of this macro within the PCH file.
    std::pair<FileID, StringRef::size_type> MacroLoc =
        FindMacro(Buffers, Missing);
    assert(MacroLoc.second!=StringRef::npos && "Unable to find macro!");
    SourceLocation PCHMissingLoc =
        SourceMgr.getLocForStartOfFile(MacroLoc.first)
          .getFileLocWithOffset(MacroLoc.second);
    Reader.Diag(PCHMissingLoc, diag::note_using_macro_def_from_pch);
  }

  if (ConflictingDefines)
    return true;

  // Determine what predefines were introduced based on command-line
  // parameters that were not present when building the PCH
  // file. Extra #defines are okay, so long as the identifiers being
  // defined were not used within the precompiled header.
  std::vector<StringRef> ExtraPredefines;
  std::set_difference(CmdLineLines.begin(), CmdLineLines.end(),
                      PCHLines.begin(), PCHLines.end(),
                      std::back_inserter(ExtraPredefines));
  for (unsigned I = 0, N = ExtraPredefines.size(); I != N; ++I) {
    StringRef &Extra = ExtraPredefines[I];
    if (!Extra.startswith("#define ")) {
      Reader.Diag(diag::warn_pch_compiler_options_mismatch);
      return true;
    }

    // This is an extra macro definition. Determine the name of the
    // macro we're defining.
    std::string::size_type StartOfMacroName = strlen("#define ");
    std::string::size_type EndOfMacroName
      = Extra.find_first_of("( \n\r", StartOfMacroName);
    assert(EndOfMacroName != std::string::npos &&
           "Couldn't find the end of the macro name");
    StringRef MacroName = Extra.slice(StartOfMacroName, EndOfMacroName);

    // Check whether this name was used somewhere in the PCH file. If
    // so, defining it as a macro could change behavior, so we reject
    // the PCH file.
    if (IdentifierInfo *II = Reader.get(MacroName)) {
      Reader.Diag(diag::warn_macro_name_used_in_pch) << II;
      return true;
    }

    // Add this definition to the suggested predefines buffer.
    SuggestedPredefines += Extra;
    SuggestedPredefines += '\n';
  }

  // If we get here, it's because the predefines buffer had compatible
  // contents. Accept the PCH file.
  return false;
}

void PCHValidator::ReadHeaderFileInfo(const HeaderFileInfo &HFI,
                                      unsigned ID) {
  PP.getHeaderSearchInfo().setHeaderFileInfoForUID(HFI, ID);
  ++NumHeaderInfos;
}

void PCHValidator::ReadCounter(unsigned Value) {
  PP.setCounterValue(Value);
}

//===----------------------------------------------------------------------===//
// AST reader implementation
//===----------------------------------------------------------------------===//

void
ASTReader::setDeserializationListener(ASTDeserializationListener *Listener) {
  DeserializationListener = Listener;
}


namespace {
class ASTSelectorLookupTrait {
  ASTReader &Reader;
  Module &F;
  
public:
  struct data_type {
    SelectorID ID;
    ObjCMethodList Instance, Factory;
  };

  typedef Selector external_key_type;
  typedef external_key_type internal_key_type;

  ASTSelectorLookupTrait(ASTReader &Reader, Module &F) 
    : Reader(Reader), F(F) { }

  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return a == b;
  }

  static unsigned ComputeHash(Selector Sel) {
    return serialization::ComputeHash(Sel);
  }

  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace clang::io;
    unsigned KeyLen = ReadUnalignedLE16(d);
    unsigned DataLen = ReadUnalignedLE16(d);
    return std::make_pair(KeyLen, DataLen);
  }

  internal_key_type ReadKey(const unsigned char* d, unsigned) {
    using namespace clang::io;
    SelectorTable &SelTable = Reader.getContext()->Selectors;
    unsigned N = ReadUnalignedLE16(d);
    IdentifierInfo *FirstII
      = Reader.getLocalIdentifier(F, ReadUnalignedLE32(d));
    if (N == 0)
      return SelTable.getNullarySelector(FirstII);
    else if (N == 1)
      return SelTable.getUnarySelector(FirstII);

    SmallVector<IdentifierInfo *, 16> Args;
    Args.push_back(FirstII);
    for (unsigned I = 1; I != N; ++I)
      Args.push_back(Reader.getLocalIdentifier(F, ReadUnalignedLE32(d)));

    return SelTable.getSelector(N, Args.data());
  }

  data_type ReadData(Selector, const unsigned char* d, unsigned DataLen) {
    using namespace clang::io;

    data_type Result;

    Result.ID = Reader.getGlobalSelectorID(F, ReadUnalignedLE32(d));
    unsigned NumInstanceMethods = ReadUnalignedLE16(d);
    unsigned NumFactoryMethods = ReadUnalignedLE16(d);

    // Load instance methods
    ObjCMethodList *Prev = 0;
    for (unsigned I = 0; I != NumInstanceMethods; ++I) {
      ObjCMethodDecl *Method
        = Reader.GetLocalDeclAs<ObjCMethodDecl>(F, ReadUnalignedLE32(d));
      if (!Result.Instance.Method) {
        // This is the first method, which is the easy case.
        Result.Instance.Method = Method;
        Prev = &Result.Instance;
        continue;
      }

      ObjCMethodList *Mem =
        Reader.getSema()->BumpAlloc.Allocate<ObjCMethodList>();
      Prev->Next = new (Mem) ObjCMethodList(Method, 0);
      Prev = Prev->Next;
    }

    // Load factory methods
    Prev = 0;
    for (unsigned I = 0; I != NumFactoryMethods; ++I) {
      ObjCMethodDecl *Method
        = Reader.GetLocalDeclAs<ObjCMethodDecl>(F, ReadUnalignedLE32(d));
      if (!Result.Factory.Method) {
        // This is the first method, which is the easy case.
        Result.Factory.Method = Method;
        Prev = &Result.Factory;
        continue;
      }

      ObjCMethodList *Mem =
        Reader.getSema()->BumpAlloc.Allocate<ObjCMethodList>();
      Prev->Next = new (Mem) ObjCMethodList(Method, 0);
      Prev = Prev->Next;
    }

    return Result;
  }
};

} // end anonymous namespace

/// \brief The on-disk hash table used for the global method pool.
typedef OnDiskChainedHashTable<ASTSelectorLookupTrait>
  ASTSelectorLookupTable;

namespace clang {
class ASTIdentifierLookupTrait {
  ASTReader &Reader;
  Module &F;

  // If we know the IdentifierInfo in advance, it is here and we will
  // not build a new one. Used when deserializing information about an
  // identifier that was constructed before the AST file was read.
  IdentifierInfo *KnownII;

public:
  typedef IdentifierInfo * data_type;

  typedef const std::pair<const char*, unsigned> external_key_type;

  typedef external_key_type internal_key_type;

  ASTIdentifierLookupTrait(ASTReader &Reader, Module &F,
                           IdentifierInfo *II = 0)
    : Reader(Reader), F(F), KnownII(II) { }

  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return (a.second == b.second) ? memcmp(a.first, b.first, a.second) == 0
                                  : false;
  }

  static unsigned ComputeHash(const internal_key_type& a) {
    return llvm::HashString(StringRef(a.first, a.second));
  }

  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }

  // This hopefully will just get inlined and removed by the optimizer.
  static const external_key_type&
  GetExternalKey(const internal_key_type& x) { return x; }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace clang::io;
    unsigned DataLen = ReadUnalignedLE16(d);
    unsigned KeyLen = ReadUnalignedLE16(d);
    return std::make_pair(KeyLen, DataLen);
  }

  static std::pair<const char*, unsigned>
  ReadKey(const unsigned char* d, unsigned n) {
    assert(n >= 2 && d[n-1] == '\0');
    return std::make_pair((const char*) d, n-1);
  }

  IdentifierInfo *ReadData(const internal_key_type& k,
                           const unsigned char* d,
                           unsigned DataLen) {
    using namespace clang::io;
    unsigned RawID = ReadUnalignedLE32(d);
    bool IsInteresting = RawID & 0x01;

    // Wipe out the "is interesting" bit.
    RawID = RawID >> 1;

    IdentID ID = Reader.getGlobalIdentifierID(F, RawID);
    if (!IsInteresting) {
      // For uninteresting identifiers, just build the IdentifierInfo
      // and associate it with the persistent ID.
      IdentifierInfo *II = KnownII;
      if (!II)
        II = &Reader.getIdentifierTable().getOwn(StringRef(k.first, k.second));
      Reader.SetIdentifierInfo(ID, II);
      II->setIsFromAST();
      return II;
    }

    unsigned Bits = ReadUnalignedLE16(d);
    bool CPlusPlusOperatorKeyword = Bits & 0x01;
    Bits >>= 1;
    bool HasRevertedTokenIDToIdentifier = Bits & 0x01;
    Bits >>= 1;
    bool Poisoned = Bits & 0x01;
    Bits >>= 1;
    bool ExtensionToken = Bits & 0x01;
    Bits >>= 1;
    bool hasMacroDefinition = Bits & 0x01;
    Bits >>= 1;
    unsigned ObjCOrBuiltinID = Bits & 0x3FF;
    Bits >>= 10;

    assert(Bits == 0 && "Extra bits in the identifier?");
    DataLen -= 6;

    // Build the IdentifierInfo itself and link the identifier ID with
    // the new IdentifierInfo.
    IdentifierInfo *II = KnownII;
    if (!II)
      II = &Reader.getIdentifierTable().getOwn(StringRef(k.first, k.second));
    Reader.SetIdentifierInfo(ID, II);

    // Set or check the various bits in the IdentifierInfo structure.
    // Token IDs are read-only.
    if (HasRevertedTokenIDToIdentifier)
      II->RevertTokenIDToIdentifier();
    II->setObjCOrBuiltinID(ObjCOrBuiltinID);
    assert(II->isExtensionToken() == ExtensionToken &&
           "Incorrect extension token flag");
    (void)ExtensionToken;
    II->setIsPoisoned(Poisoned);
    assert(II->isCPlusPlusOperatorKeyword() == CPlusPlusOperatorKeyword &&
           "Incorrect C++ operator keyword flag");
    (void)CPlusPlusOperatorKeyword;

    // If this identifier is a macro, deserialize the macro
    // definition.
    if (hasMacroDefinition) {
      uint32_t Offset = ReadUnalignedLE32(d);
      Reader.SetIdentifierIsMacro(II, F, Offset);
      DataLen -= 4;
    }

    // Read all of the declarations visible at global scope with this
    // name.
    if (Reader.getContext() == 0) return II;
    if (DataLen > 0) {
      SmallVector<uint32_t, 4> DeclIDs;
      for (; DataLen > 0; DataLen -= 4)
        DeclIDs.push_back(Reader.getGlobalDeclID(F, ReadUnalignedLE32(d)));
      Reader.SetGloballyVisibleDecls(II, DeclIDs);
    }

    II->setIsFromAST();
    return II;
  }
};

} // end anonymous namespace

/// \brief The on-disk hash table used to contain information about
/// all of the identifiers in the program.
typedef OnDiskChainedHashTable<ASTIdentifierLookupTrait>
  ASTIdentifierLookupTable;

namespace {
class ASTDeclContextNameLookupTrait {
  ASTReader &Reader;
  Module &F;
  
public:
  /// \brief Pair of begin/end iterators for DeclIDs.
  ///
  /// Note that these declaration IDs are local to the module that contains this
  /// particular lookup t
  typedef std::pair<DeclID *, DeclID *> data_type;

  /// \brief Special internal key for declaration names.
  /// The hash table creates keys for comparison; we do not create
  /// a DeclarationName for the internal key to avoid deserializing types.
  struct DeclNameKey {
    DeclarationName::NameKind Kind;
    uint64_t Data;
    DeclNameKey() : Kind((DeclarationName::NameKind)0), Data(0) { }
  };

  typedef DeclarationName external_key_type;
  typedef DeclNameKey internal_key_type;

  explicit ASTDeclContextNameLookupTrait(ASTReader &Reader, 
                                         Module &F) 
    : Reader(Reader), F(F) { }

  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return a.Kind == b.Kind && a.Data == b.Data;
  }

  unsigned ComputeHash(const DeclNameKey &Key) const {
    llvm::FoldingSetNodeID ID;
    ID.AddInteger(Key.Kind);

    switch (Key.Kind) {
    case DeclarationName::Identifier:
    case DeclarationName::CXXLiteralOperatorName:
      ID.AddString(((IdentifierInfo*)Key.Data)->getName());
      break;
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      ID.AddInteger(serialization::ComputeHash(Selector(Key.Data)));
      break;
    case DeclarationName::CXXOperatorName:
      ID.AddInteger((OverloadedOperatorKind)Key.Data);
      break;
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
    case DeclarationName::CXXUsingDirective:
      break;
    }

    return ID.ComputeHash();
  }

  internal_key_type GetInternalKey(const external_key_type& Name) const {
    DeclNameKey Key;
    Key.Kind = Name.getNameKind();
    switch (Name.getNameKind()) {
    case DeclarationName::Identifier:
      Key.Data = (uint64_t)Name.getAsIdentifierInfo();
      break;
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      Key.Data = (uint64_t)Name.getObjCSelector().getAsOpaquePtr();
      break;
    case DeclarationName::CXXOperatorName:
      Key.Data = Name.getCXXOverloadedOperator();
      break;
    case DeclarationName::CXXLiteralOperatorName:
      Key.Data = (uint64_t)Name.getCXXLiteralIdentifier();
      break;
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
    case DeclarationName::CXXUsingDirective:
      Key.Data = 0;
      break;
    }

    return Key;
  }

  external_key_type GetExternalKey(const internal_key_type& Key) const {
    ASTContext *Context = Reader.getContext();
    switch (Key.Kind) {
    case DeclarationName::Identifier:
      return DeclarationName((IdentifierInfo*)Key.Data);

    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      return DeclarationName(Selector(Key.Data));

    case DeclarationName::CXXConstructorName:
      return Context->DeclarationNames.getCXXConstructorName(
               Context->getCanonicalType(Reader.getLocalType(F, Key.Data)));

    case DeclarationName::CXXDestructorName:
      return Context->DeclarationNames.getCXXDestructorName(
               Context->getCanonicalType(Reader.getLocalType(F, Key.Data)));

    case DeclarationName::CXXConversionFunctionName:
      return Context->DeclarationNames.getCXXConversionFunctionName(
               Context->getCanonicalType(Reader.getLocalType(F, Key.Data)));

    case DeclarationName::CXXOperatorName:
      return Context->DeclarationNames.getCXXOperatorName(
                                         (OverloadedOperatorKind)Key.Data);

    case DeclarationName::CXXLiteralOperatorName:
      return Context->DeclarationNames.getCXXLiteralOperatorName(
                                                     (IdentifierInfo*)Key.Data);

    case DeclarationName::CXXUsingDirective:
      return DeclarationName::getUsingDirectiveName();
    }

    llvm_unreachable("Invalid Name Kind ?");
  }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace clang::io;
    unsigned KeyLen = ReadUnalignedLE16(d);
    unsigned DataLen = ReadUnalignedLE16(d);
    return std::make_pair(KeyLen, DataLen);
  }

  internal_key_type ReadKey(const unsigned char* d, unsigned) {
    using namespace clang::io;

    DeclNameKey Key;
    Key.Kind = (DeclarationName::NameKind)*d++;
    switch (Key.Kind) {
    case DeclarationName::Identifier:
      Key.Data = (uint64_t)Reader.getLocalIdentifier(F, ReadUnalignedLE32(d));
      break;
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      Key.Data =
         (uint64_t)Reader.getLocalSelector(F, ReadUnalignedLE32(d))
                     .getAsOpaquePtr();
      break;
    case DeclarationName::CXXOperatorName:
      Key.Data = *d++; // OverloadedOperatorKind
      break;
    case DeclarationName::CXXLiteralOperatorName:
      Key.Data = (uint64_t)Reader.getLocalIdentifier(F, ReadUnalignedLE32(d));
      break;
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
    case DeclarationName::CXXUsingDirective:
      Key.Data = 0;
      break;
    }

    return Key;
  }

  data_type ReadData(internal_key_type, const unsigned char* d,
                     unsigned DataLen) {
    using namespace clang::io;
    unsigned NumDecls = ReadUnalignedLE16(d);
    DeclID *Start = (DeclID *)d;
    return std::make_pair(Start, Start + NumDecls);
  }
};

} // end anonymous namespace

/// \brief The on-disk hash table used for the DeclContext's Name lookup table.
typedef OnDiskChainedHashTable<ASTDeclContextNameLookupTrait>
  ASTDeclContextNameLookupTable;

bool ASTReader::ReadDeclContextStorage(llvm::BitstreamCursor &Cursor,
                                   const std::pair<uint64_t, uint64_t> &Offsets,
                                       DeclContextInfo &Info) {
  SavedStreamPosition SavedPosition(Cursor);
  // First the lexical decls.
  if (Offsets.first != 0) {
    Cursor.JumpToBit(Offsets.first);

    RecordData Record;
    const char *Blob;
    unsigned BlobLen;
    unsigned Code = Cursor.ReadCode();
    unsigned RecCode = Cursor.ReadRecord(Code, Record, &Blob, &BlobLen);
    if (RecCode != DECL_CONTEXT_LEXICAL) {
      Error("Expected lexical block");
      return true;
    }

    Info.LexicalDecls = reinterpret_cast<const KindDeclIDPair*>(Blob);
    Info.NumLexicalDecls = BlobLen / sizeof(KindDeclIDPair);
  } else {
    Info.LexicalDecls = 0;
    Info.NumLexicalDecls = 0;
  }

  // Now the lookup table.
  if (Offsets.second != 0) {
    Cursor.JumpToBit(Offsets.second);

    RecordData Record;
    const char *Blob;
    unsigned BlobLen;
    unsigned Code = Cursor.ReadCode();
    unsigned RecCode = Cursor.ReadRecord(Code, Record, &Blob, &BlobLen);
    if (RecCode != DECL_CONTEXT_VISIBLE) {
      Error("Expected visible lookup table block");
      return true;
    }
    Info.NameLookupTableData
      = ASTDeclContextNameLookupTable::Create(
                    (const unsigned char *)Blob + Record[0],
                    (const unsigned char *)Blob,
                    ASTDeclContextNameLookupTrait(*this, *Info.F));
  } else {
    Info.NameLookupTableData = 0;
  }

  return false;
}

void ASTReader::Error(StringRef Msg) {
  Error(diag::err_fe_pch_malformed, Msg);
}

void ASTReader::Error(unsigned DiagID,
                      StringRef Arg1, StringRef Arg2) {
  if (Diags.isDiagnosticInFlight())
    Diags.SetDelayedDiagnostic(DiagID, Arg1, Arg2);
  else
    Diag(DiagID) << Arg1 << Arg2;
}

/// \brief Tell the AST listener about the predefines buffers in the chain.
bool ASTReader::CheckPredefinesBuffers() {
  if (Listener)
    return Listener->ReadPredefinesBuffer(PCHPredefinesBuffers,
                                          ActualOriginalFileName,
                                          SuggestedPredefines,
                                          FileMgr);
  return false;
}

//===----------------------------------------------------------------------===//
// Source Manager Deserialization
//===----------------------------------------------------------------------===//

/// \brief Read the line table in the source manager block.
/// \returns true if there was an error.
bool ASTReader::ParseLineTable(Module &F,
                               SmallVectorImpl<uint64_t> &Record) {
  unsigned Idx = 0;
  LineTableInfo &LineTable = SourceMgr.getLineTable();

  // Parse the file names
  std::map<int, int> FileIDs;
  for (int I = 0, N = Record[Idx++]; I != N; ++I) {
    // Extract the file name
    unsigned FilenameLen = Record[Idx++];
    std::string Filename(&Record[Idx], &Record[Idx] + FilenameLen);
    Idx += FilenameLen;
    MaybeAddSystemRootToFilename(Filename);
    FileIDs[I] = LineTable.getLineTableFilenameID(Filename);
  }

  // Parse the line entries
  std::vector<LineEntry> Entries;
  while (Idx < Record.size()) {
    int FID = Record[Idx++];
    assert(FID >= 0 && "Serialized line entries for non-local file.");
    // Remap FileID from 1-based old view.
    FID += F.SLocEntryBaseID - 1;

    // Extract the line entries
    unsigned NumEntries = Record[Idx++];
    assert(NumEntries && "Numentries is 00000");
    Entries.clear();
    Entries.reserve(NumEntries);
    for (unsigned I = 0; I != NumEntries; ++I) {
      unsigned FileOffset = Record[Idx++];
      unsigned LineNo = Record[Idx++];
      int FilenameID = FileIDs[Record[Idx++]];
      SrcMgr::CharacteristicKind FileKind
        = (SrcMgr::CharacteristicKind)Record[Idx++];
      unsigned IncludeOffset = Record[Idx++];
      Entries.push_back(LineEntry::get(FileOffset, LineNo, FilenameID,
                                       FileKind, IncludeOffset));
    }
    LineTable.AddEntry(FID, Entries);
  }

  return false;
}

namespace {

class ASTStatData {
public:
  const ino_t ino;
  const dev_t dev;
  const mode_t mode;
  const time_t mtime;
  const off_t size;

  ASTStatData(ino_t i, dev_t d, mode_t mo, time_t m, off_t s)
    : ino(i), dev(d), mode(mo), mtime(m), size(s) {}
};

class ASTStatLookupTrait {
 public:
  typedef const char *external_key_type;
  typedef const char *internal_key_type;

  typedef ASTStatData data_type;

  static unsigned ComputeHash(const char *path) {
    return llvm::HashString(path);
  }

  static internal_key_type GetInternalKey(const char *path) { return path; }

  static bool EqualKey(internal_key_type a, internal_key_type b) {
    return strcmp(a, b) == 0;
  }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    unsigned KeyLen = (unsigned) clang::io::ReadUnalignedLE16(d);
    unsigned DataLen = (unsigned) *d++;
    return std::make_pair(KeyLen + 1, DataLen);
  }

  static internal_key_type ReadKey(const unsigned char *d, unsigned) {
    return (const char *)d;
  }

  static data_type ReadData(const internal_key_type, const unsigned char *d,
                            unsigned /*DataLen*/) {
    using namespace clang::io;

    ino_t ino = (ino_t) ReadUnalignedLE32(d);
    dev_t dev = (dev_t) ReadUnalignedLE32(d);
    mode_t mode = (mode_t) ReadUnalignedLE16(d);
    time_t mtime = (time_t) ReadUnalignedLE64(d);
    off_t size = (off_t) ReadUnalignedLE64(d);
    return data_type(ino, dev, mode, mtime, size);
  }
};

/// \brief stat() cache for precompiled headers.
///
/// This cache is very similar to the stat cache used by pretokenized
/// headers.
class ASTStatCache : public FileSystemStatCache {
  typedef OnDiskChainedHashTable<ASTStatLookupTrait> CacheTy;
  CacheTy *Cache;

  unsigned &NumStatHits, &NumStatMisses;
public:
  ASTStatCache(const unsigned char *Buckets, const unsigned char *Base,
               unsigned &NumStatHits, unsigned &NumStatMisses)
    : Cache(0), NumStatHits(NumStatHits), NumStatMisses(NumStatMisses) {
    Cache = CacheTy::Create(Buckets, Base);
  }

  ~ASTStatCache() { delete Cache; }

  LookupResult getStat(const char *Path, struct stat &StatBuf,
                       int *FileDescriptor) {
    // Do the lookup for the file's data in the AST file.
    CacheTy::iterator I = Cache->find(Path);

    // If we don't get a hit in the AST file just forward to 'stat'.
    if (I == Cache->end()) {
      ++NumStatMisses;
      return statChained(Path, StatBuf, FileDescriptor);
    }

    ++NumStatHits;
    ASTStatData Data = *I;

    StatBuf.st_ino = Data.ino;
    StatBuf.st_dev = Data.dev;
    StatBuf.st_mtime = Data.mtime;
    StatBuf.st_mode = Data.mode;
    StatBuf.st_size = Data.size;
    return CacheExists;
  }
};
} // end anonymous namespace


/// \brief Read a source manager block
ASTReader::ASTReadResult ASTReader::ReadSourceManagerBlock(Module &F) {
  using namespace SrcMgr;

  llvm::BitstreamCursor &SLocEntryCursor = F.SLocEntryCursor;

  // Set the source-location entry cursor to the current position in
  // the stream. This cursor will be used to read the contents of the
  // source manager block initially, and then lazily read
  // source-location entries as needed.
  SLocEntryCursor = F.Stream;

  // The stream itself is going to skip over the source manager block.
  if (F.Stream.SkipBlock()) {
    Error("malformed block record in AST file");
    return Failure;
  }

  // Enter the source manager block.
  if (SLocEntryCursor.EnterSubBlock(SOURCE_MANAGER_BLOCK_ID)) {
    Error("malformed source manager block record in AST file");
    return Failure;
  }

  RecordData Record;
  while (true) {
    unsigned Code = SLocEntryCursor.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (SLocEntryCursor.ReadBlockEnd()) {
        Error("error at end of Source Manager block in AST file");
        return Failure;
      }
      return Success;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      SLocEntryCursor.ReadSubBlockID();
      if (SLocEntryCursor.SkipBlock()) {
        Error("malformed block record in AST file");
        return Failure;
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      SLocEntryCursor.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    const char *BlobStart;
    unsigned BlobLen;
    Record.clear();
    switch (SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;

    case SM_SLOC_FILE_ENTRY:
    case SM_SLOC_BUFFER_ENTRY:
    case SM_SLOC_EXPANSION_ENTRY:
      // Once we hit one of the source location entries, we're done.
      return Success;
    }
  }
}

/// \brief If a header file is not found at the path that we expect it to be
/// and the PCH file was moved from its original location, try to resolve the
/// file by assuming that header+PCH were moved together and the header is in
/// the same place relative to the PCH.
static std::string
resolveFileRelativeToOriginalDir(const std::string &Filename,
                                 const std::string &OriginalDir,
                                 const std::string &CurrDir) {
  assert(OriginalDir != CurrDir &&
         "No point trying to resolve the file if the PCH dir didn't change");
  using namespace llvm::sys;
  llvm::SmallString<128> filePath(Filename);
  fs::make_absolute(filePath);
  assert(path::is_absolute(OriginalDir));
  llvm::SmallString<128> currPCHPath(CurrDir);

  path::const_iterator fileDirI = path::begin(path::parent_path(filePath)),
                       fileDirE = path::end(path::parent_path(filePath));
  path::const_iterator origDirI = path::begin(OriginalDir),
                       origDirE = path::end(OriginalDir);
  // Skip the common path components from filePath and OriginalDir.
  while (fileDirI != fileDirE && origDirI != origDirE &&
         *fileDirI == *origDirI) {
    ++fileDirI;
    ++origDirI;
  }
  for (; origDirI != origDirE; ++origDirI)
    path::append(currPCHPath, "..");
  path::append(currPCHPath, fileDirI, fileDirE);
  path::append(currPCHPath, path::filename(Filename));
  return currPCHPath.str();
}

/// \brief Read in the source location entry with the given ID.
ASTReader::ASTReadResult ASTReader::ReadSLocEntryRecord(int ID) {
  if (ID == 0)
    return Success;

  if (unsigned(-ID) - 2 >= getTotalNumSLocs() || ID > 0) {
    Error("source location entry ID out-of-range for AST file");
    return Failure;
  }

  Module *F = GlobalSLocEntryMap.find(-ID)->second;
  F->SLocEntryCursor.JumpToBit(F->SLocEntryOffsets[ID - F->SLocEntryBaseID]);
  llvm::BitstreamCursor &SLocEntryCursor = F->SLocEntryCursor;
  unsigned BaseOffset = F->SLocEntryBaseOffset;

  ++NumSLocEntriesRead;
  unsigned Code = SLocEntryCursor.ReadCode();
  if (Code == llvm::bitc::END_BLOCK ||
      Code == llvm::bitc::ENTER_SUBBLOCK ||
      Code == llvm::bitc::DEFINE_ABBREV) {
    Error("incorrectly-formatted source location entry in AST file");
    return Failure;
  }

  RecordData Record;
  const char *BlobStart;
  unsigned BlobLen;
  switch (SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
  default:
    Error("incorrectly-formatted source location entry in AST file");
    return Failure;

  case SM_SLOC_FILE_ENTRY: {
    std::string Filename(BlobStart, BlobStart + BlobLen);
    MaybeAddSystemRootToFilename(Filename);
    const FileEntry *File = FileMgr.getFile(Filename);
    if (File == 0 && !OriginalDir.empty() && !CurrentDir.empty() &&
        OriginalDir != CurrentDir) {
      std::string resolved = resolveFileRelativeToOriginalDir(Filename,
                                                              OriginalDir,
                                                              CurrentDir);
      if (!resolved.empty())
        File = FileMgr.getFile(resolved);
    }
    if (File == 0)
      File = FileMgr.getVirtualFile(Filename, (off_t)Record[4],
                                    (time_t)Record[5]);
    if (File == 0) {
      std::string ErrorStr = "could not find file '";
      ErrorStr += Filename;
      ErrorStr += "' referenced by AST file";
      Error(ErrorStr.c_str());
      return Failure;
    }

    if (Record.size() < 6) {
      Error("source location entry is incorrect");
      return Failure;
    }

    if (!DisableValidation &&
        ((off_t)Record[4] != File->getSize()
#if !defined(LLVM_ON_WIN32)
        // In our regression testing, the Windows file system seems to
        // have inconsistent modification times that sometimes
        // erroneously trigger this error-handling path.
         || (time_t)Record[5] != File->getModificationTime()
#endif
        )) {
      Error(diag::err_fe_pch_file_modified, Filename);
      return Failure;
    }

    SourceLocation IncludeLoc = ReadSourceLocation(*F, Record[1]);
    if (IncludeLoc.isInvalid() && F->Kind != MK_MainFile) {
      // This is the module's main file.
      IncludeLoc = getImportLocation(F);
    }
    FileID FID = SourceMgr.createFileID(File, IncludeLoc,
                                        (SrcMgr::CharacteristicKind)Record[2],
                                        ID, BaseOffset + Record[0]);
    if (Record[3])
      const_cast<SrcMgr::FileInfo&>(SourceMgr.getSLocEntry(FID).getFile())
        .setHasLineDirectives();
    
    break;
  }

  case SM_SLOC_BUFFER_ENTRY: {
    const char *Name = BlobStart;
    unsigned Offset = Record[0];
    unsigned Code = SLocEntryCursor.ReadCode();
    Record.clear();
    unsigned RecCode
      = SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen);

    if (RecCode != SM_SLOC_BUFFER_BLOB) {
      Error("AST record has invalid code");
      return Failure;
    }

    llvm::MemoryBuffer *Buffer
    = llvm::MemoryBuffer::getMemBuffer(StringRef(BlobStart, BlobLen - 1),
                                       Name);
    FileID BufferID = SourceMgr.createFileIDForMemBuffer(Buffer, ID,
                                                         BaseOffset + Offset);

    if (strcmp(Name, "<built-in>") == 0) {
      PCHPredefinesBlock Block = {
        BufferID,
        StringRef(BlobStart, BlobLen - 1)
      };
      PCHPredefinesBuffers.push_back(Block);
    }

    break;
  }

  case SM_SLOC_EXPANSION_ENTRY: {
    SourceLocation SpellingLoc = ReadSourceLocation(*F, Record[1]);
    SourceMgr.createExpansionLoc(SpellingLoc,
                                     ReadSourceLocation(*F, Record[2]),
                                     ReadSourceLocation(*F, Record[3]),
                                     Record[4],
                                     ID,
                                     BaseOffset + Record[0]);
    break;
  }
  }

  return Success;
}

/// \brief Find the location where the module F is imported.
SourceLocation ASTReader::getImportLocation(Module *F) {
  if (F->ImportLoc.isValid())
    return F->ImportLoc;
  
  // Otherwise we have a PCH. It's considered to be "imported" at the first
  // location of its includer.
  if (F->ImportedBy.empty() || !F->ImportedBy[0]) {
    // Main file is the importer. We assume that it is the first entry in the
    // entry table. We can't ask the manager, because at the time of PCH loading
    // the main file entry doesn't exist yet.
    // The very first entry is the invalid instantiation loc, which takes up
    // offsets 0 and 1.
    return SourceLocation::getFromRawEncoding(2U);
  }
  //return F->Loaders[0]->FirstLoc;
  return F->ImportedBy[0]->FirstLoc;
}

/// ReadBlockAbbrevs - Enter a subblock of the specified BlockID with the
/// specified cursor.  Read the abbreviations that are at the top of the block
/// and then leave the cursor pointing into the block.
bool ASTReader::ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor,
                                 unsigned BlockID) {
  if (Cursor.EnterSubBlock(BlockID)) {
    Error("malformed block record in AST file");
    return Failure;
  }

  while (true) {
    uint64_t Offset = Cursor.GetCurrentBitNo();
    unsigned Code = Cursor.ReadCode();

    // We expect all abbrevs to be at the start of the block.
    if (Code != llvm::bitc::DEFINE_ABBREV) {
      Cursor.JumpToBit(Offset);
      return false;
    }
    Cursor.ReadAbbrevRecord();
  }
}

void ASTReader::ReadMacroRecord(Module &F, uint64_t Offset) {
  assert(PP && "Forgot to set Preprocessor ?");
  llvm::BitstreamCursor &Stream = F.MacroCursor;

  // Keep track of where we are in the stream, then jump back there
  // after reading this macro.
  SavedStreamPosition SavedPosition(Stream);

  Stream.JumpToBit(Offset);
  RecordData Record;
  SmallVector<IdentifierInfo*, 16> MacroArgs;
  MacroInfo *Macro = 0;

  while (true) {
    unsigned Code = Stream.ReadCode();
    switch (Code) {
    case llvm::bitc::END_BLOCK:
      return;

    case llvm::bitc::ENTER_SUBBLOCK:
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock()) {
        Error("malformed block record in AST file");
        return;
      }
      continue;

    case llvm::bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      continue;
    default: break;
    }

    // Read a record.
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    Record.clear();
    PreprocessorRecordTypes RecType =
      (PreprocessorRecordTypes)Stream.ReadRecord(Code, Record, BlobStart,
                                                 BlobLen);
    switch (RecType) {
    case PP_MACRO_OBJECT_LIKE:
    case PP_MACRO_FUNCTION_LIKE: {
      // If we already have a macro, that means that we've hit the end
      // of the definition of the macro we were looking for. We're
      // done.
      if (Macro)
        return;

      IdentifierInfo *II = getLocalIdentifier(F, Record[0]);
      if (II == 0) {
        Error("macro must have a name in AST file");
        return;
      }
      SourceLocation Loc = ReadSourceLocation(F, Record[1]);
      bool isUsed = Record[2];

      MacroInfo *MI = PP->AllocateMacroInfo(Loc);
      MI->setIsUsed(isUsed);
      MI->setIsFromAST();

      unsigned NextIndex = 3;
      if (RecType == PP_MACRO_FUNCTION_LIKE) {
        // Decode function-like macro info.
        bool isC99VarArgs = Record[3];
        bool isGNUVarArgs = Record[4];
        MacroArgs.clear();
        unsigned NumArgs = Record[5];
        NextIndex = 6 + NumArgs;
        for (unsigned i = 0; i != NumArgs; ++i)
          MacroArgs.push_back(getLocalIdentifier(F, Record[6+i]));

        // Install function-like macro info.
        MI->setIsFunctionLike();
        if (isC99VarArgs) MI->setIsC99Varargs();
        if (isGNUVarArgs) MI->setIsGNUVarargs();
        MI->setArgumentList(MacroArgs.data(), MacroArgs.size(),
                            PP->getPreprocessorAllocator());
      }

      // Finally, install the macro.
      PP->setMacroInfo(II, MI);

      // Remember that we saw this macro last so that we add the tokens that
      // form its body to it.
      Macro = MI;

      if (NextIndex + 1 == Record.size() && PP->getPreprocessingRecord()) {
        // We have a macro definition. Load it now.
        PP->getPreprocessingRecord()->RegisterMacroDefinition(Macro,
              getLocalMacroDefinition(F, Record[NextIndex]));
      }

      ++NumMacrosRead;
      break;
    }

    case PP_TOKEN: {
      // If we see a TOKEN before a PP_MACRO_*, then the file is
      // erroneous, just pretend we didn't see this.
      if (Macro == 0) break;

      Token Tok;
      Tok.startToken();
      Tok.setLocation(ReadSourceLocation(F, Record[0]));
      Tok.setLength(Record[1]);
      if (IdentifierInfo *II = getLocalIdentifier(F, Record[2]))
        Tok.setIdentifierInfo(II);
      Tok.setKind((tok::TokenKind)Record[3]);
      Tok.setFlag((Token::TokenFlags)Record[4]);
      Macro->AddTokenToBody(Tok);
      break;
    }
  }
  }
  
  return;
}

PreprocessedEntity *ASTReader::LoadPreprocessedEntity(Module &F) {
  assert(PP && "Forgot to set Preprocessor ?");
  unsigned Code = F.PreprocessorDetailCursor.ReadCode();
  switch (Code) {
  case llvm::bitc::END_BLOCK:
    return 0;
    
  case llvm::bitc::ENTER_SUBBLOCK:
    Error("unexpected subblock record in preprocessor detail block");
    return 0;
      
  case llvm::bitc::DEFINE_ABBREV:
    Error("unexpected abbrevation record in preprocessor detail block");
    return 0;
      
  default:
    break;
  }

  if (!PP->getPreprocessingRecord()) {
    Error("no preprocessing record");
    return 0;
  }
  
  // Read the record.
  PreprocessingRecord &PPRec = *PP->getPreprocessingRecord();
  const char *BlobStart = 0;
  unsigned BlobLen = 0;
  RecordData Record;
  PreprocessorDetailRecordTypes RecType =
    (PreprocessorDetailRecordTypes)F.PreprocessorDetailCursor.ReadRecord(
                                             Code, Record, BlobStart, BlobLen);
  switch (RecType) {
  case PPD_MACRO_EXPANSION: {
    PreprocessedEntityID GlobalID = getGlobalPreprocessedEntityID(F, Record[0]);
    if (PreprocessedEntity *PE = PPRec.getLoadedPreprocessedEntity(GlobalID-1))
      return PE;
    
    MacroExpansion *ME =
      new (PPRec) MacroExpansion(getLocalIdentifier(F, Record[3]),
                                 SourceRange(ReadSourceLocation(F, Record[1]),
                                             ReadSourceLocation(F, Record[2])),
                                 getLocalMacroDefinition(F, Record[4]));
    PPRec.setLoadedPreallocatedEntity(GlobalID - 1, ME);
    return ME;
  }
      
  case PPD_MACRO_DEFINITION: {
    PreprocessedEntityID GlobalID = getGlobalPreprocessedEntityID(F, Record[0]);
    if (PreprocessedEntity *PE = PPRec.getLoadedPreprocessedEntity(GlobalID-1))
      return PE;

    unsigned MacroDefID = getGlobalMacroDefinitionID(F, Record[1]);
    if (MacroDefID > MacroDefinitionsLoaded.size()) {
      Error("out-of-bounds macro definition record");
      return 0;
    }
    
    // Decode the identifier info and then check again; if the macro is
    // still defined and associated with the identifier,
    IdentifierInfo *II = getLocalIdentifier(F, Record[4]);
    if (!MacroDefinitionsLoaded[MacroDefID - 1]) {
      MacroDefinition *MD
        = new (PPRec) MacroDefinition(II,
                                      ReadSourceLocation(F, Record[5]),
                                      SourceRange(
                                            ReadSourceLocation(F, Record[2]),
                                            ReadSourceLocation(F, Record[3])));
      
      PPRec.setLoadedPreallocatedEntity(GlobalID - 1, MD);
      MacroDefinitionsLoaded[MacroDefID - 1] = MD;
      
      if (DeserializationListener)
        DeserializationListener->MacroDefinitionRead(MacroDefID, MD);
    }
    
    return MacroDefinitionsLoaded[MacroDefID - 1];
  }
      
  case PPD_INCLUSION_DIRECTIVE: {
    PreprocessedEntityID GlobalID = getGlobalPreprocessedEntityID(F, Record[0]);
    if (PreprocessedEntity *PE = PPRec.getLoadedPreprocessedEntity(GlobalID-1))
      return PE;
    
    const char *FullFileNameStart = BlobStart + Record[3];
    const FileEntry *File
      = PP->getFileManager().getFile(StringRef(FullFileNameStart,
                                               BlobLen - Record[3]));
    
    // FIXME: Stable encoding
    InclusionDirective::InclusionKind Kind
      = static_cast<InclusionDirective::InclusionKind>(Record[5]);
    InclusionDirective *ID
      = new (PPRec) InclusionDirective(PPRec, Kind,
                                       StringRef(BlobStart, Record[3]),
                                       Record[4],
                                       File,
                                 SourceRange(ReadSourceLocation(F, Record[1]),
                                             ReadSourceLocation(F, Record[2])));
    PPRec.setLoadedPreallocatedEntity(GlobalID - 1, ID);
    return ID;
  }
  }
  
  Error("invalid offset in preprocessor detail block");
  return 0;
}

PreprocessedEntityID 
ASTReader::getGlobalPreprocessedEntityID(Module &M, unsigned LocalID) {
  ContinuousRangeMap<uint32_t, int, 2>::iterator 
    I = M.PreprocessedEntityRemap.find(LocalID - NUM_PREDEF_PP_ENTITY_IDS);
  assert(I != M.PreprocessedEntityRemap.end() 
         && "Invalid index into preprocessed entity index remap");
  
  return LocalID + I->second;
}

namespace {
  /// \brief Trait class used to search the on-disk hash table containing all of
  /// the header search information.
  ///
  /// The on-disk hash table contains a mapping from each header path to 
  /// information about that header (how many times it has been included, its
  /// controlling macro, etc.). Note that we actually hash based on the 
  /// filename, and support "deep" comparisons of file names based on current
  /// inode numbers, so that the search can cope with non-normalized path names
  /// and symlinks.
  class HeaderFileInfoTrait {
    ASTReader &Reader;
    Module &M;
    HeaderSearch *HS;
    const char *FrameworkStrings;
    const char *SearchPath;
    struct stat SearchPathStatBuf;
    llvm::Optional<int> SearchPathStatResult;
    
    int StatSimpleCache(const char *Path, struct stat *StatBuf) {
      if (Path == SearchPath) {
        if (!SearchPathStatResult)
          SearchPathStatResult = stat(Path, &SearchPathStatBuf);
        
        *StatBuf = SearchPathStatBuf;
        return *SearchPathStatResult;
      }
      
      return stat(Path, StatBuf);
    }
    
  public:
    typedef const char *external_key_type;
    typedef const char *internal_key_type;
    
    typedef HeaderFileInfo data_type;
    
    HeaderFileInfoTrait(ASTReader &Reader, Module &M, HeaderSearch *HS,
                        const char *FrameworkStrings,
                        const char *SearchPath = 0) 
      : Reader(Reader), M(M), HS(HS), FrameworkStrings(FrameworkStrings), 
        SearchPath(SearchPath) { }
    
    static unsigned ComputeHash(const char *path) {
      return llvm::HashString(llvm::sys::path::filename(path));
    }
    
    static internal_key_type GetInternalKey(const char *path) { return path; }
    
    bool EqualKey(internal_key_type a, internal_key_type b) {
      if (strcmp(a, b) == 0)
        return true;
      
      if (llvm::sys::path::filename(a) != llvm::sys::path::filename(b))
        return false;
      
      // The file names match, but the path names don't. stat() the files to
      // see if they are the same.      
      struct stat StatBufA, StatBufB;
      if (StatSimpleCache(a, &StatBufA) || StatSimpleCache(b, &StatBufB))
        return false;
      
      return StatBufA.st_ino == StatBufB.st_ino;
    }
    
    static std::pair<unsigned, unsigned>
    ReadKeyDataLength(const unsigned char*& d) {
      unsigned KeyLen = (unsigned) clang::io::ReadUnalignedLE16(d);
      unsigned DataLen = (unsigned) *d++;
      return std::make_pair(KeyLen + 1, DataLen);
    }
    
    static internal_key_type ReadKey(const unsigned char *d, unsigned) {
      return (const char *)d;
    }
    
    data_type ReadData(const internal_key_type, const unsigned char *d,
                       unsigned DataLen) {
      const unsigned char *End = d + DataLen;
      using namespace clang::io;
      HeaderFileInfo HFI;
      unsigned Flags = *d++;
      HFI.isImport = (Flags >> 5) & 0x01;
      HFI.isPragmaOnce = (Flags >> 4) & 0x01;
      HFI.DirInfo = (Flags >> 2) & 0x03;
      HFI.Resolved = (Flags >> 1) & 0x01;
      HFI.IndexHeaderMapHeader = Flags & 0x01;
      HFI.NumIncludes = ReadUnalignedLE16(d);
      HFI.ControllingMacroID = Reader.getGlobalDeclID(M, ReadUnalignedLE32(d));
      if (unsigned FrameworkOffset = ReadUnalignedLE32(d)) {
        // The framework offset is 1 greater than the actual offset, 
        // since 0 is used as an indicator for "no framework name".
        StringRef FrameworkName(FrameworkStrings + FrameworkOffset - 1);
        HFI.Framework = HS->getUniqueFrameworkName(FrameworkName);
      }
      
      assert(End == d && "Wrong data length in HeaderFileInfo deserialization");
      (void)End;
            
      // This HeaderFileInfo was externally loaded.
      HFI.External = true;
      return HFI;
    }
  };
}

/// \brief The on-disk hash table used for the global method pool.
typedef OnDiskChainedHashTable<HeaderFileInfoTrait>
  HeaderFileInfoLookupTable;

void ASTReader::SetIdentifierIsMacro(IdentifierInfo *II, Module &F,
                                     uint64_t LocalOffset) {
  // Note that this identifier has a macro definition.
  II->setHasMacroDefinition(true);
  
  // Adjust the offset to a global offset.
  UnreadMacroRecordOffsets[II] = F.GlobalBitOffset + LocalOffset;
}

void ASTReader::ReadDefinedMacros() {
  for (ModuleReverseIterator I = ModuleMgr.rbegin(),
      E = ModuleMgr.rend(); I != E; ++I) {
    llvm::BitstreamCursor &MacroCursor = (*I)->MacroCursor;

    // If there was no preprocessor block, skip this file.
    if (!MacroCursor.getBitStreamReader())
      continue;

    llvm::BitstreamCursor Cursor = MacroCursor;
    Cursor.JumpToBit((*I)->MacroStartOffset);

    RecordData Record;
    while (true) {
      unsigned Code = Cursor.ReadCode();
      if (Code == llvm::bitc::END_BLOCK)
        break;

      if (Code == llvm::bitc::ENTER_SUBBLOCK) {
        // No known subblocks, always skip them.
        Cursor.ReadSubBlockID();
        if (Cursor.SkipBlock()) {
          Error("malformed block record in AST file");
          return;
        }
        continue;
      }

      if (Code == llvm::bitc::DEFINE_ABBREV) {
        Cursor.ReadAbbrevRecord();
        continue;
      }

      // Read a record.
      const char *BlobStart;
      unsigned BlobLen;
      Record.clear();
      switch (Cursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
      default:  // Default behavior: ignore.
        break;

      case PP_MACRO_OBJECT_LIKE:
      case PP_MACRO_FUNCTION_LIKE:
        getLocalIdentifier(**I, Record[0]);
        break;

      case PP_TOKEN:
        // Ignore tokens.
        break;
      }
    }
  }
  
  // Drain the unread macro-record offsets map.
  while (!UnreadMacroRecordOffsets.empty())
    LoadMacroDefinition(UnreadMacroRecordOffsets.begin());
}

void ASTReader::LoadMacroDefinition(
                     llvm::DenseMap<IdentifierInfo *, uint64_t>::iterator Pos) {
  assert(Pos != UnreadMacroRecordOffsets.end() && "Unknown macro definition");
  uint64_t Offset = Pos->second;
  UnreadMacroRecordOffsets.erase(Pos);
  
  RecordLocation Loc = getLocalBitOffset(Offset);
  ReadMacroRecord(*Loc.F, Loc.Offset);
}

void ASTReader::LoadMacroDefinition(IdentifierInfo *II) {
  llvm::DenseMap<IdentifierInfo *, uint64_t>::iterator Pos
    = UnreadMacroRecordOffsets.find(II);
  LoadMacroDefinition(Pos);
}

MacroDefinition *ASTReader::getMacroDefinition(MacroID ID) {
  if (ID == 0 || ID > MacroDefinitionsLoaded.size())
    return 0;

  if (!MacroDefinitionsLoaded[ID - 1]) {
    GlobalMacroDefinitionMapType::iterator I =GlobalMacroDefinitionMap.find(ID);
    assert(I != GlobalMacroDefinitionMap.end() && 
           "Corrupted global macro definition map");
    Module &M = *I->second;
    unsigned Index = ID - 1 - M.BaseMacroDefinitionID;
    SavedStreamPosition SavedPosition(M.PreprocessorDetailCursor);  
    M.PreprocessorDetailCursor.JumpToBit(M.MacroDefinitionOffsets[Index]);
    LoadPreprocessedEntity(M);
  }

  return MacroDefinitionsLoaded[ID - 1];
}

const FileEntry *ASTReader::getFileEntry(StringRef filenameStrRef) {
  std::string Filename = filenameStrRef;
  MaybeAddSystemRootToFilename(Filename);
  const FileEntry *File = FileMgr.getFile(Filename);
  if (File == 0 && !OriginalDir.empty() && !CurrentDir.empty() &&
      OriginalDir != CurrentDir) {
    std::string resolved = resolveFileRelativeToOriginalDir(Filename,
                                                            OriginalDir,
                                                            CurrentDir);
    if (!resolved.empty())
      File = FileMgr.getFile(resolved);
  }

  return File;
}

MacroID ASTReader::getGlobalMacroDefinitionID(Module &M, unsigned LocalID) {
  if (LocalID < NUM_PREDEF_MACRO_IDS)
    return LocalID;
  
  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = M.MacroDefinitionRemap.find(LocalID - NUM_PREDEF_MACRO_IDS);
  assert(I != M.MacroDefinitionRemap.end() && 
         "Invalid index into macro definition ID remap");
  
  return LocalID + I->second;
}

/// \brief If we are loading a relocatable PCH file, and the filename is
/// not an absolute path, add the system root to the beginning of the file
/// name.
void ASTReader::MaybeAddSystemRootToFilename(std::string &Filename) {
  // If this is not a relocatable PCH file, there's nothing to do.
  if (!RelocatablePCH)
    return;

  if (Filename.empty() || llvm::sys::path::is_absolute(Filename))
    return;

  if (isysroot.empty()) {
    // If no system root was given, default to '/'
    Filename.insert(Filename.begin(), '/');
    return;
  }

  unsigned Length = isysroot.size();
  if (isysroot[Length - 1] != '/')
    Filename.insert(Filename.begin(), '/');

  Filename.insert(Filename.begin(), isysroot.begin(), isysroot.end());
}

ASTReader::ASTReadResult
ASTReader::ReadASTBlock(Module &F) {
  llvm::BitstreamCursor &Stream = F.Stream;

  if (Stream.EnterSubBlock(AST_BLOCK_ID)) {
    Error("malformed block record in AST file");
    return Failure;
  }

  // Read all of the records and blocks for the ASt file.
  RecordData Record;
  bool First = true;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Error("error at end of module block in AST file");
        return Failure;
      }

      return Success;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      case DECLTYPES_BLOCK_ID:
        // We lazily load the decls block, but we want to set up the
        // DeclsCursor cursor to point into it.  Clone our current bitcode
        // cursor to it, enter the block and read the abbrevs in that block.
        // With the main cursor, we just skip over it.
        F.DeclsCursor = Stream;
        if (Stream.SkipBlock() ||  // Skip with the main cursor.
            // Read the abbrevs.
            ReadBlockAbbrevs(F.DeclsCursor, DECLTYPES_BLOCK_ID)) {
          Error("malformed block record in AST file");
          return Failure;
        }
        break;

      case DECL_UPDATES_BLOCK_ID:
        if (Stream.SkipBlock()) {
          Error("malformed block record in AST file");
          return Failure;
        }
        break;

      case PREPROCESSOR_BLOCK_ID:
        F.MacroCursor = Stream;
        if (PP)
          PP->setExternalSource(this);

        if (Stream.SkipBlock() ||
            ReadBlockAbbrevs(F.MacroCursor, PREPROCESSOR_BLOCK_ID)) {
          Error("malformed block record in AST file");
          return Failure;
        }
        F.MacroStartOffset = F.MacroCursor.GetCurrentBitNo();
        break;

      case PREPROCESSOR_DETAIL_BLOCK_ID:
        F.PreprocessorDetailCursor = Stream;
        if (Stream.SkipBlock() ||
            ReadBlockAbbrevs(F.PreprocessorDetailCursor, 
                             PREPROCESSOR_DETAIL_BLOCK_ID)) {
          Error("malformed preprocessor detail record in AST file");
          return Failure;
        }
        F.PreprocessorDetailStartOffset
          = F.PreprocessorDetailCursor.GetCurrentBitNo();
        break;
        
      case SOURCE_MANAGER_BLOCK_ID:
        switch (ReadSourceManagerBlock(F)) {
        case Success:
          break;

        case Failure:
          Error("malformed source manager block in AST file");
          return Failure;

        case IgnorePCH:
          return IgnorePCH;
        }
        break;
      }
      First = false;
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read and process a record.
    Record.clear();
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    switch ((ASTRecordTypes)Stream.ReadRecord(Code, Record,
                                              &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;

    case METADATA: {
      if (Record[0] != VERSION_MAJOR && !DisableValidation) {
        Diag(Record[0] < VERSION_MAJOR? diag::warn_pch_version_too_old
                                           : diag::warn_pch_version_too_new);
        return IgnorePCH;
      }

      RelocatablePCH = Record[4];
      if (Listener) {
        std::string TargetTriple(BlobStart, BlobLen);
        if (Listener->ReadTargetTriple(TargetTriple))
          return IgnorePCH;
      }
      break;
    }

    case CHAINED_METADATA: {
      if (!First) {
        Error("CHAINED_METADATA is not first record in block");
        return Failure;
      }
      if (Record[0] != VERSION_MAJOR && !DisableValidation) {
        Diag(Record[0] < VERSION_MAJOR? diag::warn_pch_version_too_old
                                           : diag::warn_pch_version_too_new);
        return IgnorePCH;
      }

      // Load the chained file, which is always a PCH file.
      // FIXME: This could end up being a module.
      switch(ReadASTCore(StringRef(BlobStart, BlobLen), MK_PCH)) {
      case Failure: return Failure;
        // If we have to ignore the dependency, we'll have to ignore this too.
      case IgnorePCH: return IgnorePCH;
      case Success: break;
      }     
      break;
    }

    case TYPE_OFFSET: {
      if (F.LocalNumTypes != 0) {
        Error("duplicate TYPE_OFFSET record in AST file");
        return Failure;
      }
      F.TypeOffsets = (const uint32_t *)BlobStart;
      F.LocalNumTypes = Record[0];
      unsigned LocalBaseTypeIndex = Record[1];
      F.BaseTypeIndex = getTotalNumTypes();
        
      if (F.LocalNumTypes > 0) {
        // Introduce the global -> local mapping for types within this module.
        GlobalTypeMap.insert(std::make_pair(getTotalNumTypes(), &F));
        
        // Introduce the local -> global mapping for types within this module.
        F.TypeRemap.insert(std::make_pair(LocalBaseTypeIndex, 
                             F.BaseTypeIndex - LocalBaseTypeIndex));
        
        TypesLoaded.resize(TypesLoaded.size() + F.LocalNumTypes);
      }
      break;
    }
        
    case DECL_OFFSET: {
      if (F.LocalNumDecls != 0) {
        Error("duplicate DECL_OFFSET record in AST file");
        return Failure;
      }
      F.DeclOffsets = (const uint32_t *)BlobStart;
      F.LocalNumDecls = Record[0];
      unsigned LocalBaseDeclID = Record[1];
      F.BaseDeclID = getTotalNumDecls();
        
      if (F.LocalNumDecls > 0) {
        // Introduce the global -> local mapping for declarations within this 
        // module.
        GlobalDeclMap.insert(std::make_pair(getTotalNumDecls() + 1, &F));
        
        // Introduce the local -> global mapping for declarations within this
        // module.
        F.DeclRemap.insert(std::make_pair(LocalBaseDeclID, 
                                          F.BaseDeclID - LocalBaseDeclID));
        
        DeclsLoaded.resize(DeclsLoaded.size() + F.LocalNumDecls);
      }
      break;
    }
        
    case TU_UPDATE_LEXICAL: {
      DeclContextInfo Info = {
        &F,
        /* No visible information */ 0,
        reinterpret_cast<const KindDeclIDPair *>(BlobStart),
        static_cast<unsigned int>(BlobLen / sizeof(KindDeclIDPair))
      };
      DeclContextOffsets[Context ? Context->getTranslationUnitDecl() : 0]
        .push_back(Info);
      break;
    }

    case UPDATE_VISIBLE: {
      unsigned Idx = 0;
      serialization::DeclID ID = ReadDeclID(F, Record, Idx);
      void *Table = ASTDeclContextNameLookupTable::Create(
                        (const unsigned char *)BlobStart + Record[Idx++],
                        (const unsigned char *)BlobStart,
                        ASTDeclContextNameLookupTrait(*this, F));
      // FIXME: Complete hack to check for the TU
      if (ID == (*(ModuleMgr.end() - 1))->BaseDeclID + 1 && Context) { // Is it the TU?
        DeclContextInfo Info = {
          &F, Table, /* No lexical information */ 0, 0
        };
        DeclContextOffsets[Context->getTranslationUnitDecl()].push_back(Info);
      } else
        PendingVisibleUpdates[ID].push_back(std::make_pair(Table, &F));
      break;
    }

    case REDECLS_UPDATE_LATEST: {
      assert(Record.size() % 2 == 0 && "Expected pairs of DeclIDs");
      for (unsigned i = 0, e = Record.size(); i < e; /* in loop */) {
        DeclID First = ReadDeclID(F, Record, i);
        DeclID Latest = ReadDeclID(F, Record, i);
        FirstLatestDeclIDs[First] = Latest;
      }
      break;
    }

    case LANGUAGE_OPTIONS:
      if (ParseLanguageOptions(Record) && !DisableValidation)
        return IgnorePCH;
      break;

    case IDENTIFIER_TABLE:
      F.IdentifierTableData = BlobStart;
      if (Record[0]) {
        F.IdentifierLookupTable
          = ASTIdentifierLookupTable::Create(
                       (const unsigned char *)F.IdentifierTableData + Record[0],
                       (const unsigned char *)F.IdentifierTableData,
                       ASTIdentifierLookupTrait(*this, F));
        if (PP) {
          PP->getIdentifierTable().setExternalIdentifierLookup(this);
          PP->getHeaderSearchInfo().SetExternalLookup(this);
        }
      }
      break;

    case IDENTIFIER_OFFSET: {
      if (F.LocalNumIdentifiers != 0) {
        Error("duplicate IDENTIFIER_OFFSET record in AST file");
        return Failure;
      }
      F.IdentifierOffsets = (const uint32_t *)BlobStart;
      F.LocalNumIdentifiers = Record[0];
      unsigned LocalBaseIdentifierID = Record[1];
      F.BaseIdentifierID = getTotalNumIdentifiers();
        
      if (F.LocalNumIdentifiers > 0) {
        // Introduce the global -> local mapping for identifiers within this
        // module.
        GlobalIdentifierMap.insert(std::make_pair(getTotalNumIdentifiers() + 1, 
                                                  &F));
        
        // Introduce the local -> global mapping for identifiers within this
        // module.
        F.IdentifierRemap.insert(
                            std::make_pair(LocalBaseIdentifierID,
                              F.BaseIdentifierID - LocalBaseIdentifierID));
        
        IdentifiersLoaded.resize(IdentifiersLoaded.size() 
                                 + F.LocalNumIdentifiers);
      }
      break;
    }
        
    case EXTERNAL_DEFINITIONS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        ExternalDefinitions.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case SPECIAL_TYPES:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        SpecialTypes.push_back(getGlobalTypeID(F, Record[I]));
      break;

    case STATISTICS:
      TotalNumStatements += Record[0];
      TotalNumMacros += Record[1];
      TotalLexicalDeclContexts += Record[2];
      TotalVisibleDeclContexts += Record[3];
      break;

    case UNUSED_FILESCOPED_DECLS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        UnusedFileScopedDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case DELEGATING_CTORS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        DelegatingCtorDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case WEAK_UNDECLARED_IDENTIFIERS:
      if (Record.size() % 4 != 0) {
        Error("invalid weak identifiers record");
        return Failure;
      }
        
      // FIXME: Ignore weak undeclared identifiers from non-original PCH 
      // files. This isn't the way to do it :)
      WeakUndeclaredIdentifiers.clear();
        
      // Translate the weak, undeclared identifiers into global IDs.
      for (unsigned I = 0, N = Record.size(); I < N; /* in loop */) {
        WeakUndeclaredIdentifiers.push_back(
          getGlobalIdentifierID(F, Record[I++]));
        WeakUndeclaredIdentifiers.push_back(
          getGlobalIdentifierID(F, Record[I++]));
        WeakUndeclaredIdentifiers.push_back(
          ReadSourceLocation(F, Record, I).getRawEncoding());
        WeakUndeclaredIdentifiers.push_back(Record[I++]);
      }
      break;

    case LOCALLY_SCOPED_EXTERNAL_DECLS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        LocallyScopedExternalDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case SELECTOR_OFFSETS: {
      F.SelectorOffsets = (const uint32_t *)BlobStart;
      F.LocalNumSelectors = Record[0];
      unsigned LocalBaseSelectorID = Record[1];
      F.BaseSelectorID = getTotalNumSelectors();
        
      if (F.LocalNumSelectors > 0) {
        // Introduce the global -> local mapping for selectors within this 
        // module.
        GlobalSelectorMap.insert(std::make_pair(getTotalNumSelectors()+1, &F));
        
        // Introduce the local -> global mapping for selectors within this 
        // module.
        F.SelectorRemap.insert(std::make_pair(LocalBaseSelectorID,
                                 F.BaseSelectorID - LocalBaseSelectorID));

        SelectorsLoaded.resize(SelectorsLoaded.size() + F.LocalNumSelectors);        
      }
      break;
    }
        
    case METHOD_POOL:
      F.SelectorLookupTableData = (const unsigned char *)BlobStart;
      if (Record[0])
        F.SelectorLookupTable
          = ASTSelectorLookupTable::Create(
                        F.SelectorLookupTableData + Record[0],
                        F.SelectorLookupTableData,
                        ASTSelectorLookupTrait(*this, F));
      TotalNumMethodPoolEntries += Record[1];
      break;

    case REFERENCED_SELECTOR_POOL:
      if (!Record.empty()) {
        for (unsigned Idx = 0, N = Record.size() - 1; Idx < N; /* in loop */) {
          ReferencedSelectorsData.push_back(getGlobalSelectorID(F, 
                                                                Record[Idx++]));
          ReferencedSelectorsData.push_back(ReadSourceLocation(F, Record, Idx).
                                              getRawEncoding());
        }
      }
      break;

    case PP_COUNTER_VALUE:
      if (!Record.empty() && Listener)
        Listener->ReadCounter(Record[0]);
      break;

    case SOURCE_LOCATION_OFFSETS: {
      F.SLocEntryOffsets = (const uint32_t *)BlobStart;
      F.LocalNumSLocEntries = Record[0];
      llvm::tie(F.SLocEntryBaseID, F.SLocEntryBaseOffset) =
          SourceMgr.AllocateLoadedSLocEntries(F.LocalNumSLocEntries, Record[1]);
      // Make our entry in the range map. BaseID is negative and growing, so
      // we invert it. Because we invert it, though, we need the other end of
      // the range.
      unsigned RangeStart =
          unsigned(-F.SLocEntryBaseID) - F.LocalNumSLocEntries + 1;
      GlobalSLocEntryMap.insert(std::make_pair(RangeStart, &F));
      F.FirstLoc = SourceLocation::getFromRawEncoding(F.SLocEntryBaseOffset);

      // Initialize the remapping table.
      // Invalid stays invalid.
      F.SLocRemap.insert(std::make_pair(0U, 0));
      // This module. Base was 2 when being compiled.
      F.SLocRemap.insert(std::make_pair(2U,
                                  static_cast<int>(F.SLocEntryBaseOffset - 2)));
      
      TotalNumSLocEntries += F.LocalNumSLocEntries;
      break;
    }

    case MODULE_OFFSET_MAP: {
      // Additional remapping information.
      const unsigned char *Data = (const unsigned char*)BlobStart;
      const unsigned char *DataEnd = Data + BlobLen;
      
      // Continuous range maps we may be updating in our module.
      ContinuousRangeMap<uint32_t, int, 2>::Builder SLocRemap(F.SLocRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        IdentifierRemap(F.IdentifierRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        PreprocessedEntityRemap(F.PreprocessedEntityRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        MacroDefinitionRemap(F.MacroDefinitionRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        SelectorRemap(F.SelectorRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder DeclRemap(F.DeclRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder TypeRemap(F.TypeRemap);

      while(Data < DataEnd) {
        uint16_t Len = io::ReadUnalignedLE16(Data);
        StringRef Name = StringRef((const char*)Data, Len);
        Data += Len;
        Module *OM = ModuleMgr.lookup(Name);
        if (!OM) {
          Error("SourceLocation remap refers to unknown module");
          return Failure;
        }

        uint32_t SLocOffset = io::ReadUnalignedLE32(Data);
        uint32_t IdentifierIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t PreprocessedEntityIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t MacroDefinitionIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t SelectorIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t DeclIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t TypeIndexOffset = io::ReadUnalignedLE32(Data);
        
        // Source location offset is mapped to OM->SLocEntryBaseOffset.
        SLocRemap.insert(std::make_pair(SLocOffset,
          static_cast<int>(OM->SLocEntryBaseOffset - SLocOffset)));
        IdentifierRemap.insert(
          std::make_pair(IdentifierIDOffset, 
                         OM->BaseIdentifierID - IdentifierIDOffset));
        PreprocessedEntityRemap.insert(
          std::make_pair(PreprocessedEntityIDOffset, 
            OM->BasePreprocessedEntityID - PreprocessedEntityIDOffset));
        MacroDefinitionRemap.insert(
          std::make_pair(MacroDefinitionIDOffset,
                         OM->BaseMacroDefinitionID - MacroDefinitionIDOffset));
        SelectorRemap.insert(std::make_pair(SelectorIDOffset, 
                               OM->BaseSelectorID - SelectorIDOffset));
        DeclRemap.insert(std::make_pair(DeclIDOffset, 
                                        OM->BaseDeclID - DeclIDOffset));
        
        TypeRemap.insert(std::make_pair(TypeIndexOffset, 
                                    OM->BaseTypeIndex - TypeIndexOffset));
      }
      break;
    }

    case SOURCE_MANAGER_LINE_TABLE:
      if (ParseLineTable(F, Record))
        return Failure;
      break;

    case FILE_SOURCE_LOCATION_OFFSETS:
      F.SLocFileOffsets = (const uint32_t *)BlobStart;
      F.LocalNumSLocFileEntries = Record[0];
      break;

    case SOURCE_LOCATION_PRELOADS: {
      // Need to transform from the local view (1-based IDs) to the global view,
      // which is based off F.SLocEntryBaseID.
      PreloadSLocEntries.reserve(PreloadSLocEntries.size() + Record.size());
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        PreloadSLocEntries.push_back(int(Record[I] - 1) + F.SLocEntryBaseID);
      break;
    }

    case STAT_CACHE: {
      if (!DisableStatCache) {
        ASTStatCache *MyStatCache =
          new ASTStatCache((const unsigned char *)BlobStart + Record[0],
                           (const unsigned char *)BlobStart,
                           NumStatHits, NumStatMisses);
        FileMgr.addStatCache(MyStatCache);
        F.StatCache = MyStatCache;
      }
      break;
    }

    case EXT_VECTOR_DECLS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        ExtVectorDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case VTABLE_USES:
      if (Record.size() % 3 != 0) {
        Error("Invalid VTABLE_USES record");
        return Failure;
      }
        
      // Later tables overwrite earlier ones.
      // FIXME: Modules will have some trouble with this. This is clearly not
      // the right way to do this.
      VTableUses.clear();
        
      for (unsigned Idx = 0, N = Record.size(); Idx != N; /* In loop */) {
        VTableUses.push_back(getGlobalDeclID(F, Record[Idx++]));
        VTableUses.push_back(
          ReadSourceLocation(F, Record, Idx).getRawEncoding());
        VTableUses.push_back(Record[Idx++]);
      }
      break;

    case DYNAMIC_CLASSES:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        DynamicClasses.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case PENDING_IMPLICIT_INSTANTIATIONS:
      if (PendingInstantiations.size() % 2 != 0) {
        Error("Invalid PENDING_IMPLICIT_INSTANTIATIONS block");
        return Failure;
      }
        
      // Later lists of pending instantiations overwrite earlier ones.
      // FIXME: This is most certainly wrong for modules.
      PendingInstantiations.clear();
      for (unsigned I = 0, N = Record.size(); I != N; /* in loop */) {
        PendingInstantiations.push_back(getGlobalDeclID(F, Record[I++]));
        PendingInstantiations.push_back(
          ReadSourceLocation(F, Record, I).getRawEncoding());
      }
      break;

    case SEMA_DECL_REFS:
      // Later tables overwrite earlier ones.
      // FIXME: Modules will have some trouble with this.
      SemaDeclRefs.clear();
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        SemaDeclRefs.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case ORIGINAL_FILE_NAME:
      // The primary AST will be the last to get here, so it will be the one
      // that's used.
      ActualOriginalFileName.assign(BlobStart, BlobLen);
      OriginalFileName = ActualOriginalFileName;
      MaybeAddSystemRootToFilename(OriginalFileName);
      break;

    case ORIGINAL_FILE_ID:
      OriginalFileID = FileID::get(Record[0]);
      break;
        
    case ORIGINAL_PCH_DIR:
      // The primary AST will be the last to get here, so it will be the one
      // that's used.
      OriginalDir.assign(BlobStart, BlobLen);
      break;

    case VERSION_CONTROL_BRANCH_REVISION: {
      const std::string &CurBranch = getClangFullRepositoryVersion();
      StringRef ASTBranch(BlobStart, BlobLen);
      if (StringRef(CurBranch) != ASTBranch && !DisableValidation) {
        Diag(diag::warn_pch_different_branch) << ASTBranch << CurBranch;
        return IgnorePCH;
      }
      break;
    }

    case MACRO_DEFINITION_OFFSETS: {
      F.MacroDefinitionOffsets = (const uint32_t *)BlobStart;
      F.NumPreallocatedPreprocessingEntities = Record[0];
      unsigned LocalBasePreprocessedEntityID = Record[1];
      F.LocalNumMacroDefinitions = Record[2];
      unsigned LocalBaseMacroID = Record[3];
      
      unsigned StartingID;
      if (PP) {
        if (!PP->getPreprocessingRecord())
          PP->createPreprocessingRecord(true);
        if (!PP->getPreprocessingRecord()->getExternalSource())
          PP->getPreprocessingRecord()->SetExternalSource(*this);
        StartingID 
          = PP->getPreprocessingRecord()
              ->allocateLoadedEntities(F.NumPreallocatedPreprocessingEntities);
      } else {
        // FIXME: We'll eventually want to kill this path, since it assumes
        // a particular allocation strategy in the preprocessing record.
        StartingID = getTotalNumPreprocessedEntities() 
                   - F.NumPreallocatedPreprocessingEntities;
      }
      F.BaseMacroDefinitionID = getTotalNumMacroDefinitions();
      F.BasePreprocessedEntityID = StartingID;

      if (F.NumPreallocatedPreprocessingEntities > 0) {
        // Introduce the global -> local mapping for preprocessed entities in
        // this module.
        GlobalPreprocessedEntityMap.insert(std::make_pair(StartingID, &F));
       
        // Introduce the local -> global mapping for preprocessed entities in
        // this module.
        F.PreprocessedEntityRemap.insert(
          std::make_pair(LocalBasePreprocessedEntityID,
            F.BasePreprocessedEntityID - LocalBasePreprocessedEntityID));
      }
      

      if (F.LocalNumMacroDefinitions > 0) {
        // Introduce the global -> local mapping for macro definitions within 
        // this module.
        GlobalMacroDefinitionMap.insert(
          std::make_pair(getTotalNumMacroDefinitions() + 1, &F));
        
        // Introduce the local -> global mapping for macro definitions within
        // this module.
        F.MacroDefinitionRemap.insert(
          std::make_pair(LocalBaseMacroID,
                         F.BaseMacroDefinitionID - LocalBaseMacroID));
        
        MacroDefinitionsLoaded.resize(
                    MacroDefinitionsLoaded.size() + F.LocalNumMacroDefinitions);
      }
      
      break;
    }
        
    case DECL_UPDATE_OFFSETS: {
      if (Record.size() % 2 != 0) {
        Error("invalid DECL_UPDATE_OFFSETS block in AST file");
        return Failure;
      }
      for (unsigned I = 0, N = Record.size(); I != N; I += 2)
        DeclUpdateOffsets[getGlobalDeclID(F, Record[I])]
          .push_back(std::make_pair(&F, Record[I+1]));
      break;
    }

    case DECL_REPLACEMENTS: {
      if (Record.size() % 2 != 0) {
        Error("invalid DECL_REPLACEMENTS block in AST file");
        return Failure;
      }
      for (unsigned I = 0, N = Record.size(); I != N; I += 2)
        ReplacedDecls[getGlobalDeclID(F, Record[I])]
          = std::make_pair(&F, Record[I+1]);
      break;
    }
        
    case CXX_BASE_SPECIFIER_OFFSETS: {
      if (F.LocalNumCXXBaseSpecifiers != 0) {
        Error("duplicate CXX_BASE_SPECIFIER_OFFSETS record in AST file");
        return Failure;
      }
      
      F.LocalNumCXXBaseSpecifiers = Record[0];
      F.CXXBaseSpecifiersOffsets = (const uint32_t *)BlobStart;
      NumCXXBaseSpecifiersLoaded += F.LocalNumCXXBaseSpecifiers;
      break;
    }

    case DIAG_PRAGMA_MAPPINGS:
      if (Record.size() % 2 != 0) {
        Error("invalid DIAG_USER_MAPPINGS block in AST file");
        return Failure;
      }
        
      if (F.PragmaDiagMappings.empty())
        F.PragmaDiagMappings.swap(Record);
      else
        F.PragmaDiagMappings.insert(F.PragmaDiagMappings.end(),
                                    Record.begin(), Record.end());
      break;
        
    case CUDA_SPECIAL_DECL_REFS:
      // Later tables overwrite earlier ones.
      // FIXME: Modules will have trouble with this.
      CUDASpecialDeclRefs.clear();
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        CUDASpecialDeclRefs.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case HEADER_SEARCH_TABLE: {
      F.HeaderFileInfoTableData = BlobStart;
      F.LocalNumHeaderFileInfos = Record[1];
      F.HeaderFileFrameworkStrings = BlobStart + Record[2];
      if (Record[0]) {
        F.HeaderFileInfoTable
          = HeaderFileInfoLookupTable::Create(
                   (const unsigned char *)F.HeaderFileInfoTableData + Record[0],
                   (const unsigned char *)F.HeaderFileInfoTableData,
                   HeaderFileInfoTrait(*this, F, 
                                       PP? &PP->getHeaderSearchInfo() : 0,
                                       BlobStart + Record[2]));
        if (PP)
          PP->getHeaderSearchInfo().SetExternalSource(this);
      }
      break;
    }
        
    case FP_PRAGMA_OPTIONS:
      // Later tables overwrite earlier ones.
      FPPragmaOptions.swap(Record);
      break;

    case OPENCL_EXTENSIONS:
      // Later tables overwrite earlier ones.
      OpenCLExtensions.swap(Record);
      break;

    case TENTATIVE_DEFINITIONS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        TentativeDefinitions.push_back(getGlobalDeclID(F, Record[I]));
      break;
        
    case KNOWN_NAMESPACES:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        KnownNamespaces.push_back(getGlobalDeclID(F, Record[I]));
      break;
    }
    First = false;
  }
  Error("premature end of bitstream in AST file");
  return Failure;
}

ASTReader::ASTReadResult ASTReader::validateFileEntries() {
  for (ModuleIterator I = ModuleMgr.begin(),
      E = ModuleMgr.end(); I != E; ++I) {
    Module *F = *I;
    llvm::BitstreamCursor &SLocEntryCursor = F->SLocEntryCursor;

    for (unsigned i = 0, e = F->LocalNumSLocFileEntries; i != e; ++i) {
      SLocEntryCursor.JumpToBit(F->SLocFileOffsets[i]);
      unsigned Code = SLocEntryCursor.ReadCode();
      if (Code == llvm::bitc::END_BLOCK ||
          Code == llvm::bitc::ENTER_SUBBLOCK ||
          Code == llvm::bitc::DEFINE_ABBREV) {
        Error("incorrectly-formatted source location entry in AST file");
        return Failure;
      }
  
      RecordData Record;
      const char *BlobStart;
      unsigned BlobLen;
      switch (SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
      default:
        Error("incorrectly-formatted source location entry in AST file");
        return Failure;
  
      case SM_SLOC_FILE_ENTRY: {
        StringRef Filename(BlobStart, BlobLen);
        const FileEntry *File = getFileEntry(Filename);

        if (File == 0) {
          std::string ErrorStr = "could not find file '";
          ErrorStr += Filename;
          ErrorStr += "' referenced by AST file";
          Error(ErrorStr.c_str());
          return IgnorePCH;
        }
  
        if (Record.size() < 6) {
          Error("source location entry is incorrect");
          return Failure;
        }

        // The stat info from the FileEntry came from the cached stat
        // info of the PCH, so we cannot trust it.
        struct stat StatBuf;
        if (::stat(File->getName(), &StatBuf) != 0) {
          StatBuf.st_size = File->getSize();
          StatBuf.st_mtime = File->getModificationTime();
        }

        if (((off_t)Record[4] != StatBuf.st_size
#if !defined(LLVM_ON_WIN32)
            // In our regression testing, the Windows file system seems to
            // have inconsistent modification times that sometimes
            // erroneously trigger this error-handling path.
             || (time_t)Record[5] != StatBuf.st_mtime
#endif
            )) {
          Error(diag::err_fe_pch_file_modified, Filename);
          return IgnorePCH;
        }

        break;
      }
      }
    }
  }

  return Success;
}

ASTReader::ASTReadResult ASTReader::ReadAST(const std::string &FileName,
                                            ModuleKind Type) {
  switch(ReadASTCore(FileName, Type)) {
  case Failure: return Failure;
  case IgnorePCH: return IgnorePCH;
  case Success: break;
  }

  // Here comes stuff that we only do once the entire chain is loaded.

  if (!DisableValidation) {
    switch(validateFileEntries()) {
    case Failure: return Failure;
    case IgnorePCH: return IgnorePCH;
    case Success: break;
    }
  }

  // Preload SLocEntries.
  for (unsigned I = 0, N = PreloadSLocEntries.size(); I != N; ++I) {
    ASTReadResult Result = ReadSLocEntryRecord(PreloadSLocEntries[I]);
    if (Result != Success)
      return Failure;
  }
  PreloadSLocEntries.clear();
  
  // Check the predefines buffers.
  if (!DisableValidation && Type != MK_Module && CheckPredefinesBuffers())
    return IgnorePCH;

  if (PP) {
    // Initialization of keywords and pragmas occurs before the
    // AST file is read, so there may be some identifiers that were
    // loaded into the IdentifierTable before we intercepted the
    // creation of identifiers. Iterate through the list of known
    // identifiers and determine whether we have to establish
    // preprocessor definitions or top-level identifier declaration
    // chains for those identifiers.
    //
    // We copy the IdentifierInfo pointers to a small vector first,
    // since de-serializing declarations or macro definitions can add
    // new entries into the identifier table, invalidating the
    // iterators.
    //
    // FIXME: We need a lazier way to load this information, e.g., by marking
    // the identifier data as 'dirty', so that it will be looked up in the
    // AST file(s) if it is uttered in the source. This could save us some
    // module load time.
    SmallVector<IdentifierInfo *, 128> Identifiers;
    for (IdentifierTable::iterator Id = PP->getIdentifierTable().begin(),
                                IdEnd = PP->getIdentifierTable().end();
         Id != IdEnd; ++Id)
      Identifiers.push_back(Id->second);
    // We need to search the tables in all files.
    for (ModuleIterator J = ModuleMgr.begin(),
        M = ModuleMgr.end(); J != M; ++J) {
      ASTIdentifierLookupTable *IdTable
        = (ASTIdentifierLookupTable *)(*J)->IdentifierLookupTable;
      // Not all AST files necessarily have identifier tables, only the useful
      // ones.
      if (!IdTable)
        continue;
      for (unsigned I = 0, N = Identifiers.size(); I != N; ++I) {
        IdentifierInfo *II = Identifiers[I];
        // Look in the on-disk hash tables for an entry for this identifier
        ASTIdentifierLookupTrait Info(*this, *(*J), II);
        std::pair<const char*,unsigned> Key(II->getNameStart(),II->getLength());
        ASTIdentifierLookupTable::iterator Pos = IdTable->find(Key, &Info);
        if (Pos == IdTable->end())
          continue;

        // Dereferencing the iterator has the effect of populating the
        // IdentifierInfo node with the various declarations it needs.
        (void)*Pos;
      }
    }
  }

  if (Context)
    InitializeContext(*Context);

  if (DeserializationListener)
    DeserializationListener->ReaderInitialized(this);

  // If this AST file is a precompiled preamble, then set the main file ID of 
  // the source manager to the file source file from which the preamble was
  // built. This is the only valid way to use a precompiled preamble.
  if (Type == MK_Preamble) {
    if (OriginalFileID.isInvalid()) {
      SourceLocation Loc
        = SourceMgr.getLocation(FileMgr.getFile(getOriginalSourceFile()), 1, 1);
      if (Loc.isValid())
        OriginalFileID = SourceMgr.getDecomposedLoc(Loc).first;
    }
    else {
      OriginalFileID = FileID::get(ModuleMgr.getPrimaryModule().SLocEntryBaseID
                                        + OriginalFileID.getOpaqueValue() - 1);
    }

    if (!OriginalFileID.isInvalid())
      SourceMgr.SetPreambleFileID(OriginalFileID);
  }
  
  return Success;
}

ASTReader::ASTReadResult ASTReader::ReadASTCore(StringRef FileName,
                                                ModuleKind Type) {
  Module &F = ModuleMgr.addModule(FileName, Type);

  if (FileName != "-") {
    CurrentDir = llvm::sys::path::parent_path(FileName);
    if (CurrentDir.empty()) CurrentDir = ".";
  }

  if (llvm::MemoryBuffer *Buffer = ModuleMgr.lookupBuffer(FileName)) {
    F.Buffer.reset(Buffer);
    assert(F.Buffer && "Passed null buffer");
  } else {
    // Open the AST file.
    //
    // FIXME: This shouldn't be here, we should just take a raw_ostream.
    std::string ErrStr;
    llvm::error_code ec;
    if (FileName == "-") {
      ec = llvm::MemoryBuffer::getSTDIN(F.Buffer);
      if (ec)
        ErrStr = ec.message();
    } else
      F.Buffer.reset(FileMgr.getBufferForFile(FileName, &ErrStr));
    if (!F.Buffer) {
      Error(ErrStr.c_str());
      return IgnorePCH;
    }
  }

  // Initialize the stream
  F.StreamFile.init((const unsigned char *)F.Buffer->getBufferStart(),
                    (const unsigned char *)F.Buffer->getBufferEnd());
  llvm::BitstreamCursor &Stream = F.Stream;
  Stream.init(F.StreamFile);
  F.SizeInBits = F.Buffer->getBufferSize() * 8;
  
  // Sniff for the signature.
  if (Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'P' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'H') {
    Diag(diag::err_not_a_pch_file) << FileName;
    return Failure;
  }

  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();

    if (Code != llvm::bitc::ENTER_SUBBLOCK) {
      Error("invalid record at top-level of AST file");
      return Failure;
    }

    unsigned BlockID = Stream.ReadSubBlockID();

    // We only know the AST subblock ID.
    switch (BlockID) {
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (Stream.ReadBlockInfoBlock()) {
        Error("malformed BlockInfoBlock in AST file");
        return Failure;
      }
      break;
    case AST_BLOCK_ID:
      switch (ReadASTBlock(F)) {
      case Success:
        break;

      case Failure:
        return Failure;

      case IgnorePCH:
        // FIXME: We could consider reading through to the end of this
        // AST block, skipping subblocks, to see if there are other
        // AST blocks elsewhere.

        // FIXME: We can't clear loaded slocentries anymore.
        //SourceMgr.ClearPreallocatedSLocEntries();

        // Remove the stat cache.
        if (F.StatCache)
          FileMgr.removeStatCache((ASTStatCache*)F.StatCache);

        return IgnorePCH;
      }
      break;
    default:
      if (Stream.SkipBlock()) {
        Error("malformed block record in AST file");
        return Failure;
      }
      break;
    }
  }
  
  // Once read, set the Module bit base offset and update the size in 
  // bits of all files we've seen.
  F.GlobalBitOffset = TotalModulesSizeInBits;
  TotalModulesSizeInBits += F.SizeInBits;
  GlobalBitOffsetsMap.insert(std::make_pair(F.GlobalBitOffset, &F));
  return Success;
}

void ASTReader::setPreprocessor(Preprocessor &pp) {
  PP = &pp;
  
  if (unsigned N = getTotalNumPreprocessedEntities()) {
    if (!PP->getPreprocessingRecord())
      PP->createPreprocessingRecord(true);
    PP->getPreprocessingRecord()->SetExternalSource(*this);
    PP->getPreprocessingRecord()->allocateLoadedEntities(N);
  }
  
  PP->getHeaderSearchInfo().SetExternalLookup(this);
  PP->getHeaderSearchInfo().SetExternalSource(this);
}

void ASTReader::InitializeContext(ASTContext &Ctx) {
  Context = &Ctx;
  assert(Context && "Passed null context!");

  assert(PP && "Forgot to set Preprocessor ?");
  PP->getIdentifierTable().setExternalIdentifierLookup(this);
  PP->setExternalSource(this);
  
  // If we have an update block for the TU waiting, we have to add it before
  // deserializing the decl.
  DeclContextOffsetsMap::iterator DCU = DeclContextOffsets.find(0);
  if (DCU != DeclContextOffsets.end()) {
    // Insertion could invalidate map, so grab vector.
    DeclContextInfos T;
    T.swap(DCU->second);
    DeclContextOffsets.erase(DCU);
    DeclContextOffsets[Ctx.getTranslationUnitDecl()].swap(T);
  }

  // Load the translation unit declaration
  GetTranslationUnitDecl();

  // Load the special types.
  Context->setBuiltinVaListType(
    GetType(SpecialTypes[SPECIAL_TYPE_BUILTIN_VA_LIST]));
  if (unsigned Id = SpecialTypes[SPECIAL_TYPE_OBJC_ID])
    Context->setObjCIdType(GetType(Id));
  if (unsigned Sel = SpecialTypes[SPECIAL_TYPE_OBJC_SELECTOR])
    Context->setObjCSelType(GetType(Sel));
  if (unsigned Proto = SpecialTypes[SPECIAL_TYPE_OBJC_PROTOCOL])
    Context->setObjCProtoType(GetType(Proto));
  if (unsigned Class = SpecialTypes[SPECIAL_TYPE_OBJC_CLASS])
    Context->setObjCClassType(GetType(Class));

  if (unsigned String = SpecialTypes[SPECIAL_TYPE_CF_CONSTANT_STRING])
    Context->setCFConstantStringType(GetType(String));
  if (unsigned FastEnum
        = SpecialTypes[SPECIAL_TYPE_OBJC_FAST_ENUMERATION_STATE])
    Context->setObjCFastEnumerationStateType(GetType(FastEnum));
  if (unsigned File = SpecialTypes[SPECIAL_TYPE_FILE]) {
    QualType FileType = GetType(File);
    if (FileType.isNull()) {
      Error("FILE type is NULL");
      return;
    }
    if (const TypedefType *Typedef = FileType->getAs<TypedefType>())
      Context->setFILEDecl(Typedef->getDecl());
    else {
      const TagType *Tag = FileType->getAs<TagType>();
      if (!Tag) {
        Error("Invalid FILE type in AST file");
        return;
      }
      Context->setFILEDecl(Tag->getDecl());
    }
  }
  if (unsigned Jmp_buf = SpecialTypes[SPECIAL_TYPE_jmp_buf]) {
    QualType Jmp_bufType = GetType(Jmp_buf);
    if (Jmp_bufType.isNull()) {
      Error("jmp_buf type is NULL");
      return;
    }
    if (const TypedefType *Typedef = Jmp_bufType->getAs<TypedefType>())
      Context->setjmp_bufDecl(Typedef->getDecl());
    else {
      const TagType *Tag = Jmp_bufType->getAs<TagType>();
      if (!Tag) {
        Error("Invalid jmp_buf type in AST file");
        return;
      }
      Context->setjmp_bufDecl(Tag->getDecl());
    }
  }
  if (unsigned Sigjmp_buf = SpecialTypes[SPECIAL_TYPE_sigjmp_buf]) {
    QualType Sigjmp_bufType = GetType(Sigjmp_buf);
    if (Sigjmp_bufType.isNull()) {
      Error("sigjmp_buf type is NULL");
      return;
    }
    if (const TypedefType *Typedef = Sigjmp_bufType->getAs<TypedefType>())
      Context->setsigjmp_bufDecl(Typedef->getDecl());
    else {
      const TagType *Tag = Sigjmp_bufType->getAs<TagType>();
      assert(Tag && "Invalid sigjmp_buf type in AST file");
      Context->setsigjmp_bufDecl(Tag->getDecl());
    }
  }
  if (unsigned ObjCIdRedef
        = SpecialTypes[SPECIAL_TYPE_OBJC_ID_REDEFINITION])
    Context->ObjCIdRedefinitionType = GetType(ObjCIdRedef);
  if (unsigned ObjCClassRedef
      = SpecialTypes[SPECIAL_TYPE_OBJC_CLASS_REDEFINITION])
    Context->ObjCClassRedefinitionType = GetType(ObjCClassRedef);
  if (unsigned String = SpecialTypes[SPECIAL_TYPE_BLOCK_DESCRIPTOR])
    Context->setBlockDescriptorType(GetType(String));
  if (unsigned String
      = SpecialTypes[SPECIAL_TYPE_BLOCK_EXTENDED_DESCRIPTOR])
    Context->setBlockDescriptorExtendedType(GetType(String));
  if (unsigned ObjCSelRedef
      = SpecialTypes[SPECIAL_TYPE_OBJC_SEL_REDEFINITION])
    Context->ObjCSelRedefinitionType = GetType(ObjCSelRedef);
  if (unsigned String = SpecialTypes[SPECIAL_TYPE_NS_CONSTANT_STRING])
    Context->setNSConstantStringType(GetType(String));

  if (SpecialTypes[SPECIAL_TYPE_INT128_INSTALLED])
    Context->setInt128Installed();

  if (unsigned AutoDeduct = SpecialTypes[SPECIAL_TYPE_AUTO_DEDUCT])
    Context->AutoDeductTy = GetType(AutoDeduct);
  if (unsigned AutoRRefDeduct = SpecialTypes[SPECIAL_TYPE_AUTO_RREF_DEDUCT])
    Context->AutoRRefDeductTy = GetType(AutoRRefDeduct);

  ReadPragmaDiagnosticMappings(Context->getDiagnostics());

  // If there were any CUDA special declarations, deserialize them.
  if (!CUDASpecialDeclRefs.empty()) {
    assert(CUDASpecialDeclRefs.size() == 1 && "More decl refs than expected!");
    Context->setcudaConfigureCallDecl(
                           cast<FunctionDecl>(GetDecl(CUDASpecialDeclRefs[0])));
  }
}

/// \brief Retrieve the name of the original source file name
/// directly from the AST file, without actually loading the AST
/// file.
std::string ASTReader::getOriginalSourceFile(const std::string &ASTFileName,
                                             FileManager &FileMgr,
                                             Diagnostic &Diags) {
  // Open the AST file.
  std::string ErrStr;
  llvm::OwningPtr<llvm::MemoryBuffer> Buffer;
  Buffer.reset(FileMgr.getBufferForFile(ASTFileName, &ErrStr));
  if (!Buffer) {
    Diags.Report(diag::err_fe_unable_to_read_pch_file) << ErrStr;
    return std::string();
  }

  // Initialize the stream
  llvm::BitstreamReader StreamFile;
  llvm::BitstreamCursor Stream;
  StreamFile.init((const unsigned char *)Buffer->getBufferStart(),
                  (const unsigned char *)Buffer->getBufferEnd());
  Stream.init(StreamFile);

  // Sniff for the signature.
  if (Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'P' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'H') {
    Diags.Report(diag::err_fe_not_a_pch_file) << ASTFileName;
    return std::string();
  }

  RecordData Record;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      unsigned BlockID = Stream.ReadSubBlockID();

      // We only know the AST subblock ID.
      switch (BlockID) {
      case AST_BLOCK_ID:
        if (Stream.EnterSubBlock(AST_BLOCK_ID)) {
          Diags.Report(diag::err_fe_pch_malformed_block) << ASTFileName;
          return std::string();
        }
        break;

      default:
        if (Stream.SkipBlock()) {
          Diags.Report(diag::err_fe_pch_malformed_block) << ASTFileName;
          return std::string();
        }
        break;
      }
      continue;
    }

    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Diags.Report(diag::err_fe_pch_error_at_end_block) << ASTFileName;
        return std::string();
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    Record.clear();
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    if (Stream.ReadRecord(Code, Record, &BlobStart, &BlobLen)
          == ORIGINAL_FILE_NAME)
      return std::string(BlobStart, BlobLen);
  }

  return std::string();
}

/// \brief Parse the record that corresponds to a LangOptions data
/// structure.
///
/// This routine parses the language options from the AST file and then gives
/// them to the AST listener if one is set.
///
/// \returns true if the listener deems the file unacceptable, false otherwise.
bool ASTReader::ParseLanguageOptions(
                             const SmallVectorImpl<uint64_t> &Record) {
  if (Listener) {
    LangOptions LangOpts;

  #define PARSE_LANGOPT(Option)                  \
      LangOpts.Option = Record[Idx];             \
      ++Idx

    unsigned Idx = 0;
    PARSE_LANGOPT(Trigraphs);
    PARSE_LANGOPT(BCPLComment);
    PARSE_LANGOPT(DollarIdents);
    PARSE_LANGOPT(AsmPreprocessor);
    PARSE_LANGOPT(GNUMode);
    PARSE_LANGOPT(GNUKeywords);
    PARSE_LANGOPT(ImplicitInt);
    PARSE_LANGOPT(Digraphs);
    PARSE_LANGOPT(HexFloats);
    PARSE_LANGOPT(C99);
    PARSE_LANGOPT(C1X);
    PARSE_LANGOPT(Microsoft);
    PARSE_LANGOPT(CPlusPlus);
    PARSE_LANGOPT(CPlusPlus0x);
    PARSE_LANGOPT(CXXOperatorNames);
    PARSE_LANGOPT(ObjC1);
    PARSE_LANGOPT(ObjC2);
    PARSE_LANGOPT(ObjCNonFragileABI);
    PARSE_LANGOPT(ObjCNonFragileABI2);
    PARSE_LANGOPT(AppleKext);
    PARSE_LANGOPT(ObjCDefaultSynthProperties);
    PARSE_LANGOPT(ObjCInferRelatedResultType);
    PARSE_LANGOPT(NoConstantCFStrings);
    PARSE_LANGOPT(PascalStrings);
    PARSE_LANGOPT(WritableStrings);
    PARSE_LANGOPT(LaxVectorConversions);
    PARSE_LANGOPT(AltiVec);
    PARSE_LANGOPT(Exceptions);
    PARSE_LANGOPT(ObjCExceptions);
    PARSE_LANGOPT(CXXExceptions);
    PARSE_LANGOPT(SjLjExceptions);
    PARSE_LANGOPT(MSBitfields);
    PARSE_LANGOPT(NeXTRuntime);
    PARSE_LANGOPT(Freestanding);
    PARSE_LANGOPT(NoBuiltin);
    PARSE_LANGOPT(ThreadsafeStatics);
    PARSE_LANGOPT(POSIXThreads);
    PARSE_LANGOPT(Blocks);
    PARSE_LANGOPT(EmitAllDecls);
    PARSE_LANGOPT(MathErrno);
    LangOpts.setSignedOverflowBehavior((LangOptions::SignedOverflowBehaviorTy)
                                       Record[Idx++]);
    PARSE_LANGOPT(HeinousExtensions);
    PARSE_LANGOPT(Optimize);
    PARSE_LANGOPT(OptimizeSize);
    PARSE_LANGOPT(Static);
    PARSE_LANGOPT(PICLevel);
    PARSE_LANGOPT(GNUInline);
    PARSE_LANGOPT(NoInline);
    PARSE_LANGOPT(Deprecated);
    PARSE_LANGOPT(AccessControl);
    PARSE_LANGOPT(CharIsSigned);
    PARSE_LANGOPT(ShortWChar);
    PARSE_LANGOPT(ShortEnums);
    LangOpts.setGCMode((LangOptions::GCMode)Record[Idx++]);
    LangOpts.setVisibilityMode((Visibility)Record[Idx++]);
    LangOpts.setStackProtectorMode((LangOptions::StackProtectorMode)
                                   Record[Idx++]);
    PARSE_LANGOPT(InstantiationDepth);
    PARSE_LANGOPT(OpenCL);
    PARSE_LANGOPT(CUDA);
    PARSE_LANGOPT(CatchUndefined);
    PARSE_LANGOPT(DefaultFPContract);
    PARSE_LANGOPT(ElideConstructors);
    PARSE_LANGOPT(SpellChecking);
    PARSE_LANGOPT(MRTD);
    PARSE_LANGOPT(ObjCAutoRefCount);
    PARSE_LANGOPT(ObjCInferRelatedReturnType);
  #undef PARSE_LANGOPT

    return Listener->ReadLanguageOptions(LangOpts);
  }

  return false;
}

void ASTReader::ReadPreprocessedEntities() {
  for (ModuleIterator I = ModuleMgr.begin(), E = ModuleMgr.end(); I != E; ++I) {
    Module &F = *(*I);
    if (!F.PreprocessorDetailCursor.getBitStreamReader())
      continue;

    SavedStreamPosition SavedPosition(F.PreprocessorDetailCursor);
    F.PreprocessorDetailCursor.JumpToBit(F.PreprocessorDetailStartOffset);
    while (LoadPreprocessedEntity(F)) { }
  }
}

PreprocessedEntity *ASTReader::ReadPreprocessedEntityAtOffset(uint64_t Offset) {
  RecordLocation Loc = getLocalBitOffset(Offset);

  // Keep track of where we are in the stream, then jump back there
  // after reading this entity.
  SavedStreamPosition SavedPosition(Loc.F->PreprocessorDetailCursor);  
  Loc.F->PreprocessorDetailCursor.JumpToBit(Loc.Offset);
  return LoadPreprocessedEntity(*Loc.F);
}

HeaderFileInfo ASTReader::GetHeaderFileInfo(const FileEntry *FE) {
  for (ModuleIterator I = ModuleMgr.begin(), E = ModuleMgr.end(); I != E; ++I) {
    Module &F = *(*I);

    HeaderFileInfoTrait Trait(*this, F, &PP->getHeaderSearchInfo(),
                              F.HeaderFileFrameworkStrings,
                              FE->getName());
    
    HeaderFileInfoLookupTable *Table
      = static_cast<HeaderFileInfoLookupTable *>(F.HeaderFileInfoTable);
    if (!Table)
      continue;
    
    // Look in the on-disk hash table for an entry for this file name.
    HeaderFileInfoLookupTable::iterator Pos = Table->find(FE->getName(), 
                                                          &Trait);
    if (Pos == Table->end())
      continue;

    HeaderFileInfo HFI = *Pos;
    if (Listener)
      Listener->ReadHeaderFileInfo(HFI, FE->getUID());

    return HFI;
  }
  
  return HeaderFileInfo();
}

void ASTReader::ReadPragmaDiagnosticMappings(Diagnostic &Diag) {
  for (ModuleIterator I = ModuleMgr.begin(), E = ModuleMgr.end(); I != E; ++I) {
    Module &F = *(*I);
    unsigned Idx = 0;
    while (Idx < F.PragmaDiagMappings.size()) {
      SourceLocation Loc = ReadSourceLocation(F, F.PragmaDiagMappings[Idx++]);
      while (1) {
        assert(Idx < F.PragmaDiagMappings.size() &&
               "Invalid data, didn't find '-1' marking end of diag/map pairs");
        if (Idx >= F.PragmaDiagMappings.size()) {
          break; // Something is messed up but at least avoid infinite loop in
                 // release build.
        }
        unsigned DiagID = F.PragmaDiagMappings[Idx++];
        if (DiagID == (unsigned)-1) {
          break; // no more diag/map pairs for this location.
        }
        diag::Mapping Map = (diag::Mapping)F.PragmaDiagMappings[Idx++];
        Diag.setDiagnosticMapping(DiagID, Map, Loc);
      }
    }
  }
}

/// \brief Get the correct cursor and offset for loading a type.
ASTReader::RecordLocation ASTReader::TypeCursorForIndex(unsigned Index) {
  GlobalTypeMapType::iterator I = GlobalTypeMap.find(Index);
  assert(I != GlobalTypeMap.end() && "Corrupted global type map");
  Module *M = I->second;
  return RecordLocation(M, M->TypeOffsets[Index - M->BaseTypeIndex]);
}

/// \brief Read and return the type with the given index..
///
/// The index is the type ID, shifted and minus the number of predefs. This
/// routine actually reads the record corresponding to the type at the given
/// location. It is a helper routine for GetType, which deals with reading type
/// IDs.
QualType ASTReader::readTypeRecord(unsigned Index) {
  RecordLocation Loc = TypeCursorForIndex(Index);
  llvm::BitstreamCursor &DeclsCursor = Loc.F->DeclsCursor;

  // Keep track of where we are in the stream, then jump back there
  // after reading this type.
  SavedStreamPosition SavedPosition(DeclsCursor);

  ReadingKindTracker ReadingKind(Read_Type, *this);

  // Note that we are loading a type record.
  Deserializing AType(this);

  unsigned Idx = 0;
  DeclsCursor.JumpToBit(Loc.Offset);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  switch ((TypeCode)DeclsCursor.ReadRecord(Code, Record)) {
  case TYPE_EXT_QUAL: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of extended qualifier type");
      return QualType();
    }
    QualType Base = readType(*Loc.F, Record, Idx);
    Qualifiers Quals = Qualifiers::fromOpaqueValue(Record[Idx++]);
    return Context->getQualifiedType(Base, Quals);
  }

  case TYPE_COMPLEX: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of complex type");
      return QualType();
    }
    QualType ElemType = readType(*Loc.F, Record, Idx);
    return Context->getComplexType(ElemType);
  }

  case TYPE_POINTER: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of pointer type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context->getPointerType(PointeeType);
  }

  case TYPE_BLOCK_POINTER: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of block pointer type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context->getBlockPointerType(PointeeType);
  }

  case TYPE_LVALUE_REFERENCE: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of lvalue reference type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context->getLValueReferenceType(PointeeType, Record[1]);
  }

  case TYPE_RVALUE_REFERENCE: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of rvalue reference type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context->getRValueReferenceType(PointeeType);
  }

  case TYPE_MEMBER_POINTER: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of member pointer type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    QualType ClassType = readType(*Loc.F, Record, Idx);
    if (PointeeType.isNull() || ClassType.isNull())
      return QualType();
    
    return Context->getMemberPointerType(PointeeType, ClassType.getTypePtr());
  }

  case TYPE_CONSTANT_ARRAY: {
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    unsigned Idx = 3;
    llvm::APInt Size = ReadAPInt(Record, Idx);
    return Context->getConstantArrayType(ElementType, Size,
                                         ASM, IndexTypeQuals);
  }

  case TYPE_INCOMPLETE_ARRAY: {
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    return Context->getIncompleteArrayType(ElementType, ASM, IndexTypeQuals);
  }

  case TYPE_VARIABLE_ARRAY: {
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    SourceLocation LBLoc = ReadSourceLocation(*Loc.F, Record[3]);
    SourceLocation RBLoc = ReadSourceLocation(*Loc.F, Record[4]);
    return Context->getVariableArrayType(ElementType, ReadExpr(*Loc.F),
                                         ASM, IndexTypeQuals,
                                         SourceRange(LBLoc, RBLoc));
  }

  case TYPE_VECTOR: {
    if (Record.size() != 3) {
      Error("incorrect encoding of vector type in AST file");
      return QualType();
    }

    QualType ElementType = readType(*Loc.F, Record, Idx);
    unsigned NumElements = Record[1];
    unsigned VecKind = Record[2];
    return Context->getVectorType(ElementType, NumElements,
                                  (VectorType::VectorKind)VecKind);
  }

  case TYPE_EXT_VECTOR: {
    if (Record.size() != 3) {
      Error("incorrect encoding of extended vector type in AST file");
      return QualType();
    }

    QualType ElementType = readType(*Loc.F, Record, Idx);
    unsigned NumElements = Record[1];
    return Context->getExtVectorType(ElementType, NumElements);
  }

  case TYPE_FUNCTION_NO_PROTO: {
    if (Record.size() != 6) {
      Error("incorrect encoding of no-proto function type");
      return QualType();
    }
    QualType ResultType = readType(*Loc.F, Record, Idx);
    FunctionType::ExtInfo Info(Record[1], Record[2], Record[3],
                               (CallingConv)Record[4], Record[5]);
    return Context->getFunctionNoProtoType(ResultType, Info);
  }

  case TYPE_FUNCTION_PROTO: {
    QualType ResultType = readType(*Loc.F, Record, Idx);

    FunctionProtoType::ExtProtoInfo EPI;
    EPI.ExtInfo = FunctionType::ExtInfo(/*noreturn*/ Record[1],
                                        /*hasregparm*/ Record[2],
                                        /*regparm*/ Record[3],
                                        static_cast<CallingConv>(Record[4]),
                                        /*produces*/ Record[5]);

    unsigned Idx = 6;
    unsigned NumParams = Record[Idx++];
    SmallVector<QualType, 16> ParamTypes;
    for (unsigned I = 0; I != NumParams; ++I)
      ParamTypes.push_back(readType(*Loc.F, Record, Idx));

    EPI.Variadic = Record[Idx++];
    EPI.TypeQuals = Record[Idx++];
    EPI.RefQualifier = static_cast<RefQualifierKind>(Record[Idx++]);
    ExceptionSpecificationType EST =
        static_cast<ExceptionSpecificationType>(Record[Idx++]);
    EPI.ExceptionSpecType = EST;
    if (EST == EST_Dynamic) {
      EPI.NumExceptions = Record[Idx++];
      SmallVector<QualType, 2> Exceptions;
      for (unsigned I = 0; I != EPI.NumExceptions; ++I)
        Exceptions.push_back(readType(*Loc.F, Record, Idx));
      EPI.Exceptions = Exceptions.data();
    } else if (EST == EST_ComputedNoexcept) {
      EPI.NoexceptExpr = ReadExpr(*Loc.F);
    }
    return Context->getFunctionType(ResultType, ParamTypes.data(), NumParams,
                                    EPI);
  }

  case TYPE_UNRESOLVED_USING: {
    unsigned Idx = 0;
    return Context->getTypeDeclType(
                  ReadDeclAs<UnresolvedUsingTypenameDecl>(*Loc.F, Record, Idx));
  }
      
  case TYPE_TYPEDEF: {
    if (Record.size() != 2) {
      Error("incorrect encoding of typedef type");
      return QualType();
    }
    unsigned Idx = 0;
    TypedefNameDecl *Decl = ReadDeclAs<TypedefNameDecl>(*Loc.F, Record, Idx);
    QualType Canonical = readType(*Loc.F, Record, Idx);
    if (!Canonical.isNull())
      Canonical = Context->getCanonicalType(Canonical);
    return Context->getTypedefType(Decl, Canonical);
  }

  case TYPE_TYPEOF_EXPR:
    return Context->getTypeOfExprType(ReadExpr(*Loc.F));

  case TYPE_TYPEOF: {
    if (Record.size() != 1) {
      Error("incorrect encoding of typeof(type) in AST file");
      return QualType();
    }
    QualType UnderlyingType = readType(*Loc.F, Record, Idx);
    return Context->getTypeOfType(UnderlyingType);
  }

  case TYPE_DECLTYPE:
    return Context->getDecltypeType(ReadExpr(*Loc.F));

  case TYPE_UNARY_TRANSFORM: {
    QualType BaseType = readType(*Loc.F, Record, Idx);
    QualType UnderlyingType = readType(*Loc.F, Record, Idx);
    UnaryTransformType::UTTKind UKind = (UnaryTransformType::UTTKind)Record[2];
    return Context->getUnaryTransformType(BaseType, UnderlyingType, UKind);
  }

  case TYPE_AUTO:
    return Context->getAutoType(readType(*Loc.F, Record, Idx));

  case TYPE_RECORD: {
    if (Record.size() != 2) {
      Error("incorrect encoding of record type");
      return QualType();
    }
    unsigned Idx = 0;
    bool IsDependent = Record[Idx++];
    QualType T
      = Context->getRecordType(ReadDeclAs<RecordDecl>(*Loc.F, Record, Idx));
    const_cast<Type*>(T.getTypePtr())->setDependent(IsDependent);
    return T;
  }

  case TYPE_ENUM: {
    if (Record.size() != 2) {
      Error("incorrect encoding of enum type");
      return QualType();
    }
    unsigned Idx = 0;
    bool IsDependent = Record[Idx++];
    QualType T
      = Context->getEnumType(ReadDeclAs<EnumDecl>(*Loc.F, Record, Idx));
    const_cast<Type*>(T.getTypePtr())->setDependent(IsDependent);
    return T;
  }

  case TYPE_ATTRIBUTED: {
    if (Record.size() != 3) {
      Error("incorrect encoding of attributed type");
      return QualType();
    }
    QualType modifiedType = readType(*Loc.F, Record, Idx);
    QualType equivalentType = readType(*Loc.F, Record, Idx);
    AttributedType::Kind kind = static_cast<AttributedType::Kind>(Record[2]);
    return Context->getAttributedType(kind, modifiedType, equivalentType);
  }

  case TYPE_PAREN: {
    if (Record.size() != 1) {
      Error("incorrect encoding of paren type");
      return QualType();
    }
    QualType InnerType = readType(*Loc.F, Record, Idx);
    return Context->getParenType(InnerType);
  }

  case TYPE_PACK_EXPANSION: {
    if (Record.size() != 2) {
      Error("incorrect encoding of pack expansion type");
      return QualType();
    }
    QualType Pattern = readType(*Loc.F, Record, Idx);
    if (Pattern.isNull())
      return QualType();
    llvm::Optional<unsigned> NumExpansions;
    if (Record[1])
      NumExpansions = Record[1] - 1;
    return Context->getPackExpansionType(Pattern, NumExpansions);
  }

  case TYPE_ELABORATED: {
    unsigned Idx = 0;
    ElaboratedTypeKeyword Keyword = (ElaboratedTypeKeyword)Record[Idx++];
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(*Loc.F, Record, Idx);
    QualType NamedType = readType(*Loc.F, Record, Idx);
    return Context->getElaboratedType(Keyword, NNS, NamedType);
  }

  case TYPE_OBJC_INTERFACE: {
    unsigned Idx = 0;
    ObjCInterfaceDecl *ItfD
      = ReadDeclAs<ObjCInterfaceDecl>(*Loc.F, Record, Idx);
    return Context->getObjCInterfaceType(ItfD);
  }

  case TYPE_OBJC_OBJECT: {
    unsigned Idx = 0;
    QualType Base = readType(*Loc.F, Record, Idx);
    unsigned NumProtos = Record[Idx++];
    SmallVector<ObjCProtocolDecl*, 4> Protos;
    for (unsigned I = 0; I != NumProtos; ++I)
      Protos.push_back(ReadDeclAs<ObjCProtocolDecl>(*Loc.F, Record, Idx));
    return Context->getObjCObjectType(Base, Protos.data(), NumProtos);
  }

  case TYPE_OBJC_OBJECT_POINTER: {
    unsigned Idx = 0;
    QualType Pointee = readType(*Loc.F, Record, Idx);
    return Context->getObjCObjectPointerType(Pointee);
  }

  case TYPE_SUBST_TEMPLATE_TYPE_PARM: {
    unsigned Idx = 0;
    QualType Parm = readType(*Loc.F, Record, Idx);
    QualType Replacement = readType(*Loc.F, Record, Idx);
    return
      Context->getSubstTemplateTypeParmType(cast<TemplateTypeParmType>(Parm),
                                            Replacement);
  }

  case TYPE_SUBST_TEMPLATE_TYPE_PARM_PACK: {
    unsigned Idx = 0;
    QualType Parm = readType(*Loc.F, Record, Idx);
    TemplateArgument ArgPack = ReadTemplateArgument(*Loc.F, Record, Idx);
    return Context->getSubstTemplateTypeParmPackType(
                                               cast<TemplateTypeParmType>(Parm),
                                                     ArgPack);
  }

  case TYPE_INJECTED_CLASS_NAME: {
    CXXRecordDecl *D = ReadDeclAs<CXXRecordDecl>(*Loc.F, Record, Idx);
    QualType TST = readType(*Loc.F, Record, Idx); // probably derivable
    // FIXME: ASTContext::getInjectedClassNameType is not currently suitable
    // for AST reading, too much interdependencies.
    return
      QualType(new (*Context, TypeAlignment) InjectedClassNameType(D, TST), 0);
  }

  case TYPE_TEMPLATE_TYPE_PARM: {
    unsigned Idx = 0;
    unsigned Depth = Record[Idx++];
    unsigned Index = Record[Idx++];
    bool Pack = Record[Idx++];
    TemplateTypeParmDecl *D
      = ReadDeclAs<TemplateTypeParmDecl>(*Loc.F, Record, Idx);
    return Context->getTemplateTypeParmType(Depth, Index, Pack, D);
  }

  case TYPE_DEPENDENT_NAME: {
    unsigned Idx = 0;
    ElaboratedTypeKeyword Keyword = (ElaboratedTypeKeyword)Record[Idx++];
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(*Loc.F, Record, Idx);
    const IdentifierInfo *Name = this->GetIdentifierInfo(*Loc.F, Record, Idx);
    QualType Canon = readType(*Loc.F, Record, Idx);
    if (!Canon.isNull())
      Canon = Context->getCanonicalType(Canon);
    return Context->getDependentNameType(Keyword, NNS, Name, Canon);
  }

  case TYPE_DEPENDENT_TEMPLATE_SPECIALIZATION: {
    unsigned Idx = 0;
    ElaboratedTypeKeyword Keyword = (ElaboratedTypeKeyword)Record[Idx++];
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(*Loc.F, Record, Idx);
    const IdentifierInfo *Name = this->GetIdentifierInfo(*Loc.F, Record, Idx);
    unsigned NumArgs = Record[Idx++];
    SmallVector<TemplateArgument, 8> Args;
    Args.reserve(NumArgs);
    while (NumArgs--)
      Args.push_back(ReadTemplateArgument(*Loc.F, Record, Idx));
    return Context->getDependentTemplateSpecializationType(Keyword, NNS, Name,
                                                      Args.size(), Args.data());
  }

  case TYPE_DEPENDENT_SIZED_ARRAY: {
    unsigned Idx = 0;

    // ArrayType
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM
      = (ArrayType::ArraySizeModifier)Record[Idx++];
    unsigned IndexTypeQuals = Record[Idx++];

    // DependentSizedArrayType
    Expr *NumElts = ReadExpr(*Loc.F);
    SourceRange Brackets = ReadSourceRange(*Loc.F, Record, Idx);

    return Context->getDependentSizedArrayType(ElementType, NumElts, ASM,
                                               IndexTypeQuals, Brackets);
  }

  case TYPE_TEMPLATE_SPECIALIZATION: {
    unsigned Idx = 0;
    bool IsDependent = Record[Idx++];
    TemplateName Name = ReadTemplateName(*Loc.F, Record, Idx);
    SmallVector<TemplateArgument, 8> Args;
    ReadTemplateArgumentList(Args, *Loc.F, Record, Idx);
    QualType Underlying = readType(*Loc.F, Record, Idx);
    QualType T;
    if (Underlying.isNull())
      T = Context->getCanonicalTemplateSpecializationType(Name, Args.data(),
                                                          Args.size());
    else
      T = Context->getTemplateSpecializationType(Name, Args.data(),
                                                 Args.size(), Underlying);
    const_cast<Type*>(T.getTypePtr())->setDependent(IsDependent);
    return T;
  }
  }
  // Suppress a GCC warning
  return QualType();
}

class clang::TypeLocReader : public TypeLocVisitor<TypeLocReader> {
  ASTReader &Reader;
  Module &F;
  llvm::BitstreamCursor &DeclsCursor;
  const ASTReader::RecordData &Record;
  unsigned &Idx;

  SourceLocation ReadSourceLocation(const ASTReader::RecordData &R,
                                    unsigned &I) {
    return Reader.ReadSourceLocation(F, R, I);
  }

  template<typename T>
  T *ReadDeclAs(const ASTReader::RecordData &Record, unsigned &Idx) {
    return Reader.ReadDeclAs<T>(F, Record, Idx);
  }
  
public:
  TypeLocReader(ASTReader &Reader, Module &F,
                const ASTReader::RecordData &Record, unsigned &Idx)
    : Reader(Reader), F(F), DeclsCursor(F.DeclsCursor), Record(Record), Idx(Idx)
  { }

  // We want compile-time assurance that we've enumerated all of
  // these, so unfortunately we have to declare them first, then
  // define them out-of-line.
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
  void Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc);
#include "clang/AST/TypeLocNodes.def"

  void VisitFunctionTypeLoc(FunctionTypeLoc);
  void VisitArrayTypeLoc(ArrayTypeLoc);
};

void TypeLocReader::VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
  // nothing to do
}
void TypeLocReader::VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
  TL.setBuiltinLoc(ReadSourceLocation(Record, Idx));
  if (TL.needsExtraLocalData()) {
    TL.setWrittenTypeSpec(static_cast<DeclSpec::TST>(Record[Idx++]));
    TL.setWrittenSignSpec(static_cast<DeclSpec::TSS>(Record[Idx++]));
    TL.setWrittenWidthSpec(static_cast<DeclSpec::TSW>(Record[Idx++]));
    TL.setModeAttr(Record[Idx++]);
  }
}
void TypeLocReader::VisitComplexTypeLoc(ComplexTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitPointerTypeLoc(PointerTypeLoc TL) {
  TL.setStarLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
  TL.setCaretLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
  TL.setAmpLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
  TL.setAmpAmpLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
  TL.setStarLoc(ReadSourceLocation(Record, Idx));
  TL.setClassTInfo(Reader.GetTypeSourceInfo(F, Record, Idx));
}
void TypeLocReader::VisitArrayTypeLoc(ArrayTypeLoc TL) {
  TL.setLBracketLoc(ReadSourceLocation(Record, Idx));
  TL.setRBracketLoc(ReadSourceLocation(Record, Idx));
  if (Record[Idx++])
    TL.setSizeExpr(Reader.ReadExpr(F));
  else
    TL.setSizeExpr(0);
}
void TypeLocReader::VisitConstantArrayTypeLoc(ConstantArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitIncompleteArrayTypeLoc(IncompleteArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitVariableArrayTypeLoc(VariableArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitDependentSizedArrayTypeLoc(
                                            DependentSizedArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitDependentSizedExtVectorTypeLoc(
                                        DependentSizedExtVectorTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitVectorTypeLoc(VectorTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitExtVectorTypeLoc(ExtVectorTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitFunctionTypeLoc(FunctionTypeLoc TL) {
  TL.setLocalRangeBegin(ReadSourceLocation(Record, Idx));
  TL.setLocalRangeEnd(ReadSourceLocation(Record, Idx));
  TL.setTrailingReturn(Record[Idx++]);
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i) {
    TL.setArg(i, ReadDeclAs<ParmVarDecl>(Record, Idx));
  }
}
void TypeLocReader::VisitFunctionProtoTypeLoc(FunctionProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocReader::VisitFunctionNoProtoTypeLoc(FunctionNoProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocReader::VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTypedefTypeLoc(TypedefTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
  TL.setTypeofLoc(ReadSourceLocation(Record, Idx));
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
  TL.setTypeofLoc(ReadSourceLocation(Record, Idx));
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
  TL.setUnderlyingTInfo(Reader.GetTypeSourceInfo(F, Record, Idx));
}
void TypeLocReader::VisitDecltypeTypeLoc(DecltypeTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitUnaryTransformTypeLoc(UnaryTransformTypeLoc TL) {
  TL.setKWLoc(ReadSourceLocation(Record, Idx));
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
  TL.setUnderlyingTInfo(Reader.GetTypeSourceInfo(F, Record, Idx));
}
void TypeLocReader::VisitAutoTypeLoc(AutoTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitRecordTypeLoc(RecordTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitEnumTypeLoc(EnumTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitAttributedTypeLoc(AttributedTypeLoc TL) {
  TL.setAttrNameLoc(ReadSourceLocation(Record, Idx));
  if (TL.hasAttrOperand()) {
    SourceRange range;
    range.setBegin(ReadSourceLocation(Record, Idx));
    range.setEnd(ReadSourceLocation(Record, Idx));
    TL.setAttrOperandParensRange(range);
  }
  if (TL.hasAttrExprOperand()) {
    if (Record[Idx++])
      TL.setAttrExprOperand(Reader.ReadExpr(F));
    else
      TL.setAttrExprOperand(0);
  } else if (TL.hasAttrEnumOperand())
    TL.setAttrEnumOperandLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitSubstTemplateTypeParmTypeLoc(
                                            SubstTemplateTypeParmTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitSubstTemplateTypeParmPackTypeLoc(
                                          SubstTemplateTypeParmPackTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTemplateSpecializationTypeLoc(
                                           TemplateSpecializationTypeLoc TL) {
  TL.setTemplateNameLoc(ReadSourceLocation(Record, Idx));
  TL.setLAngleLoc(ReadSourceLocation(Record, Idx));
  TL.setRAngleLoc(ReadSourceLocation(Record, Idx));
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    TL.setArgLocInfo(i,
        Reader.GetTemplateArgumentLocInfo(F,
                                          TL.getTypePtr()->getArg(i).getKind(),
                                          Record, Idx));
}
void TypeLocReader::VisitParenTypeLoc(ParenTypeLoc TL) {
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitElaboratedTypeLoc(ElaboratedTypeLoc TL) {
  TL.setKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setQualifierLoc(Reader.ReadNestedNameSpecifierLoc(F, Record, Idx));
}
void TypeLocReader::VisitInjectedClassNameTypeLoc(InjectedClassNameTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
  TL.setKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setQualifierLoc(Reader.ReadNestedNameSpecifierLoc(F, Record, Idx));
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitDependentTemplateSpecializationTypeLoc(
       DependentTemplateSpecializationTypeLoc TL) {
  TL.setKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setQualifierLoc(Reader.ReadNestedNameSpecifierLoc(F, Record, Idx));
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
  TL.setLAngleLoc(ReadSourceLocation(Record, Idx));
  TL.setRAngleLoc(ReadSourceLocation(Record, Idx));
  for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I)
    TL.setArgLocInfo(I,
        Reader.GetTemplateArgumentLocInfo(F,
                                          TL.getTypePtr()->getArg(I).getKind(),
                                          Record, Idx));
}
void TypeLocReader::VisitPackExpansionTypeLoc(PackExpansionTypeLoc TL) {
  TL.setEllipsisLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
  TL.setHasBaseTypeAsWritten(Record[Idx++]);
  TL.setLAngleLoc(ReadSourceLocation(Record, Idx));
  TL.setRAngleLoc(ReadSourceLocation(Record, Idx));
  for (unsigned i = 0, e = TL.getNumProtocols(); i != e; ++i)
    TL.setProtocolLoc(i, ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
  TL.setStarLoc(ReadSourceLocation(Record, Idx));
}

TypeSourceInfo *ASTReader::GetTypeSourceInfo(Module &F,
                                             const RecordData &Record,
                                             unsigned &Idx) {
  QualType InfoTy = readType(F, Record, Idx);
  if (InfoTy.isNull())
    return 0;

  TypeSourceInfo *TInfo = getContext()->CreateTypeSourceInfo(InfoTy);
  TypeLocReader TLR(*this, F, Record, Idx);
  for (TypeLoc TL = TInfo->getTypeLoc(); !TL.isNull(); TL = TL.getNextTypeLoc())
    TLR.Visit(TL);
  return TInfo;
}

QualType ASTReader::GetType(TypeID ID) {
  unsigned FastQuals = ID & Qualifiers::FastMask;
  unsigned Index = ID >> Qualifiers::FastWidth;

  if (Index < NUM_PREDEF_TYPE_IDS) {
    QualType T;
    switch ((PredefinedTypeIDs)Index) {
    case PREDEF_TYPE_NULL_ID: return QualType();
    case PREDEF_TYPE_VOID_ID: T = Context->VoidTy; break;
    case PREDEF_TYPE_BOOL_ID: T = Context->BoolTy; break;

    case PREDEF_TYPE_CHAR_U_ID:
    case PREDEF_TYPE_CHAR_S_ID:
      // FIXME: Check that the signedness of CharTy is correct!
      T = Context->CharTy;
      break;

    case PREDEF_TYPE_UCHAR_ID:      T = Context->UnsignedCharTy;     break;
    case PREDEF_TYPE_USHORT_ID:     T = Context->UnsignedShortTy;    break;
    case PREDEF_TYPE_UINT_ID:       T = Context->UnsignedIntTy;      break;
    case PREDEF_TYPE_ULONG_ID:      T = Context->UnsignedLongTy;     break;
    case PREDEF_TYPE_ULONGLONG_ID:  T = Context->UnsignedLongLongTy; break;
    case PREDEF_TYPE_UINT128_ID:    T = Context->UnsignedInt128Ty;   break;
    case PREDEF_TYPE_SCHAR_ID:      T = Context->SignedCharTy;       break;
    case PREDEF_TYPE_WCHAR_ID:      T = Context->WCharTy;            break;
    case PREDEF_TYPE_SHORT_ID:      T = Context->ShortTy;            break;
    case PREDEF_TYPE_INT_ID:        T = Context->IntTy;              break;
    case PREDEF_TYPE_LONG_ID:       T = Context->LongTy;             break;
    case PREDEF_TYPE_LONGLONG_ID:   T = Context->LongLongTy;         break;
    case PREDEF_TYPE_INT128_ID:     T = Context->Int128Ty;           break;
    case PREDEF_TYPE_FLOAT_ID:      T = Context->FloatTy;            break;
    case PREDEF_TYPE_DOUBLE_ID:     T = Context->DoubleTy;           break;
    case PREDEF_TYPE_LONGDOUBLE_ID: T = Context->LongDoubleTy;       break;
    case PREDEF_TYPE_OVERLOAD_ID:   T = Context->OverloadTy;         break;
    case PREDEF_TYPE_BOUND_MEMBER:  T = Context->BoundMemberTy;      break;
    case PREDEF_TYPE_DEPENDENT_ID:  T = Context->DependentTy;        break;
    case PREDEF_TYPE_UNKNOWN_ANY:   T = Context->UnknownAnyTy;       break;
    case PREDEF_TYPE_NULLPTR_ID:    T = Context->NullPtrTy;          break;
    case PREDEF_TYPE_CHAR16_ID:     T = Context->Char16Ty;           break;
    case PREDEF_TYPE_CHAR32_ID:     T = Context->Char32Ty;           break;
    case PREDEF_TYPE_OBJC_ID:       T = Context->ObjCBuiltinIdTy;    break;
    case PREDEF_TYPE_OBJC_CLASS:    T = Context->ObjCBuiltinClassTy; break;
    case PREDEF_TYPE_OBJC_SEL:      T = Context->ObjCBuiltinSelTy;   break;
    }

    assert(!T.isNull() && "Unknown predefined type");
    return T.withFastQualifiers(FastQuals);
  }

  Index -= NUM_PREDEF_TYPE_IDS;
  assert(Index < TypesLoaded.size() && "Type index out-of-range");
  if (TypesLoaded[Index].isNull()) {
    TypesLoaded[Index] = readTypeRecord(Index);
    if (TypesLoaded[Index].isNull())
      return QualType();

    TypesLoaded[Index]->setFromAST();
    if (DeserializationListener)
      DeserializationListener->TypeRead(TypeIdx::fromTypeID(ID),
                                        TypesLoaded[Index]);
  }

  return TypesLoaded[Index].withFastQualifiers(FastQuals);
}

QualType ASTReader::getLocalType(Module &F, unsigned LocalID) {
  return GetType(getGlobalTypeID(F, LocalID));
}

serialization::TypeID 
ASTReader::getGlobalTypeID(Module &F, unsigned LocalID) const {
  unsigned FastQuals = LocalID & Qualifiers::FastMask;
  unsigned LocalIndex = LocalID >> Qualifiers::FastWidth;
  
  if (LocalIndex < NUM_PREDEF_TYPE_IDS)
    return LocalID;

  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = F.TypeRemap.find(LocalIndex - NUM_PREDEF_TYPE_IDS);
  assert(I != F.TypeRemap.end() && "Invalid index into type index remap");
  
  unsigned GlobalIndex = LocalIndex + I->second;
  return (GlobalIndex << Qualifiers::FastWidth) | FastQuals;
}

TemplateArgumentLocInfo
ASTReader::GetTemplateArgumentLocInfo(Module &F,
                                      TemplateArgument::ArgKind Kind,
                                      const RecordData &Record,
                                      unsigned &Index) {
  switch (Kind) {
  case TemplateArgument::Expression:
    return ReadExpr(F);
  case TemplateArgument::Type:
    return GetTypeSourceInfo(F, Record, Index);
  case TemplateArgument::Template: {
    NestedNameSpecifierLoc QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, 
                                                                     Index);
    SourceLocation TemplateNameLoc = ReadSourceLocation(F, Record, Index);
    return TemplateArgumentLocInfo(QualifierLoc, TemplateNameLoc,
                                   SourceLocation());
  }
  case TemplateArgument::TemplateExpansion: {
    NestedNameSpecifierLoc QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, 
                                                                     Index);
    SourceLocation TemplateNameLoc = ReadSourceLocation(F, Record, Index);
    SourceLocation EllipsisLoc = ReadSourceLocation(F, Record, Index);
    return TemplateArgumentLocInfo(QualifierLoc, TemplateNameLoc, 
                                   EllipsisLoc);
  }
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
  case TemplateArgument::Declaration:
  case TemplateArgument::Pack:
    return TemplateArgumentLocInfo();
  }
  llvm_unreachable("unexpected template argument loc");
  return TemplateArgumentLocInfo();
}

TemplateArgumentLoc
ASTReader::ReadTemplateArgumentLoc(Module &F,
                                   const RecordData &Record, unsigned &Index) {
  TemplateArgument Arg = ReadTemplateArgument(F, Record, Index);

  if (Arg.getKind() == TemplateArgument::Expression) {
    if (Record[Index++]) // bool InfoHasSameExpr.
      return TemplateArgumentLoc(Arg, TemplateArgumentLocInfo(Arg.getAsExpr()));
  }
  return TemplateArgumentLoc(Arg, GetTemplateArgumentLocInfo(F, Arg.getKind(),
                                                             Record, Index));
}

Decl *ASTReader::GetExternalDecl(uint32_t ID) {
  return GetDecl(ID);
}

uint64_t ASTReader::readCXXBaseSpecifiers(Module &M, const RecordData &Record, 
                                          unsigned &Idx){
  if (Idx >= Record.size())
    return 0;
  
  unsigned LocalID = Record[Idx++];
  return getGlobalBitOffset(M, M.CXXBaseSpecifiersOffsets[LocalID - 1]);
}

CXXBaseSpecifier *ASTReader::GetExternalCXXBaseSpecifiers(uint64_t Offset) {
  RecordLocation Loc = getLocalBitOffset(Offset);
  llvm::BitstreamCursor &Cursor = Loc.F->DeclsCursor;
  SavedStreamPosition SavedPosition(Cursor);
  Cursor.JumpToBit(Loc.Offset);
  ReadingKindTracker ReadingKind(Read_Decl, *this);
  RecordData Record;
  unsigned Code = Cursor.ReadCode();
  unsigned RecCode = Cursor.ReadRecord(Code, Record);
  if (RecCode != DECL_CXX_BASE_SPECIFIERS) {
    Error("Malformed AST file: missing C++ base specifiers");
    return 0;
  }

  unsigned Idx = 0;
  unsigned NumBases = Record[Idx++];
  void *Mem = Context->Allocate(sizeof(CXXBaseSpecifier) * NumBases);
  CXXBaseSpecifier *Bases = new (Mem) CXXBaseSpecifier [NumBases];
  for (unsigned I = 0; I != NumBases; ++I)
    Bases[I] = ReadCXXBaseSpecifier(*Loc.F, Record, Idx);
  return Bases;
}

TranslationUnitDecl *ASTReader::GetTranslationUnitDecl() {
  // FIXME: This routine might not even make sense when we're loading multiple
  // unrelated AST files, since we'll have to merge the translation units
  // somehow.
  unsigned TranslationUnitID = (*(ModuleMgr.end() - 1))->BaseDeclID + 1;
  if (!DeclsLoaded[TranslationUnitID - 1]) {
    ReadDeclRecord(TranslationUnitID);
    if (DeserializationListener)
      DeserializationListener->DeclRead(TranslationUnitID, 
                                        DeclsLoaded[TranslationUnitID - 1]);
  }

  return cast<TranslationUnitDecl>(DeclsLoaded[TranslationUnitID - 1]);
}

serialization::DeclID 
ASTReader::getGlobalDeclID(Module &F, unsigned LocalID) const {
  if (LocalID < NUM_PREDEF_DECL_IDS)
    return LocalID;

  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = F.DeclRemap.find(LocalID - NUM_PREDEF_DECL_IDS);
  assert(I != F.DeclRemap.end() && "Invalid index into decl index remap");
  
  return LocalID + I->second;
}

Decl *ASTReader::GetDecl(DeclID ID) {
  if (ID < NUM_PREDEF_DECL_IDS) {    
    switch ((PredefinedDeclIDs)ID) {
    case serialization::PREDEF_DECL_NULL_ID:
      return 0;
    }
    
    return 0;
  }
  
  if (ID > DeclsLoaded.size()) {
    Error("declaration ID out-of-range for AST file");
    return 0;
  }

  unsigned Index = ID - NUM_PREDEF_DECL_IDS;
  if (!DeclsLoaded[Index]) {
    ReadDeclRecord(ID);
    if (DeserializationListener)
      DeserializationListener->DeclRead(ID, DeclsLoaded[Index]);
  }

  return DeclsLoaded[Index];
}

serialization::DeclID ASTReader::ReadDeclID(Module &F, 
                                            const RecordData &Record,
                                            unsigned &Idx) {
  if (Idx >= Record.size()) {
    Error("Corrupted AST file");
    return 0;
  }
  
  return getGlobalDeclID(F, Record[Idx++]);
}

/// \brief Resolve the offset of a statement into a statement.
///
/// This operation will read a new statement from the external
/// source each time it is called, and is meant to be used via a
/// LazyOffsetPtr (which is used by Decls for the body of functions, etc).
Stmt *ASTReader::GetExternalDeclStmt(uint64_t Offset) {
  // Switch case IDs are per Decl.
  ClearSwitchCaseIDs();

  // Offset here is a global offset across the entire chain.
  RecordLocation Loc = getLocalBitOffset(Offset);
  Loc.F->DeclsCursor.JumpToBit(Loc.Offset);
  return ReadStmtFromStream(*Loc.F);
}

ExternalLoadResult ASTReader::FindExternalLexicalDecls(const DeclContext *DC,
                                         bool (*isKindWeWant)(Decl::Kind),
                                         SmallVectorImpl<Decl*> &Decls) {
  // There might be lexical decls in multiple parts of the chain, for the TU
  // at least.
  // DeclContextOffsets might reallocate as we load additional decls below,
  // so make a copy of the vector.
  DeclContextInfos Infos = DeclContextOffsets[DC];
  for (DeclContextInfos::iterator I = Infos.begin(), E = Infos.end();
       I != E; ++I) {
    // IDs can be 0 if this context doesn't contain declarations.
    if (!I->LexicalDecls)
      continue;

    // Load all of the declaration IDs
    for (const KindDeclIDPair *ID = I->LexicalDecls,
                              *IDE = ID + I->NumLexicalDecls; ID != IDE; ++ID) {
      if (isKindWeWant && !isKindWeWant((Decl::Kind)ID->first))
        continue;
      
      Decl *D = GetLocalDecl(*I->F, ID->second);
      assert(D && "Null decl in lexical decls");
      Decls.push_back(D);
    }
  }

  ++NumLexicalDeclContextsRead;
  return ELR_Success;
}

DeclContext::lookup_result
ASTReader::FindExternalVisibleDeclsByName(const DeclContext *DC,
                                          DeclarationName Name) {
  assert(DC->hasExternalVisibleStorage() &&
         "DeclContext has no visible decls in storage");
  if (!Name)
    return DeclContext::lookup_result(DeclContext::lookup_iterator(0),
                                      DeclContext::lookup_iterator(0));

  SmallVector<NamedDecl *, 64> Decls;
  // There might be visible decls in multiple parts of the chain, for the TU
  // and namespaces. For any given name, the last available results replace
  // all earlier ones. For this reason, we walk in reverse.
  DeclContextInfos &Infos = DeclContextOffsets[DC];
  for (DeclContextInfos::reverse_iterator I = Infos.rbegin(), E = Infos.rend();
       I != E; ++I) {
    if (!I->NameLookupTableData)
      continue;

    ASTDeclContextNameLookupTable *LookupTable =
        (ASTDeclContextNameLookupTable*)I->NameLookupTableData;
    ASTDeclContextNameLookupTable::iterator Pos = LookupTable->find(Name);
    if (Pos == LookupTable->end())
      continue;

    ASTDeclContextNameLookupTrait::data_type Data = *Pos;
    for (; Data.first != Data.second; ++Data.first) {
      NamedDecl *ND = GetLocalDeclAs<NamedDecl>(*I->F, *Data.first);
      if (!ND)
        continue;
      
      if (ND->getDeclName() != Name) {
        assert(!Name.getCXXNameType().isNull() && 
               "Name mismatch without a type");
        continue;
      }
      
      Decls.push_back(ND);
    }
    
    // If we rejected all of the declarations we found, e.g., because the
    // name didn't actually match, continue looking through DeclContexts.
    if (Decls.empty())
      continue;
    
    break;
  }

  ++NumVisibleDeclContextsRead;

  SetExternalVisibleDeclsForName(DC, Name, Decls);
  return const_cast<DeclContext*>(DC)->lookup(Name);
}

void ASTReader::MaterializeVisibleDecls(const DeclContext *DC) {
  assert(DC->hasExternalVisibleStorage() &&
         "DeclContext has no visible decls in storage");

  SmallVector<NamedDecl *, 64> Decls;
  // There might be visible decls in multiple parts of the chain, for the TU
  // and namespaces.
  DeclContextInfos &Infos = DeclContextOffsets[DC];
  for (DeclContextInfos::iterator I = Infos.begin(), E = Infos.end();
       I != E; ++I) {
    if (!I->NameLookupTableData)
      continue;

    ASTDeclContextNameLookupTable *LookupTable =
        (ASTDeclContextNameLookupTable*)I->NameLookupTableData;
    for (ASTDeclContextNameLookupTable::item_iterator
           ItemI = LookupTable->item_begin(),
           ItemEnd = LookupTable->item_end() ; ItemI != ItemEnd; ++ItemI) {
      ASTDeclContextNameLookupTable::item_iterator::value_type Val
          = *ItemI;
      ASTDeclContextNameLookupTrait::data_type Data = Val.second;
      Decls.clear();
      for (; Data.first != Data.second; ++Data.first)
        Decls.push_back(GetLocalDeclAs<NamedDecl>(*I->F, *Data.first));
      MaterializeVisibleDeclsForName(DC, Val.first, Decls);
    }
  }
}

void ASTReader::PassInterestingDeclsToConsumer() {
  assert(Consumer);
  while (!InterestingDecls.empty()) {
    DeclGroupRef DG(InterestingDecls.front());
    InterestingDecls.pop_front();
    Consumer->HandleInterestingDecl(DG);
  }
}

void ASTReader::StartTranslationUnit(ASTConsumer *Consumer) {
  this->Consumer = Consumer;

  if (!Consumer)
    return;

  for (unsigned I = 0, N = ExternalDefinitions.size(); I != N; ++I) {
    // Force deserialization of this decl, which will cause it to be queued for
    // passing to the consumer.
    GetDecl(ExternalDefinitions[I]);
  }

  PassInterestingDeclsToConsumer();
}

void ASTReader::PrintStats() {
  std::fprintf(stderr, "*** AST File Statistics:\n");

  unsigned NumTypesLoaded
    = TypesLoaded.size() - std::count(TypesLoaded.begin(), TypesLoaded.end(),
                                      QualType());
  unsigned NumDeclsLoaded
    = DeclsLoaded.size() - std::count(DeclsLoaded.begin(), DeclsLoaded.end(),
                                      (Decl *)0);
  unsigned NumIdentifiersLoaded
    = IdentifiersLoaded.size() - std::count(IdentifiersLoaded.begin(),
                                            IdentifiersLoaded.end(),
                                            (IdentifierInfo *)0);
  unsigned NumSelectorsLoaded
    = SelectorsLoaded.size() - std::count(SelectorsLoaded.begin(),
                                          SelectorsLoaded.end(),
                                          Selector());

  std::fprintf(stderr, "  %u stat cache hits\n", NumStatHits);
  std::fprintf(stderr, "  %u stat cache misses\n", NumStatMisses);
  if (unsigned TotalNumSLocEntries = getTotalNumSLocs())
    std::fprintf(stderr, "  %u/%u source location entries read (%f%%)\n",
                 NumSLocEntriesRead, TotalNumSLocEntries,
                 ((float)NumSLocEntriesRead/TotalNumSLocEntries * 100));
  if (!TypesLoaded.empty())
    std::fprintf(stderr, "  %u/%u types read (%f%%)\n",
                 NumTypesLoaded, (unsigned)TypesLoaded.size(),
                 ((float)NumTypesLoaded/TypesLoaded.size() * 100));
  if (!DeclsLoaded.empty())
    std::fprintf(stderr, "  %u/%u declarations read (%f%%)\n",
                 NumDeclsLoaded, (unsigned)DeclsLoaded.size(),
                 ((float)NumDeclsLoaded/DeclsLoaded.size() * 100));
  if (!IdentifiersLoaded.empty())
    std::fprintf(stderr, "  %u/%u identifiers read (%f%%)\n",
                 NumIdentifiersLoaded, (unsigned)IdentifiersLoaded.size(),
                 ((float)NumIdentifiersLoaded/IdentifiersLoaded.size() * 100));
  if (!SelectorsLoaded.empty())
    std::fprintf(stderr, "  %u/%u selectors read (%f%%)\n",
                 NumSelectorsLoaded, (unsigned)SelectorsLoaded.size(),
                 ((float)NumSelectorsLoaded/SelectorsLoaded.size() * 100));
  if (TotalNumStatements)
    std::fprintf(stderr, "  %u/%u statements read (%f%%)\n",
                 NumStatementsRead, TotalNumStatements,
                 ((float)NumStatementsRead/TotalNumStatements * 100));
  if (TotalNumMacros)
    std::fprintf(stderr, "  %u/%u macros read (%f%%)\n",
                 NumMacrosRead, TotalNumMacros,
                 ((float)NumMacrosRead/TotalNumMacros * 100));
  if (TotalLexicalDeclContexts)
    std::fprintf(stderr, "  %u/%u lexical declcontexts read (%f%%)\n",
                 NumLexicalDeclContextsRead, TotalLexicalDeclContexts,
                 ((float)NumLexicalDeclContextsRead/TotalLexicalDeclContexts
                  * 100));
  if (TotalVisibleDeclContexts)
    std::fprintf(stderr, "  %u/%u visible declcontexts read (%f%%)\n",
                 NumVisibleDeclContextsRead, TotalVisibleDeclContexts,
                 ((float)NumVisibleDeclContextsRead/TotalVisibleDeclContexts
                  * 100));
  if (TotalNumMethodPoolEntries) {
    std::fprintf(stderr, "  %u/%u method pool entries read (%f%%)\n",
                 NumMethodPoolEntriesRead, TotalNumMethodPoolEntries,
                 ((float)NumMethodPoolEntriesRead/TotalNumMethodPoolEntries
                  * 100));
    std::fprintf(stderr, "  %u method pool misses\n", NumMethodPoolMisses);
  }
  std::fprintf(stderr, "\n");
  dump();
  std::fprintf(stderr, "\n");
}

template<typename Key, typename Module, unsigned InitialCapacity>
static void 
dumpModuleIDMap(StringRef Name,
                const ContinuousRangeMap<Key, Module *, 
                                         InitialCapacity> &Map) {
  if (Map.begin() == Map.end())
    return;
  
  typedef ContinuousRangeMap<Key, Module *, InitialCapacity> MapType;
  llvm::errs() << Name << ":\n";
  for (typename MapType::const_iterator I = Map.begin(), IEnd = Map.end(); 
       I != IEnd; ++I) {
    llvm::errs() << "  " << I->first << " -> " << I->second->FileName
      << "\n";
  }
}

void ASTReader::dump() {
  llvm::errs() << "*** PCH/Module Remappings:\n";
  dumpModuleIDMap("Global bit offset map", GlobalBitOffsetsMap);
  dumpModuleIDMap("Global source location entry map", GlobalSLocEntryMap);
  dumpModuleIDMap("Global type map", GlobalTypeMap);
  dumpModuleIDMap("Global declaration map", GlobalDeclMap);
  dumpModuleIDMap("Global identifier map", GlobalIdentifierMap);
  dumpModuleIDMap("Global selector map", GlobalSelectorMap);
  dumpModuleIDMap("Global macro definition map", GlobalMacroDefinitionMap);
  dumpModuleIDMap("Global preprocessed entity map", 
                  GlobalPreprocessedEntityMap);
  
  llvm::errs() << "\n*** PCH/Modules Loaded:";
  for (ModuleManager::ModuleConstIterator M = ModuleMgr.begin(), 
                                       MEnd = ModuleMgr.end();
       M != MEnd; ++M)
    (*M)->dump();
}

/// Return the amount of memory used by memory buffers, breaking down
/// by heap-backed versus mmap'ed memory.
void ASTReader::getMemoryBufferSizes(MemoryBufferSizes &sizes) const {
  for (ModuleConstIterator I = ModuleMgr.begin(),
      E = ModuleMgr.end(); I != E; ++I) {
    if (llvm::MemoryBuffer *buf = (*I)->Buffer.get()) {
      size_t bytes = buf->getBufferSize();
      switch (buf->getBufferKind()) {
        case llvm::MemoryBuffer::MemoryBuffer_Malloc:
          sizes.malloc_bytes += bytes;
          break;
        case llvm::MemoryBuffer::MemoryBuffer_MMap:
          sizes.mmap_bytes += bytes;
          break;
      }
    }
  }
}

void ASTReader::InitializeSema(Sema &S) {
  SemaObj = &S;
  S.ExternalSource = this;

  // Makes sure any declarations that were deserialized "too early"
  // still get added to the identifier's declaration chains.
  for (unsigned I = 0, N = PreloadedDecls.size(); I != N; ++I) {
    if (SemaObj->TUScope)
      SemaObj->TUScope->AddDecl(PreloadedDecls[I]);

    SemaObj->IdResolver.AddDecl(PreloadedDecls[I]);
  }
  PreloadedDecls.clear();

  // Load the offsets of the declarations that Sema references.
  // They will be lazily deserialized when needed.
  if (!SemaDeclRefs.empty()) {
    assert(SemaDeclRefs.size() == 2 && "More decl refs than expected!");
    if (!SemaObj->StdNamespace)
      SemaObj->StdNamespace = SemaDeclRefs[0];
    if (!SemaObj->StdBadAlloc)
      SemaObj->StdBadAlloc = SemaDeclRefs[1];
  }

  if (!FPPragmaOptions.empty()) {
    assert(FPPragmaOptions.size() == 1 && "Wrong number of FP_PRAGMA_OPTIONS");
    SemaObj->FPFeatures.fp_contract = FPPragmaOptions[0];
  }

  if (!OpenCLExtensions.empty()) {
    unsigned I = 0;
#define OPENCLEXT(nm)  SemaObj->OpenCLFeatures.nm = OpenCLExtensions[I++];
#include "clang/Basic/OpenCLExtensions.def"

    assert(OpenCLExtensions.size() == I && "Wrong number of OPENCL_EXTENSIONS");
  }
}

IdentifierInfo* ASTReader::get(const char *NameStart, const char *NameEnd) {
  // Try to find this name within our on-disk hash tables. We start with the
  // most recent one, since that one contains the most up-to-date info.
  for (ModuleIterator I = ModuleMgr.begin(), E = ModuleMgr.end(); I != E; ++I) {
    ASTIdentifierLookupTable *IdTable
        = (ASTIdentifierLookupTable *)(*I)->IdentifierLookupTable;
    if (!IdTable)
      continue;
    std::pair<const char*, unsigned> Key(NameStart, NameEnd - NameStart);
    ASTIdentifierLookupTable::iterator Pos = IdTable->find(Key);
    if (Pos == IdTable->end())
      continue;

    // Dereferencing the iterator has the effect of building the
    // IdentifierInfo node and populating it with the various
    // declarations it needs.
    return *Pos;
  }
  return 0;
}

namespace clang {
  /// \brief An identifier-lookup iterator that enumerates all of the
  /// identifiers stored within a set of AST files.
  class ASTIdentifierIterator : public IdentifierIterator {
    /// \brief The AST reader whose identifiers are being enumerated.
    const ASTReader &Reader;

    /// \brief The current index into the chain of AST files stored in
    /// the AST reader.
    unsigned Index;

    /// \brief The current position within the identifier lookup table
    /// of the current AST file.
    ASTIdentifierLookupTable::key_iterator Current;

    /// \brief The end position within the identifier lookup table of
    /// the current AST file.
    ASTIdentifierLookupTable::key_iterator End;

  public:
    explicit ASTIdentifierIterator(const ASTReader &Reader);

    virtual StringRef Next();
  };
}

ASTIdentifierIterator::ASTIdentifierIterator(const ASTReader &Reader)
  : Reader(Reader), Index(Reader.ModuleMgr.size() - 1) {
  ASTIdentifierLookupTable *IdTable
    = (ASTIdentifierLookupTable *)Reader.ModuleMgr[Index].IdentifierLookupTable;
  Current = IdTable->key_begin();
  End = IdTable->key_end();
}

StringRef ASTIdentifierIterator::Next() {
  while (Current == End) {
    // If we have exhausted all of our AST files, we're done.
    if (Index == 0)
      return StringRef();

    --Index;
    ASTIdentifierLookupTable *IdTable
      = (ASTIdentifierLookupTable *)Reader.ModuleMgr[Index].
        IdentifierLookupTable;
    Current = IdTable->key_begin();
    End = IdTable->key_end();
  }

  // We have any identifiers remaining in the current AST file; return
  // the next one.
  std::pair<const char*, unsigned> Key = *Current;
  ++Current;
  return StringRef(Key.first, Key.second);
}

IdentifierIterator *ASTReader::getIdentifiers() const {
  return new ASTIdentifierIterator(*this);
}

std::pair<ObjCMethodList, ObjCMethodList>
ASTReader::ReadMethodPool(Selector Sel) {
  // Find this selector in a hash table. We want to find the most recent entry.
  for (ModuleIterator I = ModuleMgr.begin(), E = ModuleMgr.end(); I != E; ++I) {
    Module &F = *(*I);
    if (!F.SelectorLookupTable)
      continue;

    ASTSelectorLookupTable *PoolTable
      = (ASTSelectorLookupTable*)F.SelectorLookupTable;
    ASTSelectorLookupTable::iterator Pos = PoolTable->find(Sel);
    if (Pos != PoolTable->end()) {
      ++NumSelectorsRead;
      // FIXME: Not quite happy with the statistics here. We probably should
      // disable this tracking when called via LoadSelector.
      // Also, should entries without methods count as misses?
      ++NumMethodPoolEntriesRead;
      ASTSelectorLookupTrait::data_type Data = *Pos;
      if (DeserializationListener)
        DeserializationListener->SelectorRead(Data.ID, Sel);
      return std::make_pair(Data.Instance, Data.Factory);
    }
  }

  ++NumMethodPoolMisses;
  return std::pair<ObjCMethodList, ObjCMethodList>();
}

void ASTReader::ReadKnownNamespaces(
                          SmallVectorImpl<NamespaceDecl *> &Namespaces) {
  Namespaces.clear();
  
  for (unsigned I = 0, N = KnownNamespaces.size(); I != N; ++I) {
    if (NamespaceDecl *Namespace 
                = dyn_cast_or_null<NamespaceDecl>(GetDecl(KnownNamespaces[I])))
      Namespaces.push_back(Namespace);
  }
}

void ASTReader::ReadTentativeDefinitions(
                  SmallVectorImpl<VarDecl *> &TentativeDefs) {
  for (unsigned I = 0, N = TentativeDefinitions.size(); I != N; ++I) {
    VarDecl *Var = dyn_cast_or_null<VarDecl>(GetDecl(TentativeDefinitions[I]));
    if (Var)
      TentativeDefs.push_back(Var);
  }
  TentativeDefinitions.clear();
}

void ASTReader::ReadUnusedFileScopedDecls(
                               SmallVectorImpl<const DeclaratorDecl *> &Decls) {
  for (unsigned I = 0, N = UnusedFileScopedDecls.size(); I != N; ++I) {
    DeclaratorDecl *D
      = dyn_cast_or_null<DeclaratorDecl>(GetDecl(UnusedFileScopedDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  UnusedFileScopedDecls.clear();
}

void ASTReader::ReadDelegatingConstructors(
                                 SmallVectorImpl<CXXConstructorDecl *> &Decls) {
  for (unsigned I = 0, N = DelegatingCtorDecls.size(); I != N; ++I) {
    CXXConstructorDecl *D
      = dyn_cast_or_null<CXXConstructorDecl>(GetDecl(DelegatingCtorDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  DelegatingCtorDecls.clear();
}

void ASTReader::ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl *> &Decls) {
  for (unsigned I = 0, N = ExtVectorDecls.size(); I != N; ++I) {
    TypedefNameDecl *D
      = dyn_cast_or_null<TypedefNameDecl>(GetDecl(ExtVectorDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  ExtVectorDecls.clear();
}

void ASTReader::ReadDynamicClasses(SmallVectorImpl<CXXRecordDecl *> &Decls) {
  for (unsigned I = 0, N = DynamicClasses.size(); I != N; ++I) {
    CXXRecordDecl *D
      = dyn_cast_or_null<CXXRecordDecl>(GetDecl(DynamicClasses[I]));
    if (D)
      Decls.push_back(D);
  }
  DynamicClasses.clear();
}

void 
ASTReader::ReadLocallyScopedExternalDecls(SmallVectorImpl<NamedDecl *> &Decls) {
  for (unsigned I = 0, N = LocallyScopedExternalDecls.size(); I != N; ++I) {
    NamedDecl *D 
      = dyn_cast_or_null<NamedDecl>(GetDecl(LocallyScopedExternalDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  LocallyScopedExternalDecls.clear();
}

void ASTReader::ReadReferencedSelectors(
       SmallVectorImpl<std::pair<Selector, SourceLocation> > &Sels) {
  if (ReferencedSelectorsData.empty())
    return;
  
  // If there are @selector references added them to its pool. This is for
  // implementation of -Wselector.
  unsigned int DataSize = ReferencedSelectorsData.size()-1;
  unsigned I = 0;
  while (I < DataSize) {
    Selector Sel = DecodeSelector(ReferencedSelectorsData[I++]);
    SourceLocation SelLoc
      = SourceLocation::getFromRawEncoding(ReferencedSelectorsData[I++]);
    Sels.push_back(std::make_pair(Sel, SelLoc));
  }
  ReferencedSelectorsData.clear();
}

void ASTReader::ReadWeakUndeclaredIdentifiers(
       SmallVectorImpl<std::pair<IdentifierInfo *, WeakInfo> > &WeakIDs) {
  if (WeakUndeclaredIdentifiers.empty())
    return;

  for (unsigned I = 0, N = WeakUndeclaredIdentifiers.size(); I < N; /*none*/) {
    IdentifierInfo *WeakId 
      = DecodeIdentifierInfo(WeakUndeclaredIdentifiers[I++]);
    IdentifierInfo *AliasId 
      = DecodeIdentifierInfo(WeakUndeclaredIdentifiers[I++]);
    SourceLocation Loc
      = SourceLocation::getFromRawEncoding(WeakUndeclaredIdentifiers[I++]);
    bool Used = WeakUndeclaredIdentifiers[I++];
    WeakInfo WI(AliasId, Loc);
    WI.setUsed(Used);
    WeakIDs.push_back(std::make_pair(WeakId, WI));
  }
  WeakUndeclaredIdentifiers.clear();
}

void ASTReader::ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables) {
  for (unsigned Idx = 0, N = VTableUses.size(); Idx < N; /* In loop */) {
    ExternalVTableUse VT;
    VT.Record = dyn_cast_or_null<CXXRecordDecl>(GetDecl(VTableUses[Idx++]));
    VT.Location = SourceLocation::getFromRawEncoding(VTableUses[Idx++]);
    VT.DefinitionRequired = VTableUses[Idx++];
    VTables.push_back(VT);
  }
  
  VTableUses.clear();
}

void ASTReader::ReadPendingInstantiations(
       SmallVectorImpl<std::pair<ValueDecl *, SourceLocation> > &Pending) {
  for (unsigned Idx = 0, N = PendingInstantiations.size(); Idx < N;) {
    ValueDecl *D = cast<ValueDecl>(GetDecl(PendingInstantiations[Idx++]));
    SourceLocation Loc
      = SourceLocation::getFromRawEncoding(PendingInstantiations[Idx++]);
    Pending.push_back(std::make_pair(D, Loc));
  }  
  PendingInstantiations.clear();
}

void ASTReader::LoadSelector(Selector Sel) {
  // It would be complicated to avoid reading the methods anyway. So don't.
  ReadMethodPool(Sel);
}

void ASTReader::SetIdentifierInfo(IdentifierID ID, IdentifierInfo *II) {
  assert(ID && "Non-zero identifier ID required");
  assert(ID <= IdentifiersLoaded.size() && "identifier ID out of range");
  IdentifiersLoaded[ID - 1] = II;
  if (DeserializationListener)
    DeserializationListener->IdentifierRead(ID, II);
}

/// \brief Set the globally-visible declarations associated with the given
/// identifier.
///
/// If the AST reader is currently in a state where the given declaration IDs
/// cannot safely be resolved, they are queued until it is safe to resolve
/// them.
///
/// \param II an IdentifierInfo that refers to one or more globally-visible
/// declarations.
///
/// \param DeclIDs the set of declaration IDs with the name @p II that are
/// visible at global scope.
///
/// \param Nonrecursive should be true to indicate that the caller knows that
/// this call is non-recursive, and therefore the globally-visible declarations
/// will not be placed onto the pending queue.
void
ASTReader::SetGloballyVisibleDecls(IdentifierInfo *II,
                              const SmallVectorImpl<uint32_t> &DeclIDs,
                                   bool Nonrecursive) {
  if (NumCurrentElementsDeserializing && !Nonrecursive) {
    PendingIdentifierInfos.push_back(PendingIdentifierInfo());
    PendingIdentifierInfo &PII = PendingIdentifierInfos.back();
    PII.II = II;
    PII.DeclIDs.append(DeclIDs.begin(), DeclIDs.end());
    return;
  }

  for (unsigned I = 0, N = DeclIDs.size(); I != N; ++I) {
    NamedDecl *D = cast<NamedDecl>(GetDecl(DeclIDs[I]));
    if (SemaObj) {
      if (SemaObj->TUScope) {
        // Introduce this declaration into the translation-unit scope
        // and add it to the declaration chain for this identifier, so
        // that (unqualified) name lookup will find it.
        SemaObj->TUScope->AddDecl(D);
      }
      SemaObj->IdResolver.AddDeclToIdentifierChain(II, D);
    } else {
      // Queue this declaration so that it will be added to the
      // translation unit scope and identifier's declaration chain
      // once a Sema object is known.
      PreloadedDecls.push_back(D);
    }
  }
}

IdentifierInfo *ASTReader::DecodeIdentifierInfo(IdentifierID ID) {
  if (ID == 0)
    return 0;

  if (IdentifiersLoaded.empty()) {
    Error("no identifier table in AST file");
    return 0;
  }

  assert(PP && "Forgot to set Preprocessor ?");
  ID -= 1;
  if (!IdentifiersLoaded[ID]) {
    GlobalIdentifierMapType::iterator I = GlobalIdentifierMap.find(ID + 1);
    assert(I != GlobalIdentifierMap.end() && "Corrupted global identifier map");
    Module *M = I->second;
    unsigned Index = ID - M->BaseIdentifierID;
    const char *Str = M->IdentifierTableData + M->IdentifierOffsets[Index];

    // All of the strings in the AST file are preceded by a 16-bit length.
    // Extract that 16-bit length to avoid having to execute strlen().
    // NOTE: 'StrLenPtr' is an 'unsigned char*' so that we load bytes as
    //  unsigned integers.  This is important to avoid integer overflow when
    //  we cast them to 'unsigned'.
    const unsigned char *StrLenPtr = (const unsigned char*) Str - 2;
    unsigned StrLen = (((unsigned) StrLenPtr[0])
                       | (((unsigned) StrLenPtr[1]) << 8)) - 1;
    IdentifiersLoaded[ID]
      = &PP->getIdentifierTable().get(StringRef(Str, StrLen));
    if (DeserializationListener)
      DeserializationListener->IdentifierRead(ID + 1, IdentifiersLoaded[ID]);
  }

  return IdentifiersLoaded[ID];
}

IdentifierInfo *ASTReader::getLocalIdentifier(Module &M, unsigned LocalID) {
  return DecodeIdentifierInfo(getGlobalIdentifierID(M, LocalID));
}

IdentifierID ASTReader::getGlobalIdentifierID(Module &M, unsigned LocalID) {
  if (LocalID < NUM_PREDEF_IDENT_IDS)
    return LocalID;
  
  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = M.IdentifierRemap.find(LocalID - NUM_PREDEF_IDENT_IDS);
  assert(I != M.IdentifierRemap.end() 
         && "Invalid index into identifier index remap");
  
  return LocalID + I->second;
}

bool ASTReader::ReadSLocEntry(int ID) {
  return ReadSLocEntryRecord(ID) != Success;
}

Selector ASTReader::getLocalSelector(Module &M, unsigned LocalID) {
  return DecodeSelector(getGlobalSelectorID(M, LocalID));
}

Selector ASTReader::DecodeSelector(serialization::SelectorID ID) {
  if (ID == 0)
    return Selector();

  if (ID > SelectorsLoaded.size()) {
    Error("selector ID out of range in AST file");
    return Selector();
  }

  if (SelectorsLoaded[ID - 1].getAsOpaquePtr() == 0) {
    // Load this selector from the selector table.
    GlobalSelectorMapType::iterator I = GlobalSelectorMap.find(ID);
    assert(I != GlobalSelectorMap.end() && "Corrupted global selector map");
    Module &M = *I->second;
    ASTSelectorLookupTrait Trait(*this, M);
    unsigned Idx = ID - M.BaseSelectorID - NUM_PREDEF_SELECTOR_IDS;
    SelectorsLoaded[ID - 1] =
      Trait.ReadKey(M.SelectorLookupTableData + M.SelectorOffsets[Idx], 0);
    if (DeserializationListener)
      DeserializationListener->SelectorRead(ID, SelectorsLoaded[ID - 1]);
  }

  return SelectorsLoaded[ID - 1];
}

Selector ASTReader::GetExternalSelector(serialization::SelectorID ID) {
  return DecodeSelector(ID);
}

uint32_t ASTReader::GetNumExternalSelectors() {
  // ID 0 (the null selector) is considered an external selector.
  return getTotalNumSelectors() + 1;
}

serialization::SelectorID
ASTReader::getGlobalSelectorID(Module &M, unsigned LocalID) const {
  if (LocalID < NUM_PREDEF_SELECTOR_IDS)
    return LocalID;
  
  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = M.SelectorRemap.find(LocalID - NUM_PREDEF_SELECTOR_IDS);
  assert(I != M.SelectorRemap.end() 
         && "Invalid index into identifier index remap");
  
  return LocalID + I->second;
}

DeclarationName
ASTReader::ReadDeclarationName(Module &F, 
                               const RecordData &Record, unsigned &Idx) {
  DeclarationName::NameKind Kind = (DeclarationName::NameKind)Record[Idx++];
  switch (Kind) {
  case DeclarationName::Identifier:
    return DeclarationName(GetIdentifierInfo(F, Record, Idx));

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    return DeclarationName(ReadSelector(F, Record, Idx));

  case DeclarationName::CXXConstructorName:
    return Context->DeclarationNames.getCXXConstructorName(
                          Context->getCanonicalType(readType(F, Record, Idx)));

  case DeclarationName::CXXDestructorName:
    return Context->DeclarationNames.getCXXDestructorName(
                          Context->getCanonicalType(readType(F, Record, Idx)));

  case DeclarationName::CXXConversionFunctionName:
    return Context->DeclarationNames.getCXXConversionFunctionName(
                          Context->getCanonicalType(readType(F, Record, Idx)));

  case DeclarationName::CXXOperatorName:
    return Context->DeclarationNames.getCXXOperatorName(
                                       (OverloadedOperatorKind)Record[Idx++]);

  case DeclarationName::CXXLiteralOperatorName:
    return Context->DeclarationNames.getCXXLiteralOperatorName(
                                       GetIdentifierInfo(F, Record, Idx));

  case DeclarationName::CXXUsingDirective:
    return DeclarationName::getUsingDirectiveName();
  }

  // Required to silence GCC warning
  return DeclarationName();
}

void ASTReader::ReadDeclarationNameLoc(Module &F,
                                       DeclarationNameLoc &DNLoc,
                                       DeclarationName Name,
                                      const RecordData &Record, unsigned &Idx) {
  switch (Name.getNameKind()) {
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    DNLoc.NamedType.TInfo = GetTypeSourceInfo(F, Record, Idx);
    break;

  case DeclarationName::CXXOperatorName:
    DNLoc.CXXOperatorName.BeginOpNameLoc
        = ReadSourceLocation(F, Record, Idx).getRawEncoding();
    DNLoc.CXXOperatorName.EndOpNameLoc
        = ReadSourceLocation(F, Record, Idx).getRawEncoding();
    break;

  case DeclarationName::CXXLiteralOperatorName:
    DNLoc.CXXLiteralOperatorName.OpNameLoc
        = ReadSourceLocation(F, Record, Idx).getRawEncoding();
    break;

  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXUsingDirective:
    break;
  }
}

void ASTReader::ReadDeclarationNameInfo(Module &F,
                                        DeclarationNameInfo &NameInfo,
                                      const RecordData &Record, unsigned &Idx) {
  NameInfo.setName(ReadDeclarationName(F, Record, Idx));
  NameInfo.setLoc(ReadSourceLocation(F, Record, Idx));
  DeclarationNameLoc DNLoc;
  ReadDeclarationNameLoc(F, DNLoc, NameInfo.getName(), Record, Idx);
  NameInfo.setInfo(DNLoc);
}

void ASTReader::ReadQualifierInfo(Module &F, QualifierInfo &Info,
                                  const RecordData &Record, unsigned &Idx) {
  Info.QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, Idx);
  unsigned NumTPLists = Record[Idx++];
  Info.NumTemplParamLists = NumTPLists;
  if (NumTPLists) {
    Info.TemplParamLists = new (*Context) TemplateParameterList*[NumTPLists];
    for (unsigned i=0; i != NumTPLists; ++i)
      Info.TemplParamLists[i] = ReadTemplateParameterList(F, Record, Idx);
  }
}

TemplateName
ASTReader::ReadTemplateName(Module &F, const RecordData &Record, 
                            unsigned &Idx) {
  TemplateName::NameKind Kind = (TemplateName::NameKind)Record[Idx++];
  switch (Kind) {
  case TemplateName::Template:
      return TemplateName(ReadDeclAs<TemplateDecl>(F, Record, Idx));

  case TemplateName::OverloadedTemplate: {
    unsigned size = Record[Idx++];
    UnresolvedSet<8> Decls;
    while (size--)
      Decls.addDecl(ReadDeclAs<NamedDecl>(F, Record, Idx));

    return Context->getOverloadedTemplateName(Decls.begin(), Decls.end());
  }

  case TemplateName::QualifiedTemplate: {
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(F, Record, Idx);
    bool hasTemplKeyword = Record[Idx++];
    TemplateDecl *Template = ReadDeclAs<TemplateDecl>(F, Record, Idx);
    return Context->getQualifiedTemplateName(NNS, hasTemplKeyword, Template);
  }

  case TemplateName::DependentTemplate: {
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(F, Record, Idx);
    if (Record[Idx++])  // isIdentifier
      return Context->getDependentTemplateName(NNS,
                                               GetIdentifierInfo(F, Record, 
                                                                 Idx));
    return Context->getDependentTemplateName(NNS,
                                         (OverloadedOperatorKind)Record[Idx++]);
  }

  case TemplateName::SubstTemplateTemplateParm: {
    TemplateTemplateParmDecl *param
      = ReadDeclAs<TemplateTemplateParmDecl>(F, Record, Idx);
    if (!param) return TemplateName();
    TemplateName replacement = ReadTemplateName(F, Record, Idx);
    return Context->getSubstTemplateTemplateParm(param, replacement);
  }
      
  case TemplateName::SubstTemplateTemplateParmPack: {
    TemplateTemplateParmDecl *Param 
      = ReadDeclAs<TemplateTemplateParmDecl>(F, Record, Idx);
    if (!Param)
      return TemplateName();
    
    TemplateArgument ArgPack = ReadTemplateArgument(F, Record, Idx);
    if (ArgPack.getKind() != TemplateArgument::Pack)
      return TemplateName();
    
    return Context->getSubstTemplateTemplateParmPack(Param, ArgPack);
  }
  }

  assert(0 && "Unhandled template name kind!");
  return TemplateName();
}

TemplateArgument
ASTReader::ReadTemplateArgument(Module &F,
                                const RecordData &Record, unsigned &Idx) {
  TemplateArgument::ArgKind Kind = (TemplateArgument::ArgKind)Record[Idx++];
  switch (Kind) {
  case TemplateArgument::Null:
    return TemplateArgument();
  case TemplateArgument::Type:
    return TemplateArgument(readType(F, Record, Idx));
  case TemplateArgument::Declaration:
    return TemplateArgument(ReadDecl(F, Record, Idx));
  case TemplateArgument::Integral: {
    llvm::APSInt Value = ReadAPSInt(Record, Idx);
    QualType T = readType(F, Record, Idx);
    return TemplateArgument(Value, T);
  }
  case TemplateArgument::Template: 
    return TemplateArgument(ReadTemplateName(F, Record, Idx));
  case TemplateArgument::TemplateExpansion: {
    TemplateName Name = ReadTemplateName(F, Record, Idx);
    llvm::Optional<unsigned> NumTemplateExpansions;
    if (unsigned NumExpansions = Record[Idx++])
      NumTemplateExpansions = NumExpansions - 1;
    return TemplateArgument(Name, NumTemplateExpansions);
  }
  case TemplateArgument::Expression:
    return TemplateArgument(ReadExpr(F));
  case TemplateArgument::Pack: {
    unsigned NumArgs = Record[Idx++];
    TemplateArgument *Args = new (*Context) TemplateArgument[NumArgs];
    for (unsigned I = 0; I != NumArgs; ++I)
      Args[I] = ReadTemplateArgument(F, Record, Idx);
    return TemplateArgument(Args, NumArgs);
  }
  }

  assert(0 && "Unhandled template argument kind!");
  return TemplateArgument();
}

TemplateParameterList *
ASTReader::ReadTemplateParameterList(Module &F,
                                     const RecordData &Record, unsigned &Idx) {
  SourceLocation TemplateLoc = ReadSourceLocation(F, Record, Idx);
  SourceLocation LAngleLoc = ReadSourceLocation(F, Record, Idx);
  SourceLocation RAngleLoc = ReadSourceLocation(F, Record, Idx);

  unsigned NumParams = Record[Idx++];
  SmallVector<NamedDecl *, 16> Params;
  Params.reserve(NumParams);
  while (NumParams--)
    Params.push_back(ReadDeclAs<NamedDecl>(F, Record, Idx));

  TemplateParameterList* TemplateParams =
    TemplateParameterList::Create(*Context, TemplateLoc, LAngleLoc,
                                  Params.data(), Params.size(), RAngleLoc);
  return TemplateParams;
}

void
ASTReader::
ReadTemplateArgumentList(SmallVector<TemplateArgument, 8> &TemplArgs,
                         Module &F, const RecordData &Record,
                         unsigned &Idx) {
  unsigned NumTemplateArgs = Record[Idx++];
  TemplArgs.reserve(NumTemplateArgs);
  while (NumTemplateArgs--)
    TemplArgs.push_back(ReadTemplateArgument(F, Record, Idx));
}

/// \brief Read a UnresolvedSet structure.
void ASTReader::ReadUnresolvedSet(Module &F, UnresolvedSetImpl &Set,
                                  const RecordData &Record, unsigned &Idx) {
  unsigned NumDecls = Record[Idx++];
  while (NumDecls--) {
    NamedDecl *D = ReadDeclAs<NamedDecl>(F, Record, Idx);
    AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
    Set.addDecl(D, AS);
  }
}

CXXBaseSpecifier
ASTReader::ReadCXXBaseSpecifier(Module &F,
                                const RecordData &Record, unsigned &Idx) {
  bool isVirtual = static_cast<bool>(Record[Idx++]);
  bool isBaseOfClass = static_cast<bool>(Record[Idx++]);
  AccessSpecifier AS = static_cast<AccessSpecifier>(Record[Idx++]);
  bool inheritConstructors = static_cast<bool>(Record[Idx++]);
  TypeSourceInfo *TInfo = GetTypeSourceInfo(F, Record, Idx);
  SourceRange Range = ReadSourceRange(F, Record, Idx);
  SourceLocation EllipsisLoc = ReadSourceLocation(F, Record, Idx);
  CXXBaseSpecifier Result(Range, isVirtual, isBaseOfClass, AS, TInfo, 
                          EllipsisLoc);
  Result.setInheritConstructors(inheritConstructors);
  return Result;
}

std::pair<CXXCtorInitializer **, unsigned>
ASTReader::ReadCXXCtorInitializers(Module &F, const RecordData &Record,
                                   unsigned &Idx) {
  CXXCtorInitializer **CtorInitializers = 0;
  unsigned NumInitializers = Record[Idx++];
  if (NumInitializers) {
    ASTContext &C = *getContext();

    CtorInitializers
        = new (C) CXXCtorInitializer*[NumInitializers];
    for (unsigned i=0; i != NumInitializers; ++i) {
      TypeSourceInfo *BaseClassInfo = 0;
      bool IsBaseVirtual = false;
      FieldDecl *Member = 0;
      IndirectFieldDecl *IndirectMember = 0;
      CXXConstructorDecl *Target = 0;

      CtorInitializerType Type = (CtorInitializerType)Record[Idx++];
      switch (Type) {
       case CTOR_INITIALIZER_BASE:
        BaseClassInfo = GetTypeSourceInfo(F, Record, Idx);
        IsBaseVirtual = Record[Idx++];
        break;

       case CTOR_INITIALIZER_DELEGATING:
        Target = ReadDeclAs<CXXConstructorDecl>(F, Record, Idx);
        break;

       case CTOR_INITIALIZER_MEMBER:
        Member = ReadDeclAs<FieldDecl>(F, Record, Idx);
        break;

       case CTOR_INITIALIZER_INDIRECT_MEMBER:
        IndirectMember = ReadDeclAs<IndirectFieldDecl>(F, Record, Idx);
        break;
      }

      SourceLocation MemberOrEllipsisLoc = ReadSourceLocation(F, Record, Idx);
      Expr *Init = ReadExpr(F);
      SourceLocation LParenLoc = ReadSourceLocation(F, Record, Idx);
      SourceLocation RParenLoc = ReadSourceLocation(F, Record, Idx);
      bool IsWritten = Record[Idx++];
      unsigned SourceOrderOrNumArrayIndices;
      SmallVector<VarDecl *, 8> Indices;
      if (IsWritten) {
        SourceOrderOrNumArrayIndices = Record[Idx++];
      } else {
        SourceOrderOrNumArrayIndices = Record[Idx++];
        Indices.reserve(SourceOrderOrNumArrayIndices);
        for (unsigned i=0; i != SourceOrderOrNumArrayIndices; ++i)
          Indices.push_back(ReadDeclAs<VarDecl>(F, Record, Idx));
      }

      CXXCtorInitializer *BOMInit;
      if (Type == CTOR_INITIALIZER_BASE) {
        BOMInit = new (C) CXXCtorInitializer(C, BaseClassInfo, IsBaseVirtual,
                                             LParenLoc, Init, RParenLoc,
                                             MemberOrEllipsisLoc);
      } else if (Type == CTOR_INITIALIZER_DELEGATING) {
        BOMInit = new (C) CXXCtorInitializer(C, MemberOrEllipsisLoc, LParenLoc,
                                             Target, Init, RParenLoc);
      } else if (IsWritten) {
        if (Member)
          BOMInit = new (C) CXXCtorInitializer(C, Member, MemberOrEllipsisLoc,
                                               LParenLoc, Init, RParenLoc);
        else 
          BOMInit = new (C) CXXCtorInitializer(C, IndirectMember,
                                               MemberOrEllipsisLoc, LParenLoc,
                                               Init, RParenLoc);
      } else {
        BOMInit = CXXCtorInitializer::Create(C, Member, MemberOrEllipsisLoc,
                                             LParenLoc, Init, RParenLoc,
                                             Indices.data(), Indices.size());
      }

      if (IsWritten)
        BOMInit->setSourceOrder(SourceOrderOrNumArrayIndices);
      CtorInitializers[i] = BOMInit;
    }
  }

  return std::make_pair(CtorInitializers, NumInitializers);
}

NestedNameSpecifier *
ASTReader::ReadNestedNameSpecifier(Module &F,
                                   const RecordData &Record, unsigned &Idx) {
  unsigned N = Record[Idx++];
  NestedNameSpecifier *NNS = 0, *Prev = 0;
  for (unsigned I = 0; I != N; ++I) {
    NestedNameSpecifier::SpecifierKind Kind
      = (NestedNameSpecifier::SpecifierKind)Record[Idx++];
    switch (Kind) {
    case NestedNameSpecifier::Identifier: {
      IdentifierInfo *II = GetIdentifierInfo(F, Record, Idx);
      NNS = NestedNameSpecifier::Create(*Context, Prev, II);
      break;
    }

    case NestedNameSpecifier::Namespace: {
      NamespaceDecl *NS = ReadDeclAs<NamespaceDecl>(F, Record, Idx);
      NNS = NestedNameSpecifier::Create(*Context, Prev, NS);
      break;
    }

    case NestedNameSpecifier::NamespaceAlias: {
      NamespaceAliasDecl *Alias =ReadDeclAs<NamespaceAliasDecl>(F, Record, Idx);
      NNS = NestedNameSpecifier::Create(*Context, Prev, Alias);
      break;
    }

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate: {
      const Type *T = readType(F, Record, Idx).getTypePtrOrNull();
      if (!T)
        return 0;
      
      bool Template = Record[Idx++];
      NNS = NestedNameSpecifier::Create(*Context, Prev, Template, T);
      break;
    }

    case NestedNameSpecifier::Global: {
      NNS = NestedNameSpecifier::GlobalSpecifier(*Context);
      // No associated value, and there can't be a prefix.
      break;
    }
    }
    Prev = NNS;
  }
  return NNS;
}

NestedNameSpecifierLoc
ASTReader::ReadNestedNameSpecifierLoc(Module &F, const RecordData &Record, 
                                      unsigned &Idx) {
  unsigned N = Record[Idx++];
  NestedNameSpecifierLocBuilder Builder;
  for (unsigned I = 0; I != N; ++I) {
    NestedNameSpecifier::SpecifierKind Kind
      = (NestedNameSpecifier::SpecifierKind)Record[Idx++];
    switch (Kind) {
    case NestedNameSpecifier::Identifier: {
      IdentifierInfo *II = GetIdentifierInfo(F, Record, Idx);      
      SourceRange Range = ReadSourceRange(F, Record, Idx);
      Builder.Extend(*Context, II, Range.getBegin(), Range.getEnd());
      break;
    }

    case NestedNameSpecifier::Namespace: {
      NamespaceDecl *NS = ReadDeclAs<NamespaceDecl>(F, Record, Idx);
      SourceRange Range = ReadSourceRange(F, Record, Idx);
      Builder.Extend(*Context, NS, Range.getBegin(), Range.getEnd());
      break;
    }

    case NestedNameSpecifier::NamespaceAlias: {
      NamespaceAliasDecl *Alias =ReadDeclAs<NamespaceAliasDecl>(F, Record, Idx);
      SourceRange Range = ReadSourceRange(F, Record, Idx);
      Builder.Extend(*Context, Alias, Range.getBegin(), Range.getEnd());
      break;
    }

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate: {
      bool Template = Record[Idx++];
      TypeSourceInfo *T = GetTypeSourceInfo(F, Record, Idx);
      if (!T)
        return NestedNameSpecifierLoc();
      SourceLocation ColonColonLoc = ReadSourceLocation(F, Record, Idx);

      // FIXME: 'template' keyword location not saved anywhere, so we fake it.
      Builder.Extend(*Context, 
                     Template? T->getTypeLoc().getBeginLoc() : SourceLocation(),
                     T->getTypeLoc(), ColonColonLoc);
      break;
    }

    case NestedNameSpecifier::Global: {
      SourceLocation ColonColonLoc = ReadSourceLocation(F, Record, Idx);
      Builder.MakeGlobal(*Context, ColonColonLoc);
      break;
    }
    }
  }
  
  return Builder.getWithLocInContext(*Context);
}

SourceRange
ASTReader::ReadSourceRange(Module &F, const RecordData &Record,
                           unsigned &Idx) {
  SourceLocation beg = ReadSourceLocation(F, Record, Idx);
  SourceLocation end = ReadSourceLocation(F, Record, Idx);
  return SourceRange(beg, end);
}

/// \brief Read an integral value
llvm::APInt ASTReader::ReadAPInt(const RecordData &Record, unsigned &Idx) {
  unsigned BitWidth = Record[Idx++];
  unsigned NumWords = llvm::APInt::getNumWords(BitWidth);
  llvm::APInt Result(BitWidth, NumWords, &Record[Idx]);
  Idx += NumWords;
  return Result;
}

/// \brief Read a signed integral value
llvm::APSInt ASTReader::ReadAPSInt(const RecordData &Record, unsigned &Idx) {
  bool isUnsigned = Record[Idx++];
  return llvm::APSInt(ReadAPInt(Record, Idx), isUnsigned);
}

/// \brief Read a floating-point value
llvm::APFloat ASTReader::ReadAPFloat(const RecordData &Record, unsigned &Idx) {
  return llvm::APFloat(ReadAPInt(Record, Idx));
}

// \brief Read a string
std::string ASTReader::ReadString(const RecordData &Record, unsigned &Idx) {
  unsigned Len = Record[Idx++];
  std::string Result(Record.data() + Idx, Record.data() + Idx + Len);
  Idx += Len;
  return Result;
}

VersionTuple ASTReader::ReadVersionTuple(const RecordData &Record, 
                                         unsigned &Idx) {
  unsigned Major = Record[Idx++];
  unsigned Minor = Record[Idx++];
  unsigned Subminor = Record[Idx++];
  if (Minor == 0)
    return VersionTuple(Major);
  if (Subminor == 0)
    return VersionTuple(Major, Minor - 1);
  return VersionTuple(Major, Minor - 1, Subminor - 1);
}

CXXTemporary *ASTReader::ReadCXXTemporary(Module &F, 
                                          const RecordData &Record,
                                          unsigned &Idx) {
  CXXDestructorDecl *Decl = ReadDeclAs<CXXDestructorDecl>(F, Record, Idx);
  return CXXTemporary::Create(*Context, Decl);
}

DiagnosticBuilder ASTReader::Diag(unsigned DiagID) {
  return Diag(SourceLocation(), DiagID);
}

DiagnosticBuilder ASTReader::Diag(SourceLocation Loc, unsigned DiagID) {
  return Diags.Report(Loc, DiagID);
}

/// \brief Retrieve the identifier table associated with the
/// preprocessor.
IdentifierTable &ASTReader::getIdentifierTable() {
  assert(PP && "Forgot to set Preprocessor ?");
  return PP->getIdentifierTable();
}

/// \brief Record that the given ID maps to the given switch-case
/// statement.
void ASTReader::RecordSwitchCaseID(SwitchCase *SC, unsigned ID) {
  assert(SwitchCaseStmts[ID] == 0 && "Already have a SwitchCase with this ID");
  SwitchCaseStmts[ID] = SC;
}

/// \brief Retrieve the switch-case statement with the given ID.
SwitchCase *ASTReader::getSwitchCaseWithID(unsigned ID) {
  assert(SwitchCaseStmts[ID] != 0 && "No SwitchCase with this ID");
  return SwitchCaseStmts[ID];
}

void ASTReader::ClearSwitchCaseIDs() {
  SwitchCaseStmts.clear();
}

void ASTReader::FinishedDeserializing() {
  assert(NumCurrentElementsDeserializing &&
         "FinishedDeserializing not paired with StartedDeserializing");
  if (NumCurrentElementsDeserializing == 1) {
    // If any identifiers with corresponding top-level declarations have
    // been loaded, load those declarations now.
    while (!PendingIdentifierInfos.empty()) {
      SetGloballyVisibleDecls(PendingIdentifierInfos.front().II,
                              PendingIdentifierInfos.front().DeclIDs, true);
      PendingIdentifierInfos.pop_front();
    }

    // Ready to load previous declarations of Decls that were delayed.
    while (!PendingPreviousDecls.empty()) {
      loadAndAttachPreviousDecl(PendingPreviousDecls.front().first,
                                PendingPreviousDecls.front().second);
      PendingPreviousDecls.pop_front();
    }

    // We are not in recursive loading, so it's safe to pass the "interesting"
    // decls to the consumer.
    if (Consumer)
      PassInterestingDeclsToConsumer();

    assert(PendingForwardRefs.size() == 0 &&
           "Some forward refs did not get linked to the definition!");
  }
  --NumCurrentElementsDeserializing;
}

ASTReader::ASTReader(Preprocessor &PP, ASTContext *Context,
                     StringRef isysroot, bool DisableValidation,
                     bool DisableStatCache)
  : Listener(new PCHValidator(PP, *this)), DeserializationListener(0),
    SourceMgr(PP.getSourceManager()), FileMgr(PP.getFileManager()),
    Diags(PP.getDiagnostics()), SemaObj(0), PP(&PP), Context(Context),
    Consumer(0), ModuleMgr(FileMgr.getFileSystemOptions()),
    RelocatablePCH(false), isysroot(isysroot),
    DisableValidation(DisableValidation),
    DisableStatCache(DisableStatCache), NumStatHits(0), NumStatMisses(0), 
    NumSLocEntriesRead(0), TotalNumSLocEntries(0), 
    NumStatementsRead(0), TotalNumStatements(0), NumMacrosRead(0), 
    TotalNumMacros(0), NumSelectorsRead(0), NumMethodPoolEntriesRead(0), 
    NumMethodPoolMisses(0), TotalNumMethodPoolEntries(0), 
    NumLexicalDeclContextsRead(0), TotalLexicalDeclContexts(0), 
    NumVisibleDeclContextsRead(0), TotalVisibleDeclContexts(0),
    TotalModulesSizeInBits(0), NumCurrentElementsDeserializing(0),
    NumCXXBaseSpecifiersLoaded(0)
{
  SourceMgr.setExternalSLocEntrySource(this);
}

ASTReader::ASTReader(SourceManager &SourceMgr, FileManager &FileMgr,
                     Diagnostic &Diags, StringRef isysroot,
                     bool DisableValidation, bool DisableStatCache)
  : DeserializationListener(0), SourceMgr(SourceMgr), FileMgr(FileMgr),
    Diags(Diags), SemaObj(0), PP(0), Context(0),
    Consumer(0), ModuleMgr(FileMgr.getFileSystemOptions()),
    RelocatablePCH(false), isysroot(isysroot), 
    DisableValidation(DisableValidation), DisableStatCache(DisableStatCache), 
    NumStatHits(0), NumStatMisses(0), NumSLocEntriesRead(0), 
    TotalNumSLocEntries(0), NumStatementsRead(0), 
    TotalNumStatements(0), NumMacrosRead(0), TotalNumMacros(0), 
    NumSelectorsRead(0), NumMethodPoolEntriesRead(0), NumMethodPoolMisses(0),
    TotalNumMethodPoolEntries(0), NumLexicalDeclContextsRead(0),
    TotalLexicalDeclContexts(0), NumVisibleDeclContextsRead(0),
    TotalVisibleDeclContexts(0), TotalModulesSizeInBits(0),
    NumCurrentElementsDeserializing(0), NumCXXBaseSpecifiersLoaded(0)
{
  SourceMgr.setExternalSLocEntrySource(this);
}

ASTReader::~ASTReader() {
  // Delete all visible decl lookup tables
  for (DeclContextOffsetsMap::iterator I = DeclContextOffsets.begin(),
                                       E = DeclContextOffsets.end();
       I != E; ++I) {
    for (DeclContextInfos::iterator J = I->second.begin(), F = I->second.end();
         J != F; ++J) {
      if (J->NameLookupTableData)
        delete static_cast<ASTDeclContextNameLookupTable*>(
            J->NameLookupTableData);
    }
  }
  for (DeclContextVisibleUpdatesPending::iterator
           I = PendingVisibleUpdates.begin(),
           E = PendingVisibleUpdates.end();
       I != E; ++I) {
    for (DeclContextVisibleUpdates::iterator J = I->second.begin(),
                                             F = I->second.end();
         J != F; ++J)
      delete static_cast<ASTDeclContextNameLookupTable*>(J->first);
  }
}

Module::Module(ModuleKind Kind)
  : Kind(Kind), SizeInBits(0), LocalNumSLocEntries(0), SLocEntryBaseID(0),
    SLocEntryBaseOffset(0), SLocEntryOffsets(0),
    SLocFileOffsets(0), LocalNumIdentifiers(0), 
    IdentifierOffsets(0), BaseIdentifierID(0), IdentifierTableData(0),
    IdentifierLookupTable(0), BasePreprocessedEntityID(0),
    LocalNumMacroDefinitions(0), MacroDefinitionOffsets(0), 
    BaseMacroDefinitionID(0), LocalNumHeaderFileInfos(0), 
    HeaderFileInfoTableData(0), HeaderFileInfoTable(0),
    HeaderFileFrameworkStrings(0),
    LocalNumSelectors(0), SelectorOffsets(0), BaseSelectorID(0),
    SelectorLookupTableData(0), SelectorLookupTable(0), LocalNumDecls(0),
    DeclOffsets(0), BaseDeclID(0),
    LocalNumCXXBaseSpecifiers(0), CXXBaseSpecifiersOffsets(0),
    LocalNumTypes(0), TypeOffsets(0), BaseTypeIndex(0), StatCache(0),
    NumPreallocatedPreprocessingEntities(0)
{}

Module::~Module() {
  delete static_cast<ASTIdentifierLookupTable *>(IdentifierLookupTable);
  delete static_cast<HeaderFileInfoLookupTable *>(HeaderFileInfoTable);
  delete static_cast<ASTSelectorLookupTable *>(SelectorLookupTable);
}

template<typename Key, typename Offset, unsigned InitialCapacity>
static void 
dumpLocalRemap(StringRef Name,
               const ContinuousRangeMap<Key, Offset, InitialCapacity> &Map) {
  if (Map.begin() == Map.end())
    return;
  
  typedef ContinuousRangeMap<Key, Offset, InitialCapacity> MapType;
  llvm::errs() << "  " << Name << ":\n";
  for (typename MapType::const_iterator I = Map.begin(), IEnd = Map.end(); 
       I != IEnd; ++I) {
    llvm::errs() << "    " << I->first << " -> " << I->second
    << "\n";
  }
}

void Module::dump() {
  llvm::errs() << "\nModule: " << FileName << "\n";
  if (!Imports.empty()) {
    llvm::errs() << "  Imports: ";
    for (unsigned I = 0, N = Imports.size(); I != N; ++I) {
      if (I)
        llvm::errs() << ", ";
      llvm::errs() << Imports[I]->FileName;
    }
    llvm::errs() << "\n";
  }
  
  // Remapping tables.
  llvm::errs() << "  Base source location offset: " << SLocEntryBaseOffset 
               << '\n';
  dumpLocalRemap("Source location offset map", SLocRemap);
  llvm::errs() << "  Base identifier ID: " << BaseIdentifierID << '\n'
               << "  Number of identifiers: " << LocalNumIdentifiers << '\n';
  dumpLocalRemap("Identifier ID map", IdentifierRemap);
  llvm::errs() << "  Base preprocessed entity ID: " << BasePreprocessedEntityID
               << '\n'  
               << "Number of preprocessed entities: " 
               << NumPreallocatedPreprocessingEntities << '\n';
  dumpLocalRemap("Preprocessed entity ID map", PreprocessedEntityRemap);
  llvm::errs() << "  Base type index: " << BaseTypeIndex << '\n'
               << "  Number of types: " << LocalNumTypes << '\n';
  dumpLocalRemap("Type index map", TypeRemap);
  llvm::errs() << "  Base macro definition ID: " << BaseMacroDefinitionID 
               << '\n'
               << "  Number of macro definitions: " << LocalNumMacroDefinitions
               << '\n';
  dumpLocalRemap("Macro definition ID map", MacroDefinitionRemap);
  llvm::errs() << "  Base decl ID: " << BaseDeclID << '\n'
               << "  Number of decls: " << LocalNumDecls << '\n';
  dumpLocalRemap("Decl ID map", DeclRemap);
}

Module *ModuleManager::lookup(StringRef Name) {
  const FileEntry *Entry = FileMgr.getFile(Name);
  return Modules[Entry];
}

llvm::MemoryBuffer *ModuleManager::lookupBuffer(StringRef Name) {
  const FileEntry *Entry = FileMgr.getFile(Name);
  return InMemoryBuffers[Entry];
}

/// \brief Creates a new module and adds it to the list of known modules
Module &ModuleManager::addModule(StringRef FileName, ModuleKind Type) {
  Module *Prev = !size() ? 0 : &getLastModule();
  Module *Current = new Module(Type);

  Current->FileName = FileName.str();

  Chain.push_back(Current);
  const FileEntry *Entry = FileMgr.getFile(FileName);
  Modules[Entry] = Current;

  if (Prev) {
    Current->ImportedBy.insert(Prev);
    Prev->Imports.insert(Current);
  }
  
  return *Current;
}

void ModuleManager::addInMemoryBuffer(StringRef FileName, 
  llvm::MemoryBuffer *Buffer) {
  
  const FileEntry *Entry = FileMgr.getVirtualFile(FileName, 
    Buffer->getBufferSize(), 0);
  InMemoryBuffers[Entry] = Buffer;
}
/// \brief Exports the list of loaded modules with their corresponding names
void ModuleManager::exportLookup(SmallVector<ModuleOffset, 16> &Target) {
  Target.reserve(size());
  for (ModuleConstIterator I = Chain.begin(), E = Chain.end();
       I != E; ++I) {
    Target.push_back(ModuleOffset((*I)->SLocEntryBaseOffset,
                                  (*I)->FileName));
  }
  std::sort(Target.begin(), Target.end());
}

ModuleManager::ModuleManager(const FileSystemOptions &FSO) : FileMgr(FSO) { }

ModuleManager::~ModuleManager() {
  for (unsigned i = 0, e = Chain.size(); i != e; ++i)
    delete Chain[e - i - 1];
}
