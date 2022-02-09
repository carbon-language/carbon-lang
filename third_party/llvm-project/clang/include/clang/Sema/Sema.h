//===--- Sema.h - Semantic Analysis & AST Building --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sema class, which performs semantic analysis and
// builds ASTs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMA_H
#define LLVM_CLANG_SEMA_SEMA_H

#include "clang/AST/ASTConcept.h"
#include "clang/AST/ASTFwd.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Availability.h"
#include "clang/AST/ComparisonCategories.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/LocInfoType.h"
#include "clang/AST/MangleNumberingContext.h"
#include "clang/AST/NSAPI.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/BitmaskEnum.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DarwinSDKInfo.h"
#include "clang/Basic/ExpressionTraits.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/OpenCLOptions.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/PragmaKinds.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TemplateKinds.h"
#include "clang/Basic/TypeTraits.h"
#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/Sema/CleanupInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Sema/ObjCMethodList.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaConcept.h"
#include "clang/Sema/TypoCorrection.h"
#include "clang/Sema/Weak.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <deque>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace llvm {
  class APSInt;
  template <typename ValueT, typename ValueInfoT> class DenseSet;
  class SmallBitVector;
  struct InlineAsmIdentifierInfo;
}

namespace clang {
  class ADLResult;
  class ASTConsumer;
  class ASTContext;
  class ASTMutationListener;
  class ASTReader;
  class ASTWriter;
  class ArrayType;
  class ParsedAttr;
  class BindingDecl;
  class BlockDecl;
  class CapturedDecl;
  class CXXBasePath;
  class CXXBasePaths;
  class CXXBindTemporaryExpr;
  typedef SmallVector<CXXBaseSpecifier*, 4> CXXCastPath;
  class CXXConstructorDecl;
  class CXXConversionDecl;
  class CXXDeleteExpr;
  class CXXDestructorDecl;
  class CXXFieldCollector;
  class CXXMemberCallExpr;
  class CXXMethodDecl;
  class CXXScopeSpec;
  class CXXTemporary;
  class CXXTryStmt;
  class CallExpr;
  class ClassTemplateDecl;
  class ClassTemplatePartialSpecializationDecl;
  class ClassTemplateSpecializationDecl;
  class VarTemplatePartialSpecializationDecl;
  class CodeCompleteConsumer;
  class CodeCompletionAllocator;
  class CodeCompletionTUInfo;
  class CodeCompletionResult;
  class CoroutineBodyStmt;
  class Decl;
  class DeclAccessPair;
  class DeclContext;
  class DeclRefExpr;
  class DeclaratorDecl;
  class DeducedTemplateArgument;
  class DependentDiagnostic;
  class DesignatedInitExpr;
  class Designation;
  class EnableIfAttr;
  class EnumConstantDecl;
  class Expr;
  class ExtVectorType;
  class FormatAttr;
  class FriendDecl;
  class FunctionDecl;
  class FunctionProtoType;
  class FunctionTemplateDecl;
  class ImplicitConversionSequence;
  typedef MutableArrayRef<ImplicitConversionSequence> ConversionSequenceList;
  class InitListExpr;
  class InitializationKind;
  class InitializationSequence;
  class InitializedEntity;
  class IntegerLiteral;
  class LabelStmt;
  class LambdaExpr;
  class LangOptions;
  class LocalInstantiationScope;
  class LookupResult;
  class MacroInfo;
  typedef ArrayRef<std::pair<IdentifierInfo *, SourceLocation>> ModuleIdPath;
  class ModuleLoader;
  class MultiLevelTemplateArgumentList;
  class NamedDecl;
  class ObjCCategoryDecl;
  class ObjCCategoryImplDecl;
  class ObjCCompatibleAliasDecl;
  class ObjCContainerDecl;
  class ObjCImplDecl;
  class ObjCImplementationDecl;
  class ObjCInterfaceDecl;
  class ObjCIvarDecl;
  template <class T> class ObjCList;
  class ObjCMessageExpr;
  class ObjCMethodDecl;
  class ObjCPropertyDecl;
  class ObjCProtocolDecl;
  class OMPThreadPrivateDecl;
  class OMPRequiresDecl;
  class OMPDeclareReductionDecl;
  class OMPDeclareSimdDecl;
  class OMPClause;
  struct OMPVarListLocTy;
  struct OverloadCandidate;
  enum class OverloadCandidateParamOrder : char;
  enum OverloadCandidateRewriteKind : unsigned;
  class OverloadCandidateSet;
  class OverloadExpr;
  class ParenListExpr;
  class ParmVarDecl;
  class Preprocessor;
  class PseudoDestructorTypeStorage;
  class PseudoObjectExpr;
  class QualType;
  class StandardConversionSequence;
  class Stmt;
  class StringLiteral;
  class SwitchStmt;
  class TemplateArgument;
  class TemplateArgumentList;
  class TemplateArgumentLoc;
  class TemplateDecl;
  class TemplateInstantiationCallback;
  class TemplateParameterList;
  class TemplatePartialOrderingContext;
  class TemplateTemplateParmDecl;
  class Token;
  class TypeAliasDecl;
  class TypedefDecl;
  class TypedefNameDecl;
  class TypeLoc;
  class TypoCorrectionConsumer;
  class UnqualifiedId;
  class UnresolvedLookupExpr;
  class UnresolvedMemberExpr;
  class UnresolvedSetImpl;
  class UnresolvedSetIterator;
  class UsingDecl;
  class UsingShadowDecl;
  class ValueDecl;
  class VarDecl;
  class VarTemplateSpecializationDecl;
  class VisibilityAttr;
  class VisibleDeclConsumer;
  class IndirectFieldDecl;
  struct DeductionFailureInfo;
  class TemplateSpecCandidateSet;

namespace sema {
  class AccessedEntity;
  class BlockScopeInfo;
  class Capture;
  class CapturedRegionScopeInfo;
  class CapturingScopeInfo;
  class CompoundScopeInfo;
  class DelayedDiagnostic;
  class DelayedDiagnosticPool;
  class FunctionScopeInfo;
  class LambdaScopeInfo;
  class PossiblyUnreachableDiag;
  class SemaPPCallbacks;
  class TemplateDeductionInfo;
}

namespace threadSafety {
  class BeforeSet;
  void threadSafetyCleanup(BeforeSet* Cache);
}

// FIXME: No way to easily map from TemplateTypeParmTypes to
// TemplateTypeParmDecls, so we have this horrible PointerUnion.
typedef std::pair<llvm::PointerUnion<const TemplateTypeParmType*, NamedDecl*>,
                  SourceLocation> UnexpandedParameterPack;

/// Describes whether we've seen any nullability information for the given
/// file.
struct FileNullability {
  /// The first pointer declarator (of any pointer kind) in the file that does
  /// not have a corresponding nullability annotation.
  SourceLocation PointerLoc;

  /// The end location for the first pointer declarator in the file. Used for
  /// placing fix-its.
  SourceLocation PointerEndLoc;

  /// Which kind of pointer declarator we saw.
  uint8_t PointerKind;

  /// Whether we saw any type nullability annotations in the given file.
  bool SawTypeNullability = false;
};

/// A mapping from file IDs to a record of whether we've seen nullability
/// information in that file.
class FileNullabilityMap {
  /// A mapping from file IDs to the nullability information for each file ID.
  llvm::DenseMap<FileID, FileNullability> Map;

  /// A single-element cache based on the file ID.
  struct {
    FileID File;
    FileNullability Nullability;
  } Cache;

public:
  FileNullability &operator[](FileID file) {
    // Check the single-element cache.
    if (file == Cache.File)
      return Cache.Nullability;

    // It's not in the single-element cache; flush the cache if we have one.
    if (!Cache.File.isInvalid()) {
      Map[Cache.File] = Cache.Nullability;
    }

    // Pull this entry into the cache.
    Cache.File = file;
    Cache.Nullability = Map[file];
    return Cache.Nullability;
  }
};

/// Tracks expected type during expression parsing, for use in code completion.
/// The type is tied to a particular token, all functions that update or consume
/// the type take a start location of the token they are looking at as a
/// parameter. This avoids updating the type on hot paths in the parser.
class PreferredTypeBuilder {
public:
  PreferredTypeBuilder(bool Enabled) : Enabled(Enabled) {}

  void enterCondition(Sema &S, SourceLocation Tok);
  void enterReturn(Sema &S, SourceLocation Tok);
  void enterVariableInit(SourceLocation Tok, Decl *D);
  /// Handles e.g. BaseType{ .D = Tok...
  void enterDesignatedInitializer(SourceLocation Tok, QualType BaseType,
                                  const Designation &D);
  /// Computing a type for the function argument may require running
  /// overloading, so we postpone its computation until it is actually needed.
  ///
  /// Clients should be very careful when using this funciton, as it stores a
  /// function_ref, clients should make sure all calls to get() with the same
  /// location happen while function_ref is alive.
  ///
  /// The callback should also emit signature help as a side-effect, but only
  /// if the completion point has been reached.
  void enterFunctionArgument(SourceLocation Tok,
                             llvm::function_ref<QualType()> ComputeType);

  void enterParenExpr(SourceLocation Tok, SourceLocation LParLoc);
  void enterUnary(Sema &S, SourceLocation Tok, tok::TokenKind OpKind,
                  SourceLocation OpLoc);
  void enterBinary(Sema &S, SourceLocation Tok, Expr *LHS, tok::TokenKind Op);
  void enterMemAccess(Sema &S, SourceLocation Tok, Expr *Base);
  void enterSubscript(Sema &S, SourceLocation Tok, Expr *LHS);
  /// Handles all type casts, including C-style cast, C++ casts, etc.
  void enterTypeCast(SourceLocation Tok, QualType CastType);

  /// Get the expected type associated with this location, if any.
  ///
  /// If the location is a function argument, determining the expected type
  /// involves considering all function overloads and the arguments so far.
  /// In this case, signature help for these function overloads will be reported
  /// as a side-effect (only if the completion point has been reached).
  QualType get(SourceLocation Tok) const {
    if (!Enabled || Tok != ExpectedLoc)
      return QualType();
    if (!Type.isNull())
      return Type;
    if (ComputeType)
      return ComputeType();
    return QualType();
  }

private:
  bool Enabled;
  /// Start position of a token for which we store expected type.
  SourceLocation ExpectedLoc;
  /// Expected type for a token starting at ExpectedLoc.
  QualType Type;
  /// A function to compute expected type at ExpectedLoc. It is only considered
  /// if Type is null.
  llvm::function_ref<QualType()> ComputeType;
};

/// Sema - This implements semantic analysis and AST building for C.
class Sema final {
  Sema(const Sema &) = delete;
  void operator=(const Sema &) = delete;

  ///Source of additional semantic information.
  ExternalSemaSource *ExternalSource;

  ///Whether Sema has generated a multiplexer and has to delete it.
  bool isMultiplexExternalSource;

  static bool mightHaveNonExternalLinkage(const DeclaratorDecl *FD);

  bool isVisibleSlow(const NamedDecl *D);

  /// Determine whether two declarations should be linked together, given that
  /// the old declaration might not be visible and the new declaration might
  /// not have external linkage.
  bool shouldLinkPossiblyHiddenDecl(const NamedDecl *Old,
                                    const NamedDecl *New) {
    if (isVisible(Old))
     return true;
    // See comment in below overload for why it's safe to compute the linkage
    // of the new declaration here.
    if (New->isExternallyDeclarable()) {
      assert(Old->isExternallyDeclarable() &&
             "should not have found a non-externally-declarable previous decl");
      return true;
    }
    return false;
  }
  bool shouldLinkPossiblyHiddenDecl(LookupResult &Old, const NamedDecl *New);

  void setupImplicitSpecialMemberType(CXXMethodDecl *SpecialMem,
                                      QualType ResultTy,
                                      ArrayRef<QualType> Args);

public:
  /// The maximum alignment, same as in llvm::Value. We duplicate them here
  /// because that allows us not to duplicate the constants in clang code,
  /// which we must to since we can't directly use the llvm constants.
  /// The value is verified against llvm here: lib/CodeGen/CGDecl.cpp
  ///
  /// This is the greatest alignment value supported by load, store, and alloca
  /// instructions, and global values.
  static const unsigned MaxAlignmentExponent = 32;
  static const uint64_t MaximumAlignment = 1ull << MaxAlignmentExponent;

  typedef OpaquePtr<DeclGroupRef> DeclGroupPtrTy;
  typedef OpaquePtr<TemplateName> TemplateTy;
  typedef OpaquePtr<QualType> TypeTy;

  OpenCLOptions OpenCLFeatures;
  FPOptions CurFPFeatures;

  const LangOptions &LangOpts;
  Preprocessor &PP;
  ASTContext &Context;
  ASTConsumer &Consumer;
  DiagnosticsEngine &Diags;
  SourceManager &SourceMgr;

  /// Flag indicating whether or not to collect detailed statistics.
  bool CollectStats;

  /// Code-completion consumer.
  CodeCompleteConsumer *CodeCompleter;

  /// CurContext - This is the current declaration context of parsing.
  DeclContext *CurContext;

  /// Generally null except when we temporarily switch decl contexts,
  /// like in \see ActOnObjCTemporaryExitContainerContext.
  DeclContext *OriginalLexicalContext;

  /// VAListTagName - The declaration name corresponding to __va_list_tag.
  /// This is used as part of a hack to omit that class from ADL results.
  DeclarationName VAListTagName;

  bool MSStructPragmaOn; // True when \#pragma ms_struct on

  /// Controls member pointer representation format under the MS ABI.
  LangOptions::PragmaMSPointersToMembersKind
      MSPointerToMemberRepresentationMethod;

  /// Stack of active SEH __finally scopes.  Can be empty.
  SmallVector<Scope*, 2> CurrentSEHFinally;

  /// Source location for newly created implicit MSInheritanceAttrs
  SourceLocation ImplicitMSInheritanceAttrLoc;

  /// Holds TypoExprs that are created from `createDelayedTypo`. This is used by
  /// `TransformTypos` in order to keep track of any TypoExprs that are created
  /// recursively during typo correction and wipe them away if the correction
  /// fails.
  llvm::SmallVector<TypoExpr *, 2> TypoExprs;

  /// pragma clang section kind
  enum PragmaClangSectionKind {
    PCSK_Invalid      = 0,
    PCSK_BSS          = 1,
    PCSK_Data         = 2,
    PCSK_Rodata       = 3,
    PCSK_Text         = 4,
    PCSK_Relro        = 5
   };

  enum PragmaClangSectionAction {
    PCSA_Set     = 0,
    PCSA_Clear   = 1
  };

  struct PragmaClangSection {
    std::string SectionName;
    bool Valid = false;
    SourceLocation PragmaLocation;
  };

   PragmaClangSection PragmaClangBSSSection;
   PragmaClangSection PragmaClangDataSection;
   PragmaClangSection PragmaClangRodataSection;
   PragmaClangSection PragmaClangRelroSection;
   PragmaClangSection PragmaClangTextSection;

  enum PragmaMsStackAction {
    PSK_Reset     = 0x0,                // #pragma ()
    PSK_Set       = 0x1,                // #pragma (value)
    PSK_Push      = 0x2,                // #pragma (push[, id])
    PSK_Pop       = 0x4,                // #pragma (pop[, id])
    PSK_Show      = 0x8,                // #pragma (show) -- only for "pack"!
    PSK_Push_Set  = PSK_Push | PSK_Set, // #pragma (push[, id], value)
    PSK_Pop_Set   = PSK_Pop | PSK_Set,  // #pragma (pop[, id], value)
  };

  // #pragma pack and align.
  class AlignPackInfo {
  public:
    // `Native` represents default align mode, which may vary based on the
    // platform.
    enum Mode : unsigned char { Native, Natural, Packed, Mac68k };

    // #pragma pack info constructor
    AlignPackInfo(AlignPackInfo::Mode M, unsigned Num, bool IsXL)
        : PackAttr(true), AlignMode(M), PackNumber(Num), XLStack(IsXL) {
      assert(Num == PackNumber && "The pack number has been truncated.");
    }

    // #pragma align info constructor
    AlignPackInfo(AlignPackInfo::Mode M, bool IsXL)
        : PackAttr(false), AlignMode(M),
          PackNumber(M == Packed ? 1 : UninitPackVal), XLStack(IsXL) {}

    explicit AlignPackInfo(bool IsXL) : AlignPackInfo(Native, IsXL) {}

    AlignPackInfo() : AlignPackInfo(Native, false) {}

    // When a AlignPackInfo itself cannot be used, this returns an 32-bit
    // integer encoding for it. This should only be passed to
    // AlignPackInfo::getFromRawEncoding, it should not be inspected directly.
    static uint32_t getRawEncoding(const AlignPackInfo &Info) {
      std::uint32_t Encoding{};
      if (Info.IsXLStack())
        Encoding |= IsXLMask;

      Encoding |= static_cast<uint32_t>(Info.getAlignMode()) << 1;

      if (Info.IsPackAttr())
        Encoding |= PackAttrMask;

      Encoding |= static_cast<uint32_t>(Info.getPackNumber()) << 4;

      return Encoding;
    }

    static AlignPackInfo getFromRawEncoding(unsigned Encoding) {
      bool IsXL = static_cast<bool>(Encoding & IsXLMask);
      AlignPackInfo::Mode M =
          static_cast<AlignPackInfo::Mode>((Encoding & AlignModeMask) >> 1);
      int PackNumber = (Encoding & PackNumMask) >> 4;

      if (Encoding & PackAttrMask)
        return AlignPackInfo(M, PackNumber, IsXL);

      return AlignPackInfo(M, IsXL);
    }

    bool IsPackAttr() const { return PackAttr; }

    bool IsAlignAttr() const { return !PackAttr; }

    Mode getAlignMode() const { return AlignMode; }

    unsigned getPackNumber() const { return PackNumber; }

    bool IsPackSet() const {
      // #pragma align, #pragma pack(), and #pragma pack(0) do not set the pack
      // attriute on a decl.
      return PackNumber != UninitPackVal && PackNumber != 0;
    }

    bool IsXLStack() const { return XLStack; }

    bool operator==(const AlignPackInfo &Info) const {
      return std::tie(AlignMode, PackNumber, PackAttr, XLStack) ==
             std::tie(Info.AlignMode, Info.PackNumber, Info.PackAttr,
                      Info.XLStack);
    }

    bool operator!=(const AlignPackInfo &Info) const {
      return !(*this == Info);
    }

  private:
    /// \brief True if this is a pragma pack attribute,
    ///         not a pragma align attribute.
    bool PackAttr;

    /// \brief The alignment mode that is in effect.
    Mode AlignMode;

    /// \brief The pack number of the stack.
    unsigned char PackNumber;

    /// \brief True if it is a XL #pragma align/pack stack.
    bool XLStack;

    /// \brief Uninitialized pack value.
    static constexpr unsigned char UninitPackVal = -1;

    // Masks to encode and decode an AlignPackInfo.
    static constexpr uint32_t IsXLMask{0x0000'0001};
    static constexpr uint32_t AlignModeMask{0x0000'0006};
    static constexpr uint32_t PackAttrMask{0x00000'0008};
    static constexpr uint32_t PackNumMask{0x0000'01F0};
  };

  template<typename ValueType>
  struct PragmaStack {
    struct Slot {
      llvm::StringRef StackSlotLabel;
      ValueType Value;
      SourceLocation PragmaLocation;
      SourceLocation PragmaPushLocation;
      Slot(llvm::StringRef StackSlotLabel, ValueType Value,
           SourceLocation PragmaLocation, SourceLocation PragmaPushLocation)
          : StackSlotLabel(StackSlotLabel), Value(Value),
            PragmaLocation(PragmaLocation),
            PragmaPushLocation(PragmaPushLocation) {}
    };

    void Act(SourceLocation PragmaLocation, PragmaMsStackAction Action,
             llvm::StringRef StackSlotLabel, ValueType Value) {
      if (Action == PSK_Reset) {
        CurrentValue = DefaultValue;
        CurrentPragmaLocation = PragmaLocation;
        return;
      }
      if (Action & PSK_Push)
        Stack.emplace_back(StackSlotLabel, CurrentValue, CurrentPragmaLocation,
                           PragmaLocation);
      else if (Action & PSK_Pop) {
        if (!StackSlotLabel.empty()) {
          // If we've got a label, try to find it and jump there.
          auto I = llvm::find_if(llvm::reverse(Stack), [&](const Slot &x) {
            return x.StackSlotLabel == StackSlotLabel;
          });
          // If we found the label so pop from there.
          if (I != Stack.rend()) {
            CurrentValue = I->Value;
            CurrentPragmaLocation = I->PragmaLocation;
            Stack.erase(std::prev(I.base()), Stack.end());
          }
        } else if (!Stack.empty()) {
          // We do not have a label, just pop the last entry.
          CurrentValue = Stack.back().Value;
          CurrentPragmaLocation = Stack.back().PragmaLocation;
          Stack.pop_back();
        }
      }
      if (Action & PSK_Set) {
        CurrentValue = Value;
        CurrentPragmaLocation = PragmaLocation;
      }
    }

    // MSVC seems to add artificial slots to #pragma stacks on entering a C++
    // method body to restore the stacks on exit, so it works like this:
    //
    //   struct S {
    //     #pragma <name>(push, InternalPragmaSlot, <current_pragma_value>)
    //     void Method {}
    //     #pragma <name>(pop, InternalPragmaSlot)
    //   };
    //
    // It works even with #pragma vtordisp, although MSVC doesn't support
    //   #pragma vtordisp(push [, id], n)
    // syntax.
    //
    // Push / pop a named sentinel slot.
    void SentinelAction(PragmaMsStackAction Action, StringRef Label) {
      assert((Action == PSK_Push || Action == PSK_Pop) &&
             "Can only push / pop #pragma stack sentinels!");
      Act(CurrentPragmaLocation, Action, Label, CurrentValue);
    }

    // Constructors.
    explicit PragmaStack(const ValueType &Default)
        : DefaultValue(Default), CurrentValue(Default) {}

    bool hasValue() const { return CurrentValue != DefaultValue; }

    SmallVector<Slot, 2> Stack;
    ValueType DefaultValue; // Value used for PSK_Reset action.
    ValueType CurrentValue;
    SourceLocation CurrentPragmaLocation;
  };
  // FIXME: We should serialize / deserialize these if they occur in a PCH (but
  // we shouldn't do so if they're in a module).

  /// Whether to insert vtordisps prior to virtual bases in the Microsoft
  /// C++ ABI.  Possible values are 0, 1, and 2, which mean:
  ///
  /// 0: Suppress all vtordisps
  /// 1: Insert vtordisps in the presence of vbase overrides and non-trivial
  ///    structors
  /// 2: Always insert vtordisps to support RTTI on partially constructed
  ///    objects
  PragmaStack<MSVtorDispMode> VtorDispStack;
  PragmaStack<AlignPackInfo> AlignPackStack;
  // The current #pragma align/pack values and locations at each #include.
  struct AlignPackIncludeState {
    AlignPackInfo CurrentValue;
    SourceLocation CurrentPragmaLocation;
    bool HasNonDefaultValue, ShouldWarnOnInclude;
  };
  SmallVector<AlignPackIncludeState, 8> AlignPackIncludeStack;
  // Segment #pragmas.
  PragmaStack<StringLiteral *> DataSegStack;
  PragmaStack<StringLiteral *> BSSSegStack;
  PragmaStack<StringLiteral *> ConstSegStack;
  PragmaStack<StringLiteral *> CodeSegStack;

  // This stack tracks the current state of Sema.CurFPFeatures.
  PragmaStack<FPOptionsOverride> FpPragmaStack;
  FPOptionsOverride CurFPFeatureOverrides() {
    FPOptionsOverride result;
    if (!FpPragmaStack.hasValue()) {
      result = FPOptionsOverride();
    } else {
      result = FpPragmaStack.CurrentValue;
    }
    return result;
  }

  // RAII object to push / pop sentinel slots for all MS #pragma stacks.
  // Actions should be performed only if we enter / exit a C++ method body.
  class PragmaStackSentinelRAII {
  public:
    PragmaStackSentinelRAII(Sema &S, StringRef SlotLabel, bool ShouldAct);
    ~PragmaStackSentinelRAII();

  private:
    Sema &S;
    StringRef SlotLabel;
    bool ShouldAct;
  };

  /// A mapping that describes the nullability we've seen in each header file.
  FileNullabilityMap NullabilityMap;

  /// Last section used with #pragma init_seg.
  StringLiteral *CurInitSeg;
  SourceLocation CurInitSegLoc;

  /// VisContext - Manages the stack for \#pragma GCC visibility.
  void *VisContext; // Really a "PragmaVisStack*"

  /// This an attribute introduced by \#pragma clang attribute.
  struct PragmaAttributeEntry {
    SourceLocation Loc;
    ParsedAttr *Attribute;
    SmallVector<attr::SubjectMatchRule, 4> MatchRules;
    bool IsUsed;
  };

  /// A push'd group of PragmaAttributeEntries.
  struct PragmaAttributeGroup {
    /// The location of the push attribute.
    SourceLocation Loc;
    /// The namespace of this push group.
    const IdentifierInfo *Namespace;
    SmallVector<PragmaAttributeEntry, 2> Entries;
  };

  SmallVector<PragmaAttributeGroup, 2> PragmaAttributeStack;

  /// The declaration that is currently receiving an attribute from the
  /// #pragma attribute stack.
  const Decl *PragmaAttributeCurrentTargetDecl;

  /// This represents the last location of a "#pragma clang optimize off"
  /// directive if such a directive has not been closed by an "on" yet. If
  /// optimizations are currently "on", this is set to an invalid location.
  SourceLocation OptimizeOffPragmaLocation;

  /// Flag indicating if Sema is building a recovery call expression.
  ///
  /// This flag is used to avoid building recovery call expressions
  /// if Sema is already doing so, which would cause infinite recursions.
  bool IsBuildingRecoveryCallExpr;

  /// Used to control the generation of ExprWithCleanups.
  CleanupInfo Cleanup;

  /// ExprCleanupObjects - This is the stack of objects requiring
  /// cleanup that are created by the current full expression.
  SmallVector<ExprWithCleanups::CleanupObject, 8> ExprCleanupObjects;

  /// Store a set of either DeclRefExprs or MemberExprs that contain a reference
  /// to a variable (constant) that may or may not be odr-used in this Expr, and
  /// we won't know until all lvalue-to-rvalue and discarded value conversions
  /// have been applied to all subexpressions of the enclosing full expression.
  /// This is cleared at the end of each full expression.
  using MaybeODRUseExprSet = llvm::SetVector<Expr *, SmallVector<Expr *, 4>,
                                             llvm::SmallPtrSet<Expr *, 4>>;
  MaybeODRUseExprSet MaybeODRUseExprs;

  std::unique_ptr<sema::FunctionScopeInfo> CachedFunctionScope;

  /// Stack containing information about each of the nested
  /// function, block, and method scopes that are currently active.
  SmallVector<sema::FunctionScopeInfo *, 4> FunctionScopes;

  /// The index of the first FunctionScope that corresponds to the current
  /// context.
  unsigned FunctionScopesStart = 0;

  ArrayRef<sema::FunctionScopeInfo*> getFunctionScopes() const {
    return llvm::makeArrayRef(FunctionScopes.begin() + FunctionScopesStart,
                              FunctionScopes.end());
  }

  /// Stack containing information needed when in C++2a an 'auto' is encountered
  /// in a function declaration parameter type specifier in order to invent a
  /// corresponding template parameter in the enclosing abbreviated function
  /// template. This information is also present in LambdaScopeInfo, stored in
  /// the FunctionScopes stack.
  SmallVector<InventedTemplateParameterInfo, 4> InventedParameterInfos;

  /// The index of the first InventedParameterInfo that refers to the current
  /// context.
  unsigned InventedParameterInfosStart = 0;

  ArrayRef<InventedTemplateParameterInfo> getInventedParameterInfos() const {
    return llvm::makeArrayRef(InventedParameterInfos.begin() +
                                  InventedParameterInfosStart,
                              InventedParameterInfos.end());
  }

  typedef LazyVector<TypedefNameDecl *, ExternalSemaSource,
                     &ExternalSemaSource::ReadExtVectorDecls, 2, 2>
    ExtVectorDeclsType;

  /// ExtVectorDecls - This is a list all the extended vector types. This allows
  /// us to associate a raw vector type with one of the ext_vector type names.
  /// This is only necessary for issuing pretty diagnostics.
  ExtVectorDeclsType ExtVectorDecls;

  /// FieldCollector - Collects CXXFieldDecls during parsing of C++ classes.
  std::unique_ptr<CXXFieldCollector> FieldCollector;

  typedef llvm::SmallSetVector<NamedDecl *, 16> NamedDeclSetType;

  /// Set containing all declared private fields that are not used.
  NamedDeclSetType UnusedPrivateFields;

  /// Set containing all typedefs that are likely unused.
  llvm::SmallSetVector<const TypedefNameDecl *, 4>
      UnusedLocalTypedefNameCandidates;

  /// Delete-expressions to be analyzed at the end of translation unit
  ///
  /// This list contains class members, and locations of delete-expressions
  /// that could not be proven as to whether they mismatch with new-expression
  /// used in initializer of the field.
  typedef std::pair<SourceLocation, bool> DeleteExprLoc;
  typedef llvm::SmallVector<DeleteExprLoc, 4> DeleteLocs;
  llvm::MapVector<FieldDecl *, DeleteLocs> DeleteExprs;

  typedef llvm::SmallPtrSet<const CXXRecordDecl*, 8> RecordDeclSetTy;

  /// PureVirtualClassDiagSet - a set of class declarations which we have
  /// emitted a list of pure virtual functions. Used to prevent emitting the
  /// same list more than once.
  std::unique_ptr<RecordDeclSetTy> PureVirtualClassDiagSet;

  /// ParsingInitForAutoVars - a set of declarations with auto types for which
  /// we are currently parsing the initializer.
  llvm::SmallPtrSet<const Decl*, 4> ParsingInitForAutoVars;

  /// Look for a locally scoped extern "C" declaration by the given name.
  NamedDecl *findLocallyScopedExternCDecl(DeclarationName Name);

  typedef LazyVector<VarDecl *, ExternalSemaSource,
                     &ExternalSemaSource::ReadTentativeDefinitions, 2, 2>
    TentativeDefinitionsType;

  /// All the tentative definitions encountered in the TU.
  TentativeDefinitionsType TentativeDefinitions;

  /// All the external declarations encoutered and used in the TU.
  SmallVector<VarDecl *, 4> ExternalDeclarations;

  typedef LazyVector<const DeclaratorDecl *, ExternalSemaSource,
                     &ExternalSemaSource::ReadUnusedFileScopedDecls, 2, 2>
    UnusedFileScopedDeclsType;

  /// The set of file scoped decls seen so far that have not been used
  /// and must warn if not used. Only contains the first declaration.
  UnusedFileScopedDeclsType UnusedFileScopedDecls;

  typedef LazyVector<CXXConstructorDecl *, ExternalSemaSource,
                     &ExternalSemaSource::ReadDelegatingConstructors, 2, 2>
    DelegatingCtorDeclsType;

  /// All the delegating constructors seen so far in the file, used for
  /// cycle detection at the end of the TU.
  DelegatingCtorDeclsType DelegatingCtorDecls;

  /// All the overriding functions seen during a class definition
  /// that had their exception spec checks delayed, plus the overridden
  /// function.
  SmallVector<std::pair<const CXXMethodDecl*, const CXXMethodDecl*>, 2>
    DelayedOverridingExceptionSpecChecks;

  /// All the function redeclarations seen during a class definition that had
  /// their exception spec checks delayed, plus the prior declaration they
  /// should be checked against. Except during error recovery, the new decl
  /// should always be a friend declaration, as that's the only valid way to
  /// redeclare a special member before its class is complete.
  SmallVector<std::pair<FunctionDecl*, FunctionDecl*>, 2>
    DelayedEquivalentExceptionSpecChecks;

  typedef llvm::MapVector<const FunctionDecl *,
                          std::unique_ptr<LateParsedTemplate>>
      LateParsedTemplateMapT;
  LateParsedTemplateMapT LateParsedTemplateMap;

  /// Callback to the parser to parse templated functions when needed.
  typedef void LateTemplateParserCB(void *P, LateParsedTemplate &LPT);
  typedef void LateTemplateParserCleanupCB(void *P);
  LateTemplateParserCB *LateTemplateParser;
  LateTemplateParserCleanupCB *LateTemplateParserCleanup;
  void *OpaqueParser;

  void SetLateTemplateParser(LateTemplateParserCB *LTP,
                             LateTemplateParserCleanupCB *LTPCleanup,
                             void *P) {
    LateTemplateParser = LTP;
    LateTemplateParserCleanup = LTPCleanup;
    OpaqueParser = P;
  }

  class DelayedDiagnostics;

  class DelayedDiagnosticsState {
    sema::DelayedDiagnosticPool *SavedPool;
    friend class Sema::DelayedDiagnostics;
  };
  typedef DelayedDiagnosticsState ParsingDeclState;
  typedef DelayedDiagnosticsState ProcessingContextState;

  /// A class which encapsulates the logic for delaying diagnostics
  /// during parsing and other processing.
  class DelayedDiagnostics {
    /// The current pool of diagnostics into which delayed
    /// diagnostics should go.
    sema::DelayedDiagnosticPool *CurPool;

  public:
    DelayedDiagnostics() : CurPool(nullptr) {}

    /// Adds a delayed diagnostic.
    void add(const sema::DelayedDiagnostic &diag); // in DelayedDiagnostic.h

    /// Determines whether diagnostics should be delayed.
    bool shouldDelayDiagnostics() { return CurPool != nullptr; }

    /// Returns the current delayed-diagnostics pool.
    sema::DelayedDiagnosticPool *getCurrentPool() const {
      return CurPool;
    }

    /// Enter a new scope.  Access and deprecation diagnostics will be
    /// collected in this pool.
    DelayedDiagnosticsState push(sema::DelayedDiagnosticPool &pool) {
      DelayedDiagnosticsState state;
      state.SavedPool = CurPool;
      CurPool = &pool;
      return state;
    }

    /// Leave a delayed-diagnostic state that was previously pushed.
    /// Do not emit any of the diagnostics.  This is performed as part
    /// of the bookkeeping of popping a pool "properly".
    void popWithoutEmitting(DelayedDiagnosticsState state) {
      CurPool = state.SavedPool;
    }

    /// Enter a new scope where access and deprecation diagnostics are
    /// not delayed.
    DelayedDiagnosticsState pushUndelayed() {
      DelayedDiagnosticsState state;
      state.SavedPool = CurPool;
      CurPool = nullptr;
      return state;
    }

    /// Undo a previous pushUndelayed().
    void popUndelayed(DelayedDiagnosticsState state) {
      assert(CurPool == nullptr);
      CurPool = state.SavedPool;
    }
  } DelayedDiagnostics;

  /// A RAII object to temporarily push a declaration context.
  class ContextRAII {
  private:
    Sema &S;
    DeclContext *SavedContext;
    ProcessingContextState SavedContextState;
    QualType SavedCXXThisTypeOverride;
    unsigned SavedFunctionScopesStart;
    unsigned SavedInventedParameterInfosStart;

  public:
    ContextRAII(Sema &S, DeclContext *ContextToPush, bool NewThisContext = true)
      : S(S), SavedContext(S.CurContext),
        SavedContextState(S.DelayedDiagnostics.pushUndelayed()),
        SavedCXXThisTypeOverride(S.CXXThisTypeOverride),
        SavedFunctionScopesStart(S.FunctionScopesStart),
        SavedInventedParameterInfosStart(S.InventedParameterInfosStart)
    {
      assert(ContextToPush && "pushing null context");
      S.CurContext = ContextToPush;
      if (NewThisContext)
        S.CXXThisTypeOverride = QualType();
      // Any saved FunctionScopes do not refer to this context.
      S.FunctionScopesStart = S.FunctionScopes.size();
      S.InventedParameterInfosStart = S.InventedParameterInfos.size();
    }

    void pop() {
      if (!SavedContext) return;
      S.CurContext = SavedContext;
      S.DelayedDiagnostics.popUndelayed(SavedContextState);
      S.CXXThisTypeOverride = SavedCXXThisTypeOverride;
      S.FunctionScopesStart = SavedFunctionScopesStart;
      S.InventedParameterInfosStart = SavedInventedParameterInfosStart;
      SavedContext = nullptr;
    }

    ~ContextRAII() {
      pop();
    }
  };

  /// Whether the AST is currently being rebuilt to correct immediate
  /// invocations. Immediate invocation candidates and references to consteval
  /// functions aren't tracked when this is set.
  bool RebuildingImmediateInvocation = false;

  /// Used to change context to isConstantEvaluated without pushing a heavy
  /// ExpressionEvaluationContextRecord object.
  bool isConstantEvaluatedOverride;

  bool isConstantEvaluated() {
    return ExprEvalContexts.back().isConstantEvaluated() ||
           isConstantEvaluatedOverride;
  }

  /// RAII object to handle the state changes required to synthesize
  /// a function body.
  class SynthesizedFunctionScope {
    Sema &S;
    Sema::ContextRAII SavedContext;
    bool PushedCodeSynthesisContext = false;

  public:
    SynthesizedFunctionScope(Sema &S, DeclContext *DC)
        : S(S), SavedContext(S, DC) {
      S.PushFunctionScope();
      S.PushExpressionEvaluationContext(
          Sema::ExpressionEvaluationContext::PotentiallyEvaluated);
      if (auto *FD = dyn_cast<FunctionDecl>(DC))
        FD->setWillHaveBody(true);
      else
        assert(isa<ObjCMethodDecl>(DC));
    }

    void addContextNote(SourceLocation UseLoc) {
      assert(!PushedCodeSynthesisContext);

      Sema::CodeSynthesisContext Ctx;
      Ctx.Kind = Sema::CodeSynthesisContext::DefiningSynthesizedFunction;
      Ctx.PointOfInstantiation = UseLoc;
      Ctx.Entity = cast<Decl>(S.CurContext);
      S.pushCodeSynthesisContext(Ctx);

      PushedCodeSynthesisContext = true;
    }

    ~SynthesizedFunctionScope() {
      if (PushedCodeSynthesisContext)
        S.popCodeSynthesisContext();
      if (auto *FD = dyn_cast<FunctionDecl>(S.CurContext))
        FD->setWillHaveBody(false);
      S.PopExpressionEvaluationContext();
      S.PopFunctionScopeInfo();
    }
  };

  /// WeakUndeclaredIdentifiers - Identifiers contained in
  /// \#pragma weak before declared. rare. may alias another
  /// identifier, declared or undeclared
  llvm::MapVector<IdentifierInfo *, WeakInfo> WeakUndeclaredIdentifiers;

  /// ExtnameUndeclaredIdentifiers - Identifiers contained in
  /// \#pragma redefine_extname before declared.  Used in Solaris system headers
  /// to define functions that occur in multiple standards to call the version
  /// in the currently selected standard.
  llvm::DenseMap<IdentifierInfo*,AsmLabelAttr*> ExtnameUndeclaredIdentifiers;


  /// Load weak undeclared identifiers from the external source.
  void LoadExternalWeakUndeclaredIdentifiers();

  /// WeakTopLevelDecl - Translation-unit scoped declarations generated by
  /// \#pragma weak during processing of other Decls.
  /// I couldn't figure out a clean way to generate these in-line, so
  /// we store them here and handle separately -- which is a hack.
  /// It would be best to refactor this.
  SmallVector<Decl*,2> WeakTopLevelDecl;

  IdentifierResolver IdResolver;

  /// Translation Unit Scope - useful to Objective-C actions that need
  /// to lookup file scope declarations in the "ordinary" C decl namespace.
  /// For example, user-defined classes, built-in "id" type, etc.
  Scope *TUScope;

  /// The C++ "std" namespace, where the standard library resides.
  LazyDeclPtr StdNamespace;

  /// The C++ "std::bad_alloc" class, which is defined by the C++
  /// standard library.
  LazyDeclPtr StdBadAlloc;

  /// The C++ "std::align_val_t" enum class, which is defined by the C++
  /// standard library.
  LazyDeclPtr StdAlignValT;

  /// The C++ "std::experimental" namespace, where the experimental parts
  /// of the standard library resides.
  NamespaceDecl *StdExperimentalNamespaceCache;

  /// The C++ "std::initializer_list" template, which is defined in
  /// \<initializer_list>.
  ClassTemplateDecl *StdInitializerList;

  /// The C++ "std::coroutine_traits" template, which is defined in
  /// \<coroutine_traits>
  ClassTemplateDecl *StdCoroutineTraitsCache;
  /// The namespace where coroutine components are defined. In standard,
  /// they are defined in std namespace. And in the previous implementation,
  /// they are defined in std::experimental namespace.
  NamespaceDecl *CoroTraitsNamespaceCache;

  /// The C++ "type_info" declaration, which is defined in \<typeinfo>.
  RecordDecl *CXXTypeInfoDecl;

  /// The MSVC "_GUID" struct, which is defined in MSVC header files.
  RecordDecl *MSVCGuidDecl;

  /// Caches identifiers/selectors for NSFoundation APIs.
  std::unique_ptr<NSAPI> NSAPIObj;

  /// The declaration of the Objective-C NSNumber class.
  ObjCInterfaceDecl *NSNumberDecl;

  /// The declaration of the Objective-C NSValue class.
  ObjCInterfaceDecl *NSValueDecl;

  /// Pointer to NSNumber type (NSNumber *).
  QualType NSNumberPointer;

  /// Pointer to NSValue type (NSValue *).
  QualType NSValuePointer;

  /// The Objective-C NSNumber methods used to create NSNumber literals.
  ObjCMethodDecl *NSNumberLiteralMethods[NSAPI::NumNSNumberLiteralMethods];

  /// The declaration of the Objective-C NSString class.
  ObjCInterfaceDecl *NSStringDecl;

  /// Pointer to NSString type (NSString *).
  QualType NSStringPointer;

  /// The declaration of the stringWithUTF8String: method.
  ObjCMethodDecl *StringWithUTF8StringMethod;

  /// The declaration of the valueWithBytes:objCType: method.
  ObjCMethodDecl *ValueWithBytesObjCTypeMethod;

  /// The declaration of the Objective-C NSArray class.
  ObjCInterfaceDecl *NSArrayDecl;

  /// The declaration of the arrayWithObjects:count: method.
  ObjCMethodDecl *ArrayWithObjectsMethod;

  /// The declaration of the Objective-C NSDictionary class.
  ObjCInterfaceDecl *NSDictionaryDecl;

  /// The declaration of the dictionaryWithObjects:forKeys:count: method.
  ObjCMethodDecl *DictionaryWithObjectsMethod;

  /// id<NSCopying> type.
  QualType QIDNSCopying;

  /// will hold 'respondsToSelector:'
  Selector RespondsToSelectorSel;

  /// A flag to remember whether the implicit forms of operator new and delete
  /// have been declared.
  bool GlobalNewDeleteDeclared;

  /// Describes how the expressions currently being parsed are
  /// evaluated at run-time, if at all.
  enum class ExpressionEvaluationContext {
    /// The current expression and its subexpressions occur within an
    /// unevaluated operand (C++11 [expr]p7), such as the subexpression of
    /// \c sizeof, where the type of the expression may be significant but
    /// no code will be generated to evaluate the value of the expression at
    /// run time.
    Unevaluated,

    /// The current expression occurs within a braced-init-list within
    /// an unevaluated operand. This is mostly like a regular unevaluated
    /// context, except that we still instantiate constexpr functions that are
    /// referenced here so that we can perform narrowing checks correctly.
    UnevaluatedList,

    /// The current expression occurs within a discarded statement.
    /// This behaves largely similarly to an unevaluated operand in preventing
    /// definitions from being required, but not in other ways.
    DiscardedStatement,

    /// The current expression occurs within an unevaluated
    /// operand that unconditionally permits abstract references to
    /// fields, such as a SIZE operator in MS-style inline assembly.
    UnevaluatedAbstract,

    /// The current context is "potentially evaluated" in C++11 terms,
    /// but the expression is evaluated at compile-time (like the values of
    /// cases in a switch statement).
    ConstantEvaluated,

    /// In addition of being constant evaluated, the current expression
    /// occurs in an immediate function context - either a consteval function
    /// or a consteval if function.
    ImmediateFunctionContext,

    /// The current expression is potentially evaluated at run time,
    /// which means that code may be generated to evaluate the value of the
    /// expression at run time.
    PotentiallyEvaluated,

    /// The current expression is potentially evaluated, but any
    /// declarations referenced inside that expression are only used if
    /// in fact the current expression is used.
    ///
    /// This value is used when parsing default function arguments, for which
    /// we would like to provide diagnostics (e.g., passing non-POD arguments
    /// through varargs) but do not want to mark declarations as "referenced"
    /// until the default argument is used.
    PotentiallyEvaluatedIfUsed
  };

  using ImmediateInvocationCandidate = llvm::PointerIntPair<ConstantExpr *, 1>;

  /// Data structure used to record current or nested
  /// expression evaluation contexts.
  struct ExpressionEvaluationContextRecord {
    /// The expression evaluation context.
    ExpressionEvaluationContext Context;

    /// Whether the enclosing context needed a cleanup.
    CleanupInfo ParentCleanup;

    /// The number of active cleanup objects when we entered
    /// this expression evaluation context.
    unsigned NumCleanupObjects;

    /// The number of typos encountered during this expression evaluation
    /// context (i.e. the number of TypoExprs created).
    unsigned NumTypos;

    MaybeODRUseExprSet SavedMaybeODRUseExprs;

    /// The lambdas that are present within this context, if it
    /// is indeed an unevaluated context.
    SmallVector<LambdaExpr *, 2> Lambdas;

    /// The declaration that provides context for lambda expressions
    /// and block literals if the normal declaration context does not
    /// suffice, e.g., in a default function argument.
    Decl *ManglingContextDecl;

    /// If we are processing a decltype type, a set of call expressions
    /// for which we have deferred checking the completeness of the return type.
    SmallVector<CallExpr *, 8> DelayedDecltypeCalls;

    /// If we are processing a decltype type, a set of temporary binding
    /// expressions for which we have deferred checking the destructor.
    SmallVector<CXXBindTemporaryExpr *, 8> DelayedDecltypeBinds;

    llvm::SmallPtrSet<const Expr *, 8> PossibleDerefs;

    /// Expressions appearing as the LHS of a volatile assignment in this
    /// context. We produce a warning for these when popping the context if
    /// they are not discarded-value expressions nor unevaluated operands.
    SmallVector<Expr*, 2> VolatileAssignmentLHSs;

    /// Set of candidates for starting an immediate invocation.
    llvm::SmallVector<ImmediateInvocationCandidate, 4> ImmediateInvocationCandidates;

    /// Set of DeclRefExprs referencing a consteval function when used in a
    /// context not already known to be immediately invoked.
    llvm::SmallPtrSet<DeclRefExpr *, 4> ReferenceToConsteval;

    /// \brief Describes whether we are in an expression constext which we have
    /// to handle differently.
    enum ExpressionKind {
      EK_Decltype, EK_TemplateArgument, EK_Other
    } ExprContext;

    // A context can be nested in both a discarded statement context and
    // an immediate function context, so they need to be tracked independently.
    bool InDiscardedStatement;
    bool InImmediateFunctionContext;

    ExpressionEvaluationContextRecord(ExpressionEvaluationContext Context,
                                      unsigned NumCleanupObjects,
                                      CleanupInfo ParentCleanup,
                                      Decl *ManglingContextDecl,
                                      ExpressionKind ExprContext)
        : Context(Context), ParentCleanup(ParentCleanup),
          NumCleanupObjects(NumCleanupObjects), NumTypos(0),
          ManglingContextDecl(ManglingContextDecl), ExprContext(ExprContext),
          InDiscardedStatement(false), InImmediateFunctionContext(false) {}

    bool isUnevaluated() const {
      return Context == ExpressionEvaluationContext::Unevaluated ||
             Context == ExpressionEvaluationContext::UnevaluatedAbstract ||
             Context == ExpressionEvaluationContext::UnevaluatedList;
    }

    bool isConstantEvaluated() const {
      return Context == ExpressionEvaluationContext::ConstantEvaluated ||
             Context == ExpressionEvaluationContext::ImmediateFunctionContext;
    }

    bool isImmediateFunctionContext() const {
      return Context == ExpressionEvaluationContext::ImmediateFunctionContext ||
             (Context == ExpressionEvaluationContext::DiscardedStatement &&
              InImmediateFunctionContext);
    }

    bool isDiscardedStatementContext() const {
      return Context == ExpressionEvaluationContext::DiscardedStatement ||
             (Context ==
                  ExpressionEvaluationContext::ImmediateFunctionContext &&
              InDiscardedStatement);
    }
  };

  /// A stack of expression evaluation contexts.
  SmallVector<ExpressionEvaluationContextRecord, 8> ExprEvalContexts;

  /// Emit a warning for all pending noderef expressions that we recorded.
  void WarnOnPendingNoDerefs(ExpressionEvaluationContextRecord &Rec);

  /// Compute the mangling number context for a lambda expression or
  /// block literal. Also return the extra mangling decl if any.
  ///
  /// \param DC - The DeclContext containing the lambda expression or
  /// block literal.
  std::tuple<MangleNumberingContext *, Decl *>
  getCurrentMangleNumberContext(const DeclContext *DC);


  /// SpecialMemberOverloadResult - The overloading result for a special member
  /// function.
  ///
  /// This is basically a wrapper around PointerIntPair. The lowest bits of the
  /// integer are used to determine whether overload resolution succeeded.
  class SpecialMemberOverloadResult {
  public:
    enum Kind {
      NoMemberOrDeleted,
      Ambiguous,
      Success
    };

  private:
    llvm::PointerIntPair<CXXMethodDecl *, 2> Pair;

  public:
    SpecialMemberOverloadResult() {}
    SpecialMemberOverloadResult(CXXMethodDecl *MD)
        : Pair(MD, MD->isDeleted() ? NoMemberOrDeleted : Success) {}

    CXXMethodDecl *getMethod() const { return Pair.getPointer(); }
    void setMethod(CXXMethodDecl *MD) { Pair.setPointer(MD); }

    Kind getKind() const { return static_cast<Kind>(Pair.getInt()); }
    void setKind(Kind K) { Pair.setInt(K); }
  };

  class SpecialMemberOverloadResultEntry
      : public llvm::FastFoldingSetNode,
        public SpecialMemberOverloadResult {
  public:
    SpecialMemberOverloadResultEntry(const llvm::FoldingSetNodeID &ID)
      : FastFoldingSetNode(ID)
    {}
  };

  /// A cache of special member function overload resolution results
  /// for C++ records.
  llvm::FoldingSet<SpecialMemberOverloadResultEntry> SpecialMemberCache;

  /// A cache of the flags available in enumerations with the flag_bits
  /// attribute.
  mutable llvm::DenseMap<const EnumDecl*, llvm::APInt> FlagBitsCache;

  /// The kind of translation unit we are processing.
  ///
  /// When we're processing a complete translation unit, Sema will perform
  /// end-of-translation-unit semantic tasks (such as creating
  /// initializers for tentative definitions in C) once parsing has
  /// completed. Modules and precompiled headers perform different kinds of
  /// checks.
  const TranslationUnitKind TUKind;

  llvm::BumpPtrAllocator BumpAlloc;

  /// The number of SFINAE diagnostics that have been trapped.
  unsigned NumSFINAEErrors;

  typedef llvm::DenseMap<ParmVarDecl *, llvm::TinyPtrVector<ParmVarDecl *>>
    UnparsedDefaultArgInstantiationsMap;

  /// A mapping from parameters with unparsed default arguments to the
  /// set of instantiations of each parameter.
  ///
  /// This mapping is a temporary data structure used when parsing
  /// nested class templates or nested classes of class templates,
  /// where we might end up instantiating an inner class before the
  /// default arguments of its methods have been parsed.
  UnparsedDefaultArgInstantiationsMap UnparsedDefaultArgInstantiations;

  // Contains the locations of the beginning of unparsed default
  // argument locations.
  llvm::DenseMap<ParmVarDecl *, SourceLocation> UnparsedDefaultArgLocs;

  /// UndefinedInternals - all the used, undefined objects which require a
  /// definition in this translation unit.
  llvm::MapVector<NamedDecl *, SourceLocation> UndefinedButUsed;

  /// Determine if VD, which must be a variable or function, is an external
  /// symbol that nonetheless can't be referenced from outside this translation
  /// unit because its type has no linkage and it's not extern "C".
  bool isExternalWithNoLinkageType(ValueDecl *VD);

  /// Obtain a sorted list of functions that are undefined but ODR-used.
  void getUndefinedButUsed(
      SmallVectorImpl<std::pair<NamedDecl *, SourceLocation> > &Undefined);

  /// Retrieves list of suspicious delete-expressions that will be checked at
  /// the end of translation unit.
  const llvm::MapVector<FieldDecl *, DeleteLocs> &
  getMismatchingDeleteExpressions() const;

  class GlobalMethodPool {
  public:
    using Lists = std::pair<ObjCMethodList, ObjCMethodList>;
    using iterator = llvm::DenseMap<Selector, Lists>::iterator;
    iterator begin() { return Methods.begin(); }
    iterator end() { return Methods.end(); }
    iterator find(Selector Sel) { return Methods.find(Sel); }
    std::pair<iterator, bool> insert(std::pair<Selector, Lists> &&Val) {
      return Methods.insert(Val);
    }
    int count(Selector Sel) const { return Methods.count(Sel); }
    bool empty() const { return Methods.empty(); }

  private:
    llvm::DenseMap<Selector, Lists> Methods;
  };

  /// Method Pool - allows efficient lookup when typechecking messages to "id".
  /// We need to maintain a list, since selectors can have differing signatures
  /// across classes. In Cocoa, this happens to be extremely uncommon (only 1%
  /// of selectors are "overloaded").
  /// At the head of the list it is recorded whether there were 0, 1, or >= 2
  /// methods inside categories with a particular selector.
  GlobalMethodPool MethodPool;

  /// Method selectors used in a \@selector expression. Used for implementation
  /// of -Wselector.
  llvm::MapVector<Selector, SourceLocation> ReferencedSelectors;

  /// List of SourceLocations where 'self' is implicitly retained inside a
  /// block.
  llvm::SmallVector<std::pair<SourceLocation, const BlockDecl *>, 1>
      ImplicitlyRetainedSelfLocs;

  /// Kinds of C++ special members.
  enum CXXSpecialMember {
    CXXDefaultConstructor,
    CXXCopyConstructor,
    CXXMoveConstructor,
    CXXCopyAssignment,
    CXXMoveAssignment,
    CXXDestructor,
    CXXInvalid
  };

  typedef llvm::PointerIntPair<CXXRecordDecl *, 3, CXXSpecialMember>
      SpecialMemberDecl;

  /// The C++ special members which we are currently in the process of
  /// declaring. If this process recursively triggers the declaration of the
  /// same special member, we should act as if it is not yet declared.
  llvm::SmallPtrSet<SpecialMemberDecl, 4> SpecialMembersBeingDeclared;

  /// Kinds of defaulted comparison operator functions.
  enum class DefaultedComparisonKind : unsigned char {
    /// This is not a defaultable comparison operator.
    None,
    /// This is an operator== that should be implemented as a series of
    /// subobject comparisons.
    Equal,
    /// This is an operator<=> that should be implemented as a series of
    /// subobject comparisons.
    ThreeWay,
    /// This is an operator!= that should be implemented as a rewrite in terms
    /// of a == comparison.
    NotEqual,
    /// This is an <, <=, >, or >= that should be implemented as a rewrite in
    /// terms of a <=> comparison.
    Relational,
  };

  /// The function definitions which were renamed as part of typo-correction
  /// to match their respective declarations. We want to keep track of them
  /// to ensure that we don't emit a "redefinition" error if we encounter a
  /// correctly named definition after the renamed definition.
  llvm::SmallPtrSet<const NamedDecl *, 4> TypoCorrectedFunctionDefinitions;

  /// Stack of types that correspond to the parameter entities that are
  /// currently being copy-initialized. Can be empty.
  llvm::SmallVector<QualType, 4> CurrentParameterCopyTypes;

  void ReadMethodPool(Selector Sel);
  void updateOutOfDateSelector(Selector Sel);

  /// Private Helper predicate to check for 'self'.
  bool isSelfExpr(Expr *RExpr);
  bool isSelfExpr(Expr *RExpr, const ObjCMethodDecl *Method);

  /// Cause the active diagnostic on the DiagosticsEngine to be
  /// emitted. This is closely coupled to the SemaDiagnosticBuilder class and
  /// should not be used elsewhere.
  void EmitCurrentDiagnostic(unsigned DiagID);

  /// Records and restores the CurFPFeatures state on entry/exit of compound
  /// statements.
  class FPFeaturesStateRAII {
  public:
    FPFeaturesStateRAII(Sema &S) : S(S), OldFPFeaturesState(S.CurFPFeatures) {
      OldOverrides = S.FpPragmaStack.CurrentValue;
    }
    ~FPFeaturesStateRAII() {
      S.CurFPFeatures = OldFPFeaturesState;
      S.FpPragmaStack.CurrentValue = OldOverrides;
    }
    FPOptionsOverride getOverrides() { return OldOverrides; }

  private:
    Sema& S;
    FPOptions OldFPFeaturesState;
    FPOptionsOverride OldOverrides;
  };

  void addImplicitTypedef(StringRef Name, QualType T);

  bool WarnedStackExhausted = false;

  /// Increment when we find a reference; decrement when we find an ignored
  /// assignment.  Ultimately the value is 0 if every reference is an ignored
  /// assignment.
  llvm::DenseMap<const VarDecl *, int> RefsMinusAssignments;

  Optional<std::unique_ptr<DarwinSDKInfo>> CachedDarwinSDKInfo;

public:
  Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
       TranslationUnitKind TUKind = TU_Complete,
       CodeCompleteConsumer *CompletionConsumer = nullptr);
  ~Sema();

  /// Perform initialization that occurs after the parser has been
  /// initialized but before it parses anything.
  void Initialize();

  /// This virtual key function only exists to limit the emission of debug info
  /// describing the Sema class. GCC and Clang only emit debug info for a class
  /// with a vtable when the vtable is emitted. Sema is final and not
  /// polymorphic, but the debug info size savings are so significant that it is
  /// worth adding a vtable just to take advantage of this optimization.
  virtual void anchor();

  const LangOptions &getLangOpts() const { return LangOpts; }
  OpenCLOptions &getOpenCLOptions() { return OpenCLFeatures; }
  FPOptions     &getCurFPFeatures() { return CurFPFeatures; }

  DiagnosticsEngine &getDiagnostics() const { return Diags; }
  SourceManager &getSourceManager() const { return SourceMgr; }
  Preprocessor &getPreprocessor() const { return PP; }
  ASTContext &getASTContext() const { return Context; }
  ASTConsumer &getASTConsumer() const { return Consumer; }
  ASTMutationListener *getASTMutationListener() const;
  ExternalSemaSource* getExternalSource() const { return ExternalSource; }
  DarwinSDKInfo *getDarwinSDKInfoForAvailabilityChecking(SourceLocation Loc,
                                                         StringRef Platform);

  ///Registers an external source. If an external source already exists,
  /// creates a multiplex external source and appends to it.
  ///
  ///\param[in] E - A non-null external sema source.
  ///
  void addExternalSource(ExternalSemaSource *E);

  void PrintStats() const;

  /// Warn that the stack is nearly exhausted.
  void warnStackExhausted(SourceLocation Loc);

  /// Run some code with "sufficient" stack space. (Currently, at least 256K is
  /// guaranteed). Produces a warning if we're low on stack space and allocates
  /// more in that case. Use this in code that may recurse deeply (for example,
  /// in template instantiation) to avoid stack overflow.
  void runWithSufficientStackSpace(SourceLocation Loc,
                                   llvm::function_ref<void()> Fn);

  /// Helper class that creates diagnostics with optional
  /// template instantiation stacks.
  ///
  /// This class provides a wrapper around the basic DiagnosticBuilder
  /// class that emits diagnostics. ImmediateDiagBuilder is
  /// responsible for emitting the diagnostic (as DiagnosticBuilder
  /// does) and, if the diagnostic comes from inside a template
  /// instantiation, printing the template instantiation stack as
  /// well.
  class ImmediateDiagBuilder : public DiagnosticBuilder {
    Sema &SemaRef;
    unsigned DiagID;

  public:
    ImmediateDiagBuilder(DiagnosticBuilder &DB, Sema &SemaRef, unsigned DiagID)
        : DiagnosticBuilder(DB), SemaRef(SemaRef), DiagID(DiagID) {}
    ImmediateDiagBuilder(DiagnosticBuilder &&DB, Sema &SemaRef, unsigned DiagID)
        : DiagnosticBuilder(DB), SemaRef(SemaRef), DiagID(DiagID) {}

    // This is a cunning lie. DiagnosticBuilder actually performs move
    // construction in its copy constructor (but due to varied uses, it's not
    // possible to conveniently express this as actual move construction). So
    // the default copy ctor here is fine, because the base class disables the
    // source anyway, so the user-defined ~ImmediateDiagBuilder is a safe no-op
    // in that case anwyay.
    ImmediateDiagBuilder(const ImmediateDiagBuilder &) = default;

    ~ImmediateDiagBuilder() {
      // If we aren't active, there is nothing to do.
      if (!isActive()) return;

      // Otherwise, we need to emit the diagnostic. First clear the diagnostic
      // builder itself so it won't emit the diagnostic in its own destructor.
      //
      // This seems wasteful, in that as written the DiagnosticBuilder dtor will
      // do its own needless checks to see if the diagnostic needs to be
      // emitted. However, because we take care to ensure that the builder
      // objects never escape, a sufficiently smart compiler will be able to
      // eliminate that code.
      Clear();

      // Dispatch to Sema to emit the diagnostic.
      SemaRef.EmitCurrentDiagnostic(DiagID);
    }

    /// Teach operator<< to produce an object of the correct type.
    template <typename T>
    friend const ImmediateDiagBuilder &
    operator<<(const ImmediateDiagBuilder &Diag, const T &Value) {
      const DiagnosticBuilder &BaseDiag = Diag;
      BaseDiag << Value;
      return Diag;
    }

    // It is necessary to limit this to rvalue reference to avoid calling this
    // function with a bitfield lvalue argument since non-const reference to
    // bitfield is not allowed.
    template <typename T, typename = typename std::enable_if<
                              !std::is_lvalue_reference<T>::value>::type>
    const ImmediateDiagBuilder &operator<<(T &&V) const {
      const DiagnosticBuilder &BaseDiag = *this;
      BaseDiag << std::move(V);
      return *this;
    }
  };

  /// A generic diagnostic builder for errors which may or may not be deferred.
  ///
  /// In CUDA, there exist constructs (e.g. variable-length arrays, try/catch)
  /// which are not allowed to appear inside __device__ functions and are
  /// allowed to appear in __host__ __device__ functions only if the host+device
  /// function is never codegen'ed.
  ///
  /// To handle this, we use the notion of "deferred diagnostics", where we
  /// attach a diagnostic to a FunctionDecl that's emitted iff it's codegen'ed.
  ///
  /// This class lets you emit either a regular diagnostic, a deferred
  /// diagnostic, or no diagnostic at all, according to an argument you pass to
  /// its constructor, thus simplifying the process of creating these "maybe
  /// deferred" diagnostics.
  class SemaDiagnosticBuilder {
  public:
    enum Kind {
      /// Emit no diagnostics.
      K_Nop,
      /// Emit the diagnostic immediately (i.e., behave like Sema::Diag()).
      K_Immediate,
      /// Emit the diagnostic immediately, and, if it's a warning or error, also
      /// emit a call stack showing how this function can be reached by an a
      /// priori known-emitted function.
      K_ImmediateWithCallStack,
      /// Create a deferred diagnostic, which is emitted only if the function
      /// it's attached to is codegen'ed.  Also emit a call stack as with
      /// K_ImmediateWithCallStack.
      K_Deferred
    };

    SemaDiagnosticBuilder(Kind K, SourceLocation Loc, unsigned DiagID,
                          FunctionDecl *Fn, Sema &S);
    SemaDiagnosticBuilder(SemaDiagnosticBuilder &&D);
    SemaDiagnosticBuilder(const SemaDiagnosticBuilder &) = default;
    ~SemaDiagnosticBuilder();

    bool isImmediate() const { return ImmediateDiag.hasValue(); }

    /// Convertible to bool: True if we immediately emitted an error, false if
    /// we didn't emit an error or we created a deferred error.
    ///
    /// Example usage:
    ///
    ///   if (SemaDiagnosticBuilder(...) << foo << bar)
    ///     return ExprError();
    ///
    /// But see CUDADiagIfDeviceCode() and CUDADiagIfHostCode() -- you probably
    /// want to use these instead of creating a SemaDiagnosticBuilder yourself.
    operator bool() const { return isImmediate(); }

    template <typename T>
    friend const SemaDiagnosticBuilder &
    operator<<(const SemaDiagnosticBuilder &Diag, const T &Value) {
      if (Diag.ImmediateDiag.hasValue())
        *Diag.ImmediateDiag << Value;
      else if (Diag.PartialDiagId.hasValue())
        Diag.S.DeviceDeferredDiags[Diag.Fn][*Diag.PartialDiagId].second
            << Value;
      return Diag;
    }

    // It is necessary to limit this to rvalue reference to avoid calling this
    // function with a bitfield lvalue argument since non-const reference to
    // bitfield is not allowed.
    template <typename T, typename = typename std::enable_if<
                              !std::is_lvalue_reference<T>::value>::type>
    const SemaDiagnosticBuilder &operator<<(T &&V) const {
      if (ImmediateDiag.hasValue())
        *ImmediateDiag << std::move(V);
      else if (PartialDiagId.hasValue())
        S.DeviceDeferredDiags[Fn][*PartialDiagId].second << std::move(V);
      return *this;
    }

    friend const SemaDiagnosticBuilder &
    operator<<(const SemaDiagnosticBuilder &Diag, const PartialDiagnostic &PD) {
      if (Diag.ImmediateDiag.hasValue())
        PD.Emit(*Diag.ImmediateDiag);
      else if (Diag.PartialDiagId.hasValue())
        Diag.S.DeviceDeferredDiags[Diag.Fn][*Diag.PartialDiagId].second = PD;
      return Diag;
    }

    void AddFixItHint(const FixItHint &Hint) const {
      if (ImmediateDiag.hasValue())
        ImmediateDiag->AddFixItHint(Hint);
      else if (PartialDiagId.hasValue())
        S.DeviceDeferredDiags[Fn][*PartialDiagId].second.AddFixItHint(Hint);
    }

    friend ExprResult ExprError(const SemaDiagnosticBuilder &) {
      return ExprError();
    }
    friend StmtResult StmtError(const SemaDiagnosticBuilder &) {
      return StmtError();
    }
    operator ExprResult() const { return ExprError(); }
    operator StmtResult() const { return StmtError(); }
    operator TypeResult() const { return TypeError(); }
    operator DeclResult() const { return DeclResult(true); }
    operator MemInitResult() const { return MemInitResult(true); }

  private:
    Sema &S;
    SourceLocation Loc;
    unsigned DiagID;
    FunctionDecl *Fn;
    bool ShowCallStack;

    // Invariant: At most one of these Optionals has a value.
    // FIXME: Switch these to a Variant once that exists.
    llvm::Optional<ImmediateDiagBuilder> ImmediateDiag;
    llvm::Optional<unsigned> PartialDiagId;
  };

  /// Is the last error level diagnostic immediate. This is used to determined
  /// whether the next info diagnostic should be immediate.
  bool IsLastErrorImmediate = true;

  /// Emit a diagnostic.
  SemaDiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID,
                             bool DeferHint = false);

  /// Emit a partial diagnostic.
  SemaDiagnosticBuilder Diag(SourceLocation Loc, const PartialDiagnostic &PD,
                             bool DeferHint = false);

  /// Build a partial diagnostic.
  PartialDiagnostic PDiag(unsigned DiagID = 0); // in SemaInternal.h

  /// Whether deferrable diagnostics should be deferred.
  bool DeferDiags = false;

  /// RAII class to control scope of DeferDiags.
  class DeferDiagsRAII {
    Sema &S;
    bool SavedDeferDiags = false;

  public:
    DeferDiagsRAII(Sema &S, bool DeferDiags)
        : S(S), SavedDeferDiags(S.DeferDiags) {
      S.DeferDiags = DeferDiags;
    }
    ~DeferDiagsRAII() { S.DeferDiags = SavedDeferDiags; }
  };

  /// Whether uncompilable error has occurred. This includes error happens
  /// in deferred diagnostics.
  bool hasUncompilableErrorOccurred() const;

  bool findMacroSpelling(SourceLocation &loc, StringRef name);

  /// Get a string to suggest for zero-initialization of a type.
  std::string
  getFixItZeroInitializerForType(QualType T, SourceLocation Loc) const;
  std::string getFixItZeroLiteralForType(QualType T, SourceLocation Loc) const;

  /// Calls \c Lexer::getLocForEndOfToken()
  SourceLocation getLocForEndOfToken(SourceLocation Loc, unsigned Offset = 0);

  /// Retrieve the module loader associated with the preprocessor.
  ModuleLoader &getModuleLoader() const;

  /// Invent a new identifier for parameters of abbreviated templates.
  IdentifierInfo *
  InventAbbreviatedTemplateParameterTypeName(IdentifierInfo *ParamName,
                                             unsigned Index);

  void emitAndClearUnusedLocalTypedefWarnings();

  private:
    /// Function or variable declarations to be checked for whether the deferred
    /// diagnostics should be emitted.
    llvm::SmallSetVector<Decl *, 4> DeclsToCheckForDeferredDiags;

  public:
  // Emit all deferred diagnostics.
  void emitDeferredDiags();

  enum TUFragmentKind {
    /// The global module fragment, between 'module;' and a module-declaration.
    Global,
    /// A normal translation unit fragment. For a non-module unit, this is the
    /// entire translation unit. Otherwise, it runs from the module-declaration
    /// to the private-module-fragment (if any) or the end of the TU (if not).
    Normal,
    /// The private module fragment, between 'module :private;' and the end of
    /// the translation unit.
    Private
  };

  void ActOnStartOfTranslationUnit();
  void ActOnEndOfTranslationUnit();
  void ActOnEndOfTranslationUnitFragment(TUFragmentKind Kind);

  void CheckDelegatingCtorCycles();

  Scope *getScopeForContext(DeclContext *Ctx);

  void PushFunctionScope();
  void PushBlockScope(Scope *BlockScope, BlockDecl *Block);
  sema::LambdaScopeInfo *PushLambdaScope();

  /// This is used to inform Sema what the current TemplateParameterDepth
  /// is during Parsing.  Currently it is used to pass on the depth
  /// when parsing generic lambda 'auto' parameters.
  void RecordParsingTemplateParameterDepth(unsigned Depth);

  void PushCapturedRegionScope(Scope *RegionScope, CapturedDecl *CD,
                               RecordDecl *RD, CapturedRegionKind K,
                               unsigned OpenMPCaptureLevel = 0);

  /// Custom deleter to allow FunctionScopeInfos to be kept alive for a short
  /// time after they've been popped.
  class PoppedFunctionScopeDeleter {
    Sema *Self;

  public:
    explicit PoppedFunctionScopeDeleter(Sema *Self) : Self(Self) {}
    void operator()(sema::FunctionScopeInfo *Scope) const;
  };

  using PoppedFunctionScopePtr =
      std::unique_ptr<sema::FunctionScopeInfo, PoppedFunctionScopeDeleter>;

  PoppedFunctionScopePtr
  PopFunctionScopeInfo(const sema::AnalysisBasedWarnings::Policy *WP = nullptr,
                       const Decl *D = nullptr,
                       QualType BlockType = QualType());

  sema::FunctionScopeInfo *getCurFunction() const {
    return FunctionScopes.empty() ? nullptr : FunctionScopes.back();
  }

  sema::FunctionScopeInfo *getEnclosingFunction() const;

  void setFunctionHasBranchIntoScope();
  void setFunctionHasBranchProtectedScope();
  void setFunctionHasIndirectGoto();
  void setFunctionHasMustTail();

  void PushCompoundScope(bool IsStmtExpr);
  void PopCompoundScope();

  sema::CompoundScopeInfo &getCurCompoundScope() const;

  bool hasAnyUnrecoverableErrorsInThisFunction() const;

  /// Retrieve the current block, if any.
  sema::BlockScopeInfo *getCurBlock();

  /// Get the innermost lambda enclosing the current location, if any. This
  /// looks through intervening non-lambda scopes such as local functions and
  /// blocks.
  sema::LambdaScopeInfo *getEnclosingLambda() const;

  /// Retrieve the current lambda scope info, if any.
  /// \param IgnoreNonLambdaCapturingScope true if should find the top-most
  /// lambda scope info ignoring all inner capturing scopes that are not
  /// lambda scopes.
  sema::LambdaScopeInfo *
  getCurLambda(bool IgnoreNonLambdaCapturingScope = false);

  /// Retrieve the current generic lambda info, if any.
  sema::LambdaScopeInfo *getCurGenericLambda();

  /// Retrieve the current captured region, if any.
  sema::CapturedRegionScopeInfo *getCurCapturedRegion();

  /// Retrieve the current function, if any, that should be analyzed for
  /// potential availability violations.
  sema::FunctionScopeInfo *getCurFunctionAvailabilityContext();

  /// WeakTopLevelDeclDecls - access to \#pragma weak-generated Decls
  SmallVectorImpl<Decl *> &WeakTopLevelDecls() { return WeakTopLevelDecl; }

  /// Called before parsing a function declarator belonging to a function
  /// declaration.
  void ActOnStartFunctionDeclarationDeclarator(Declarator &D,
                                               unsigned TemplateParameterDepth);

  /// Called after parsing a function declarator belonging to a function
  /// declaration.
  void ActOnFinishFunctionDeclarationDeclarator(Declarator &D);

  void ActOnComment(SourceRange Comment);

  //===--------------------------------------------------------------------===//
  // Type Analysis / Processing: SemaType.cpp.
  //

  QualType BuildQualifiedType(QualType T, SourceLocation Loc, Qualifiers Qs,
                              const DeclSpec *DS = nullptr);
  QualType BuildQualifiedType(QualType T, SourceLocation Loc, unsigned CVRA,
                              const DeclSpec *DS = nullptr);
  QualType BuildPointerType(QualType T,
                            SourceLocation Loc, DeclarationName Entity);
  QualType BuildReferenceType(QualType T, bool LValueRef,
                              SourceLocation Loc, DeclarationName Entity);
  QualType BuildArrayType(QualType T, ArrayType::ArraySizeModifier ASM,
                          Expr *ArraySize, unsigned Quals,
                          SourceRange Brackets, DeclarationName Entity);
  QualType BuildVectorType(QualType T, Expr *VecSize, SourceLocation AttrLoc);
  QualType BuildExtVectorType(QualType T, Expr *ArraySize,
                              SourceLocation AttrLoc);
  QualType BuildMatrixType(QualType T, Expr *NumRows, Expr *NumColumns,
                           SourceLocation AttrLoc);

  QualType BuildAddressSpaceAttr(QualType &T, LangAS ASIdx, Expr *AddrSpace,
                                 SourceLocation AttrLoc);

  /// Same as above, but constructs the AddressSpace index if not provided.
  QualType BuildAddressSpaceAttr(QualType &T, Expr *AddrSpace,
                                 SourceLocation AttrLoc);

  bool CheckQualifiedFunctionForTypeId(QualType T, SourceLocation Loc);

  bool CheckFunctionReturnType(QualType T, SourceLocation Loc);

  /// Build a function type.
  ///
  /// This routine checks the function type according to C++ rules and
  /// under the assumption that the result type and parameter types have
  /// just been instantiated from a template. It therefore duplicates
  /// some of the behavior of GetTypeForDeclarator, but in a much
  /// simpler form that is only suitable for this narrow use case.
  ///
  /// \param T The return type of the function.
  ///
  /// \param ParamTypes The parameter types of the function. This array
  /// will be modified to account for adjustments to the types of the
  /// function parameters.
  ///
  /// \param Loc The location of the entity whose type involves this
  /// function type or, if there is no such entity, the location of the
  /// type that will have function type.
  ///
  /// \param Entity The name of the entity that involves the function
  /// type, if known.
  ///
  /// \param EPI Extra information about the function type. Usually this will
  /// be taken from an existing function with the same prototype.
  ///
  /// \returns A suitable function type, if there are no errors. The
  /// unqualified type will always be a FunctionProtoType.
  /// Otherwise, returns a NULL type.
  QualType BuildFunctionType(QualType T,
                             MutableArrayRef<QualType> ParamTypes,
                             SourceLocation Loc, DeclarationName Entity,
                             const FunctionProtoType::ExtProtoInfo &EPI);

  QualType BuildMemberPointerType(QualType T, QualType Class,
                                  SourceLocation Loc,
                                  DeclarationName Entity);
  QualType BuildBlockPointerType(QualType T,
                                 SourceLocation Loc, DeclarationName Entity);
  QualType BuildParenType(QualType T);
  QualType BuildAtomicType(QualType T, SourceLocation Loc);
  QualType BuildReadPipeType(QualType T,
                         SourceLocation Loc);
  QualType BuildWritePipeType(QualType T,
                         SourceLocation Loc);
  QualType BuildBitIntType(bool IsUnsigned, Expr *BitWidth, SourceLocation Loc);

  TypeSourceInfo *GetTypeForDeclarator(Declarator &D, Scope *S);
  TypeSourceInfo *GetTypeForDeclaratorCast(Declarator &D, QualType FromTy);

  /// Package the given type and TSI into a ParsedType.
  ParsedType CreateParsedType(QualType T, TypeSourceInfo *TInfo);
  DeclarationNameInfo GetNameForDeclarator(Declarator &D);
  DeclarationNameInfo GetNameFromUnqualifiedId(const UnqualifiedId &Name);
  static QualType GetTypeFromParser(ParsedType Ty,
                                    TypeSourceInfo **TInfo = nullptr);
  CanThrowResult canThrow(const Stmt *E);
  /// Determine whether the callee of a particular function call can throw.
  /// E, D and Loc are all optional.
  static CanThrowResult canCalleeThrow(Sema &S, const Expr *E, const Decl *D,
                                       SourceLocation Loc = SourceLocation());
  const FunctionProtoType *ResolveExceptionSpec(SourceLocation Loc,
                                                const FunctionProtoType *FPT);
  void UpdateExceptionSpec(FunctionDecl *FD,
                           const FunctionProtoType::ExceptionSpecInfo &ESI);
  bool CheckSpecifiedExceptionType(QualType &T, SourceRange Range);
  bool CheckDistantExceptionSpec(QualType T);
  bool CheckEquivalentExceptionSpec(FunctionDecl *Old, FunctionDecl *New);
  bool CheckEquivalentExceptionSpec(
      const FunctionProtoType *Old, SourceLocation OldLoc,
      const FunctionProtoType *New, SourceLocation NewLoc);
  bool CheckEquivalentExceptionSpec(
      const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
      const FunctionProtoType *Old, SourceLocation OldLoc,
      const FunctionProtoType *New, SourceLocation NewLoc);
  bool handlerCanCatch(QualType HandlerType, QualType ExceptionType);
  bool CheckExceptionSpecSubset(const PartialDiagnostic &DiagID,
                                const PartialDiagnostic &NestedDiagID,
                                const PartialDiagnostic &NoteID,
                                const PartialDiagnostic &NoThrowDiagID,
                                const FunctionProtoType *Superset,
                                SourceLocation SuperLoc,
                                const FunctionProtoType *Subset,
                                SourceLocation SubLoc);
  bool CheckParamExceptionSpec(const PartialDiagnostic &NestedDiagID,
                               const PartialDiagnostic &NoteID,
                               const FunctionProtoType *Target,
                               SourceLocation TargetLoc,
                               const FunctionProtoType *Source,
                               SourceLocation SourceLoc);

  TypeResult ActOnTypeName(Scope *S, Declarator &D);

  /// The parser has parsed the context-sensitive type 'instancetype'
  /// in an Objective-C message declaration. Return the appropriate type.
  ParsedType ActOnObjCInstanceType(SourceLocation Loc);

  /// Abstract class used to diagnose incomplete types.
  struct TypeDiagnoser {
    TypeDiagnoser() {}

    virtual void diagnose(Sema &S, SourceLocation Loc, QualType T) = 0;
    virtual ~TypeDiagnoser() {}
  };

  static int getPrintable(int I) { return I; }
  static unsigned getPrintable(unsigned I) { return I; }
  static bool getPrintable(bool B) { return B; }
  static const char * getPrintable(const char *S) { return S; }
  static StringRef getPrintable(StringRef S) { return S; }
  static const std::string &getPrintable(const std::string &S) { return S; }
  static const IdentifierInfo *getPrintable(const IdentifierInfo *II) {
    return II;
  }
  static DeclarationName getPrintable(DeclarationName N) { return N; }
  static QualType getPrintable(QualType T) { return T; }
  static SourceRange getPrintable(SourceRange R) { return R; }
  static SourceRange getPrintable(SourceLocation L) { return L; }
  static SourceRange getPrintable(const Expr *E) { return E->getSourceRange(); }
  static SourceRange getPrintable(TypeLoc TL) { return TL.getSourceRange();}

  template <typename... Ts> class BoundTypeDiagnoser : public TypeDiagnoser {
  protected:
    unsigned DiagID;
    std::tuple<const Ts &...> Args;

    template <std::size_t... Is>
    void emit(const SemaDiagnosticBuilder &DB,
              std::index_sequence<Is...>) const {
      // Apply all tuple elements to the builder in order.
      bool Dummy[] = {false, (DB << getPrintable(std::get<Is>(Args)))...};
      (void)Dummy;
    }

  public:
    BoundTypeDiagnoser(unsigned DiagID, const Ts &...Args)
        : TypeDiagnoser(), DiagID(DiagID), Args(Args...) {
      assert(DiagID != 0 && "no diagnostic for type diagnoser");
    }

    void diagnose(Sema &S, SourceLocation Loc, QualType T) override {
      const SemaDiagnosticBuilder &DB = S.Diag(Loc, DiagID);
      emit(DB, std::index_sequence_for<Ts...>());
      DB << T;
    }
  };

  /// Do a check to make sure \p Name looks like a legal argument for the
  /// swift_name attribute applied to decl \p D.  Raise a diagnostic if the name
  /// is invalid for the given declaration.
  ///
  /// \p AL is used to provide caret diagnostics in case of a malformed name.
  ///
  /// \returns true if the name is a valid swift name for \p D, false otherwise.
  bool DiagnoseSwiftName(Decl *D, StringRef Name, SourceLocation Loc,
                         const ParsedAttr &AL, bool IsAsync);

  /// A derivative of BoundTypeDiagnoser for which the diagnostic's type
  /// parameter is preceded by a 0/1 enum that is 1 if the type is sizeless.
  /// For example, a diagnostic with no other parameters would generally have
  /// the form "...%select{incomplete|sizeless}0 type %1...".
  template <typename... Ts>
  class SizelessTypeDiagnoser : public BoundTypeDiagnoser<Ts...> {
  public:
    SizelessTypeDiagnoser(unsigned DiagID, const Ts &... Args)
        : BoundTypeDiagnoser<Ts...>(DiagID, Args...) {}

    void diagnose(Sema &S, SourceLocation Loc, QualType T) override {
      const SemaDiagnosticBuilder &DB = S.Diag(Loc, this->DiagID);
      this->emit(DB, std::index_sequence_for<Ts...>());
      DB << T->isSizelessType() << T;
    }
  };

  enum class CompleteTypeKind {
    /// Apply the normal rules for complete types.  In particular,
    /// treat all sizeless types as incomplete.
    Normal,

    /// Relax the normal rules for complete types so that they include
    /// sizeless built-in types.
    AcceptSizeless,

    // FIXME: Eventually we should flip the default to Normal and opt in
    // to AcceptSizeless rather than opt out of it.
    Default = AcceptSizeless
  };

private:
  /// Methods for marking which expressions involve dereferencing a pointer
  /// marked with the 'noderef' attribute. Expressions are checked bottom up as
  /// they are parsed, meaning that a noderef pointer may not be accessed. For
  /// example, in `&*p` where `p` is a noderef pointer, we will first parse the
  /// `*p`, but need to check that `address of` is called on it. This requires
  /// keeping a container of all pending expressions and checking if the address
  /// of them are eventually taken.
  void CheckSubscriptAccessOfNoDeref(const ArraySubscriptExpr *E);
  void CheckAddressOfNoDeref(const Expr *E);
  void CheckMemberAccessOfNoDeref(const MemberExpr *E);

  bool RequireCompleteTypeImpl(SourceLocation Loc, QualType T,
                               CompleteTypeKind Kind, TypeDiagnoser *Diagnoser);

  struct ModuleScope {
    SourceLocation BeginLoc;
    clang::Module *Module = nullptr;
    bool ModuleInterface = false;
    bool ImplicitGlobalModuleFragment = false;
    VisibleModuleSet OuterVisibleModules;
  };
  /// The modules we're currently parsing.
  llvm::SmallVector<ModuleScope, 16> ModuleScopes;

  /// Namespace definitions that we will export when they finish.
  llvm::SmallPtrSet<const NamespaceDecl*, 8> DeferredExportedNamespaces;

  /// Get the module whose scope we are currently within.
  Module *getCurrentModule() const {
    return ModuleScopes.empty() ? nullptr : ModuleScopes.back().Module;
  }

  /// Helper function to judge if we are in module purview.
  /// Return false if we are not in a module.
  bool isCurrentModulePurview() const {
    return getCurrentModule() ? getCurrentModule()->isModulePurview() : false;
  }

  /// Enter the scope of the global module.
  Module *PushGlobalModuleFragment(SourceLocation BeginLoc, bool IsImplicit);
  /// Leave the scope of the global module.
  void PopGlobalModuleFragment();

  VisibleModuleSet VisibleModules;

public:
  /// Get the module owning an entity.
  Module *getOwningModule(const Decl *Entity) {
    return Entity->getOwningModule();
  }

  /// Make a merged definition of an existing hidden definition \p ND
  /// visible at the specified location.
  void makeMergedDefinitionVisible(NamedDecl *ND);

  bool isModuleVisible(const Module *M, bool ModulePrivate = false);

  // When loading a non-modular PCH files, this is used to restore module
  // visibility.
  void makeModuleVisible(Module *Mod, SourceLocation ImportLoc) {
    VisibleModules.setVisible(Mod, ImportLoc);
  }

  /// Determine whether a declaration is visible to name lookup.
  bool isVisible(const NamedDecl *D) {
    return D->isUnconditionallyVisible() || isVisibleSlow(D);
  }

  /// Determine whether any declaration of an entity is visible.
  bool
  hasVisibleDeclaration(const NamedDecl *D,
                        llvm::SmallVectorImpl<Module *> *Modules = nullptr) {
    return isVisible(D) || hasVisibleDeclarationSlow(D, Modules);
  }
  bool hasVisibleDeclarationSlow(const NamedDecl *D,
                                 llvm::SmallVectorImpl<Module *> *Modules);

  bool hasVisibleMergedDefinition(NamedDecl *Def);
  bool hasMergedDefinitionInCurrentModule(NamedDecl *Def);

  /// Determine if \p D and \p Suggested have a structurally compatible
  /// layout as described in C11 6.2.7/1.
  bool hasStructuralCompatLayout(Decl *D, Decl *Suggested);

  /// Determine if \p D has a visible definition. If not, suggest a declaration
  /// that should be made visible to expose the definition.
  bool hasVisibleDefinition(NamedDecl *D, NamedDecl **Suggested,
                            bool OnlyNeedComplete = false);
  bool hasVisibleDefinition(const NamedDecl *D) {
    NamedDecl *Hidden;
    return hasVisibleDefinition(const_cast<NamedDecl*>(D), &Hidden);
  }

  /// Determine if the template parameter \p D has a visible default argument.
  bool
  hasVisibleDefaultArgument(const NamedDecl *D,
                            llvm::SmallVectorImpl<Module *> *Modules = nullptr);

  /// Determine if there is a visible declaration of \p D that is an explicit
  /// specialization declaration for a specialization of a template. (For a
  /// member specialization, use hasVisibleMemberSpecialization.)
  bool hasVisibleExplicitSpecialization(
      const NamedDecl *D, llvm::SmallVectorImpl<Module *> *Modules = nullptr);

  /// Determine if there is a visible declaration of \p D that is a member
  /// specialization declaration (as opposed to an instantiated declaration).
  bool hasVisibleMemberSpecialization(
      const NamedDecl *D, llvm::SmallVectorImpl<Module *> *Modules = nullptr);

  /// Determine if \p A and \p B are equivalent internal linkage declarations
  /// from different modules, and thus an ambiguity error can be downgraded to
  /// an extension warning.
  bool isEquivalentInternalLinkageDeclaration(const NamedDecl *A,
                                              const NamedDecl *B);
  void diagnoseEquivalentInternalLinkageDeclarations(
      SourceLocation Loc, const NamedDecl *D,
      ArrayRef<const NamedDecl *> Equiv);

  bool isUsualDeallocationFunction(const CXXMethodDecl *FD);

  bool isCompleteType(SourceLocation Loc, QualType T,
                      CompleteTypeKind Kind = CompleteTypeKind::Default) {
    return !RequireCompleteTypeImpl(Loc, T, Kind, nullptr);
  }
  bool RequireCompleteType(SourceLocation Loc, QualType T,
                           CompleteTypeKind Kind, TypeDiagnoser &Diagnoser);
  bool RequireCompleteType(SourceLocation Loc, QualType T,
                           CompleteTypeKind Kind, unsigned DiagID);

  bool RequireCompleteType(SourceLocation Loc, QualType T,
                           TypeDiagnoser &Diagnoser) {
    return RequireCompleteType(Loc, T, CompleteTypeKind::Default, Diagnoser);
  }
  bool RequireCompleteType(SourceLocation Loc, QualType T, unsigned DiagID) {
    return RequireCompleteType(Loc, T, CompleteTypeKind::Default, DiagID);
  }

  template <typename... Ts>
  bool RequireCompleteType(SourceLocation Loc, QualType T, unsigned DiagID,
                           const Ts &...Args) {
    BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
    return RequireCompleteType(Loc, T, Diagnoser);
  }

  template <typename... Ts>
  bool RequireCompleteSizedType(SourceLocation Loc, QualType T, unsigned DiagID,
                                const Ts &... Args) {
    SizelessTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
    return RequireCompleteType(Loc, T, CompleteTypeKind::Normal, Diagnoser);
  }

  /// Get the type of expression E, triggering instantiation to complete the
  /// type if necessary -- that is, if the expression refers to a templated
  /// static data member of incomplete array type.
  ///
  /// May still return an incomplete type if instantiation was not possible or
  /// if the type is incomplete for a different reason. Use
  /// RequireCompleteExprType instead if a diagnostic is expected for an
  /// incomplete expression type.
  QualType getCompletedType(Expr *E);

  void completeExprArrayBound(Expr *E);
  bool RequireCompleteExprType(Expr *E, CompleteTypeKind Kind,
                               TypeDiagnoser &Diagnoser);
  bool RequireCompleteExprType(Expr *E, unsigned DiagID);

  template <typename... Ts>
  bool RequireCompleteExprType(Expr *E, unsigned DiagID, const Ts &...Args) {
    BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
    return RequireCompleteExprType(E, CompleteTypeKind::Default, Diagnoser);
  }

  template <typename... Ts>
  bool RequireCompleteSizedExprType(Expr *E, unsigned DiagID,
                                    const Ts &... Args) {
    SizelessTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
    return RequireCompleteExprType(E, CompleteTypeKind::Normal, Diagnoser);
  }

  bool RequireLiteralType(SourceLocation Loc, QualType T,
                          TypeDiagnoser &Diagnoser);
  bool RequireLiteralType(SourceLocation Loc, QualType T, unsigned DiagID);

  template <typename... Ts>
  bool RequireLiteralType(SourceLocation Loc, QualType T, unsigned DiagID,
                          const Ts &...Args) {
    BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
    return RequireLiteralType(Loc, T, Diagnoser);
  }

  QualType getElaboratedType(ElaboratedTypeKeyword Keyword,
                             const CXXScopeSpec &SS, QualType T,
                             TagDecl *OwnedTagDecl = nullptr);

  // Returns the underlying type of a decltype with the given expression.
  QualType getDecltypeForExpr(Expr *E);

  QualType BuildTypeofExprType(Expr *E);
  /// If AsUnevaluated is false, E is treated as though it were an evaluated
  /// context, such as when building a type for decltype(auto).
  QualType BuildDecltypeType(Expr *E, bool AsUnevaluated = true);
  QualType BuildUnaryTransformType(QualType BaseType,
                                   UnaryTransformType::UTTKind UKind,
                                   SourceLocation Loc);

  //===--------------------------------------------------------------------===//
  // Symbol table / Decl tracking callbacks: SemaDecl.cpp.
  //

  struct SkipBodyInfo {
    SkipBodyInfo()
        : ShouldSkip(false), CheckSameAsPrevious(false), Previous(nullptr),
          New(nullptr) {}
    bool ShouldSkip;
    bool CheckSameAsPrevious;
    NamedDecl *Previous;
    NamedDecl *New;
  };

  DeclGroupPtrTy ConvertDeclToDeclGroup(Decl *Ptr, Decl *OwnedType = nullptr);

  void DiagnoseUseOfUnimplementedSelectors();

  bool isSimpleTypeSpecifier(tok::TokenKind Kind) const;

  ParsedType getTypeName(const IdentifierInfo &II, SourceLocation NameLoc,
                         Scope *S, CXXScopeSpec *SS = nullptr,
                         bool isClassName = false, bool HasTrailingDot = false,
                         ParsedType ObjectType = nullptr,
                         bool IsCtorOrDtorName = false,
                         bool WantNontrivialTypeSourceInfo = false,
                         bool IsClassTemplateDeductionContext = true,
                         IdentifierInfo **CorrectedII = nullptr);
  TypeSpecifierType isTagName(IdentifierInfo &II, Scope *S);
  bool isMicrosoftMissingTypename(const CXXScopeSpec *SS, Scope *S);
  void DiagnoseUnknownTypeName(IdentifierInfo *&II,
                               SourceLocation IILoc,
                               Scope *S,
                               CXXScopeSpec *SS,
                               ParsedType &SuggestedType,
                               bool IsTemplateName = false);

  /// Attempt to behave like MSVC in situations where lookup of an unqualified
  /// type name has failed in a dependent context. In these situations, we
  /// automatically form a DependentTypeName that will retry lookup in a related
  /// scope during instantiation.
  ParsedType ActOnMSVCUnknownTypeName(const IdentifierInfo &II,
                                      SourceLocation NameLoc,
                                      bool IsTemplateTypeArg);

  /// Describes the result of the name lookup and resolution performed
  /// by \c ClassifyName().
  enum NameClassificationKind {
    /// This name is not a type or template in this context, but might be
    /// something else.
    NC_Unknown,
    /// Classification failed; an error has been produced.
    NC_Error,
    /// The name has been typo-corrected to a keyword.
    NC_Keyword,
    /// The name was classified as a type.
    NC_Type,
    /// The name was classified as a specific non-type, non-template
    /// declaration. ActOnNameClassifiedAsNonType should be called to
    /// convert the declaration to an expression.
    NC_NonType,
    /// The name was classified as an ADL-only function name.
    /// ActOnNameClassifiedAsUndeclaredNonType should be called to convert the
    /// result to an expression.
    NC_UndeclaredNonType,
    /// The name denotes a member of a dependent type that could not be
    /// resolved. ActOnNameClassifiedAsDependentNonType should be called to
    /// convert the result to an expression.
    NC_DependentNonType,
    /// The name was classified as an overload set, and an expression
    /// representing that overload set has been formed.
    /// ActOnNameClassifiedAsOverloadSet should be called to form a suitable
    /// expression referencing the overload set.
    NC_OverloadSet,
    /// The name was classified as a template whose specializations are types.
    NC_TypeTemplate,
    /// The name was classified as a variable template name.
    NC_VarTemplate,
    /// The name was classified as a function template name.
    NC_FunctionTemplate,
    /// The name was classified as an ADL-only function template name.
    NC_UndeclaredTemplate,
    /// The name was classified as a concept name.
    NC_Concept,
  };

  class NameClassification {
    NameClassificationKind Kind;
    union {
      ExprResult Expr;
      NamedDecl *NonTypeDecl;
      TemplateName Template;
      ParsedType Type;
    };

    explicit NameClassification(NameClassificationKind Kind) : Kind(Kind) {}

  public:
    NameClassification(ParsedType Type) : Kind(NC_Type), Type(Type) {}

    NameClassification(const IdentifierInfo *Keyword) : Kind(NC_Keyword) {}

    static NameClassification Error() {
      return NameClassification(NC_Error);
    }

    static NameClassification Unknown() {
      return NameClassification(NC_Unknown);
    }

    static NameClassification OverloadSet(ExprResult E) {
      NameClassification Result(NC_OverloadSet);
      Result.Expr = E;
      return Result;
    }

    static NameClassification NonType(NamedDecl *D) {
      NameClassification Result(NC_NonType);
      Result.NonTypeDecl = D;
      return Result;
    }

    static NameClassification UndeclaredNonType() {
      return NameClassification(NC_UndeclaredNonType);
    }

    static NameClassification DependentNonType() {
      return NameClassification(NC_DependentNonType);
    }

    static NameClassification TypeTemplate(TemplateName Name) {
      NameClassification Result(NC_TypeTemplate);
      Result.Template = Name;
      return Result;
    }

    static NameClassification VarTemplate(TemplateName Name) {
      NameClassification Result(NC_VarTemplate);
      Result.Template = Name;
      return Result;
    }

    static NameClassification FunctionTemplate(TemplateName Name) {
      NameClassification Result(NC_FunctionTemplate);
      Result.Template = Name;
      return Result;
    }

    static NameClassification Concept(TemplateName Name) {
      NameClassification Result(NC_Concept);
      Result.Template = Name;
      return Result;
    }

    static NameClassification UndeclaredTemplate(TemplateName Name) {
      NameClassification Result(NC_UndeclaredTemplate);
      Result.Template = Name;
      return Result;
    }

    NameClassificationKind getKind() const { return Kind; }

    ExprResult getExpression() const {
      assert(Kind == NC_OverloadSet);
      return Expr;
    }

    ParsedType getType() const {
      assert(Kind == NC_Type);
      return Type;
    }

    NamedDecl *getNonTypeDecl() const {
      assert(Kind == NC_NonType);
      return NonTypeDecl;
    }

    TemplateName getTemplateName() const {
      assert(Kind == NC_TypeTemplate || Kind == NC_FunctionTemplate ||
             Kind == NC_VarTemplate || Kind == NC_Concept ||
             Kind == NC_UndeclaredTemplate);
      return Template;
    }

    TemplateNameKind getTemplateNameKind() const {
      switch (Kind) {
      case NC_TypeTemplate:
        return TNK_Type_template;
      case NC_FunctionTemplate:
        return TNK_Function_template;
      case NC_VarTemplate:
        return TNK_Var_template;
      case NC_Concept:
        return TNK_Concept_template;
      case NC_UndeclaredTemplate:
        return TNK_Undeclared_template;
      default:
        llvm_unreachable("unsupported name classification.");
      }
    }
  };

  /// Perform name lookup on the given name, classifying it based on
  /// the results of name lookup and the following token.
  ///
  /// This routine is used by the parser to resolve identifiers and help direct
  /// parsing. When the identifier cannot be found, this routine will attempt
  /// to correct the typo and classify based on the resulting name.
  ///
  /// \param S The scope in which we're performing name lookup.
  ///
  /// \param SS The nested-name-specifier that precedes the name.
  ///
  /// \param Name The identifier. If typo correction finds an alternative name,
  /// this pointer parameter will be updated accordingly.
  ///
  /// \param NameLoc The location of the identifier.
  ///
  /// \param NextToken The token following the identifier. Used to help
  /// disambiguate the name.
  ///
  /// \param CCC The correction callback, if typo correction is desired.
  NameClassification ClassifyName(Scope *S, CXXScopeSpec &SS,
                                  IdentifierInfo *&Name, SourceLocation NameLoc,
                                  const Token &NextToken,
                                  CorrectionCandidateCallback *CCC = nullptr);

  /// Act on the result of classifying a name as an undeclared (ADL-only)
  /// non-type declaration.
  ExprResult ActOnNameClassifiedAsUndeclaredNonType(IdentifierInfo *Name,
                                                    SourceLocation NameLoc);
  /// Act on the result of classifying a name as an undeclared member of a
  /// dependent base class.
  ExprResult ActOnNameClassifiedAsDependentNonType(const CXXScopeSpec &SS,
                                                   IdentifierInfo *Name,
                                                   SourceLocation NameLoc,
                                                   bool IsAddressOfOperand);
  /// Act on the result of classifying a name as a specific non-type
  /// declaration.
  ExprResult ActOnNameClassifiedAsNonType(Scope *S, const CXXScopeSpec &SS,
                                          NamedDecl *Found,
                                          SourceLocation NameLoc,
                                          const Token &NextToken);
  /// Act on the result of classifying a name as an overload set.
  ExprResult ActOnNameClassifiedAsOverloadSet(Scope *S, Expr *OverloadSet);

  /// Describes the detailed kind of a template name. Used in diagnostics.
  enum class TemplateNameKindForDiagnostics {
    ClassTemplate,
    FunctionTemplate,
    VarTemplate,
    AliasTemplate,
    TemplateTemplateParam,
    Concept,
    DependentTemplate
  };
  TemplateNameKindForDiagnostics
  getTemplateNameKindForDiagnostics(TemplateName Name);

  /// Determine whether it's plausible that E was intended to be a
  /// template-name.
  bool mightBeIntendedToBeTemplateName(ExprResult E, bool &Dependent) {
    if (!getLangOpts().CPlusPlus || E.isInvalid())
      return false;
    Dependent = false;
    if (auto *DRE = dyn_cast<DeclRefExpr>(E.get()))
      return !DRE->hasExplicitTemplateArgs();
    if (auto *ME = dyn_cast<MemberExpr>(E.get()))
      return !ME->hasExplicitTemplateArgs();
    Dependent = true;
    if (auto *DSDRE = dyn_cast<DependentScopeDeclRefExpr>(E.get()))
      return !DSDRE->hasExplicitTemplateArgs();
    if (auto *DSME = dyn_cast<CXXDependentScopeMemberExpr>(E.get()))
      return !DSME->hasExplicitTemplateArgs();
    // Any additional cases recognized here should also be handled by
    // diagnoseExprIntendedAsTemplateName.
    return false;
  }
  void diagnoseExprIntendedAsTemplateName(Scope *S, ExprResult TemplateName,
                                          SourceLocation Less,
                                          SourceLocation Greater);

  void warnOnReservedIdentifier(const NamedDecl *D);

  Decl *ActOnDeclarator(Scope *S, Declarator &D);

  NamedDecl *HandleDeclarator(Scope *S, Declarator &D,
                              MultiTemplateParamsArg TemplateParameterLists);
  bool tryToFixVariablyModifiedVarType(TypeSourceInfo *&TInfo,
                                       QualType &T, SourceLocation Loc,
                                       unsigned FailedFoldDiagID);
  void RegisterLocallyScopedExternCDecl(NamedDecl *ND, Scope *S);
  bool DiagnoseClassNameShadow(DeclContext *DC, DeclarationNameInfo Info);
  bool diagnoseQualifiedDeclaration(CXXScopeSpec &SS, DeclContext *DC,
                                    DeclarationName Name, SourceLocation Loc,
                                    bool IsTemplateId);
  void
  diagnoseIgnoredQualifiers(unsigned DiagID, unsigned Quals,
                            SourceLocation FallbackLoc,
                            SourceLocation ConstQualLoc = SourceLocation(),
                            SourceLocation VolatileQualLoc = SourceLocation(),
                            SourceLocation RestrictQualLoc = SourceLocation(),
                            SourceLocation AtomicQualLoc = SourceLocation(),
                            SourceLocation UnalignedQualLoc = SourceLocation());

  static bool adjustContextForLocalExternDecl(DeclContext *&DC);
  void DiagnoseFunctionSpecifiers(const DeclSpec &DS);
  NamedDecl *getShadowedDeclaration(const TypedefNameDecl *D,
                                    const LookupResult &R);
  NamedDecl *getShadowedDeclaration(const VarDecl *D, const LookupResult &R);
  NamedDecl *getShadowedDeclaration(const BindingDecl *D,
                                    const LookupResult &R);
  void CheckShadow(NamedDecl *D, NamedDecl *ShadowedDecl,
                   const LookupResult &R);
  void CheckShadow(Scope *S, VarDecl *D);

  /// Warn if 'E', which is an expression that is about to be modified, refers
  /// to a shadowing declaration.
  void CheckShadowingDeclModification(Expr *E, SourceLocation Loc);

  void DiagnoseShadowingLambdaDecls(const sema::LambdaScopeInfo *LSI);

private:
  /// Map of current shadowing declarations to shadowed declarations. Warn if
  /// it looks like the user is trying to modify the shadowing declaration.
  llvm::DenseMap<const NamedDecl *, const NamedDecl *> ShadowingDecls;

public:
  void CheckCastAlign(Expr *Op, QualType T, SourceRange TRange);
  void handleTagNumbering(const TagDecl *Tag, Scope *TagScope);
  void setTagNameForLinkagePurposes(TagDecl *TagFromDeclSpec,
                                    TypedefNameDecl *NewTD);
  void CheckTypedefForVariablyModifiedType(Scope *S, TypedefNameDecl *D);
  NamedDecl* ActOnTypedefDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                    TypeSourceInfo *TInfo,
                                    LookupResult &Previous);
  NamedDecl* ActOnTypedefNameDecl(Scope* S, DeclContext* DC, TypedefNameDecl *D,
                                  LookupResult &Previous, bool &Redeclaration);
  NamedDecl *ActOnVariableDeclarator(Scope *S, Declarator &D, DeclContext *DC,
                                     TypeSourceInfo *TInfo,
                                     LookupResult &Previous,
                                     MultiTemplateParamsArg TemplateParamLists,
                                     bool &AddToScope,
                                     ArrayRef<BindingDecl *> Bindings = None);
  NamedDecl *
  ActOnDecompositionDeclarator(Scope *S, Declarator &D,
                               MultiTemplateParamsArg TemplateParamLists);
  // Returns true if the variable declaration is a redeclaration
  bool CheckVariableDeclaration(VarDecl *NewVD, LookupResult &Previous);
  void CheckVariableDeclarationType(VarDecl *NewVD);
  bool DeduceVariableDeclarationType(VarDecl *VDecl, bool DirectInit,
                                     Expr *Init);
  void CheckCompleteVariableDeclaration(VarDecl *VD);
  void CheckCompleteDecompositionDeclaration(DecompositionDecl *DD);
  void MaybeSuggestAddingStaticToDecl(const FunctionDecl *D);

  NamedDecl* ActOnFunctionDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                     TypeSourceInfo *TInfo,
                                     LookupResult &Previous,
                                     MultiTemplateParamsArg TemplateParamLists,
                                     bool &AddToScope);
  bool AddOverriddenMethods(CXXRecordDecl *DC, CXXMethodDecl *MD);

  enum class CheckConstexprKind {
    /// Diagnose issues that are non-constant or that are extensions.
    Diagnose,
    /// Identify whether this function satisfies the formal rules for constexpr
    /// functions in the current lanugage mode (with no extensions).
    CheckValid
  };

  bool CheckConstexprFunctionDefinition(const FunctionDecl *FD,
                                        CheckConstexprKind Kind);

  void DiagnoseHiddenVirtualMethods(CXXMethodDecl *MD);
  void FindHiddenVirtualMethods(CXXMethodDecl *MD,
                          SmallVectorImpl<CXXMethodDecl*> &OverloadedMethods);
  void NoteHiddenVirtualMethods(CXXMethodDecl *MD,
                          SmallVectorImpl<CXXMethodDecl*> &OverloadedMethods);
  // Returns true if the function declaration is a redeclaration
  bool CheckFunctionDeclaration(Scope *S,
                                FunctionDecl *NewFD, LookupResult &Previous,
                                bool IsMemberSpecialization);
  bool shouldLinkDependentDeclWithPrevious(Decl *D, Decl *OldDecl);
  bool canFullyTypeCheckRedeclaration(ValueDecl *NewD, ValueDecl *OldD,
                                      QualType NewT, QualType OldT);
  void CheckMain(FunctionDecl *FD, const DeclSpec &D);
  void CheckMSVCRTEntryPoint(FunctionDecl *FD);
  Attr *getImplicitCodeSegOrSectionAttrForFunction(const FunctionDecl *FD,
                                                   bool IsDefinition);
  void CheckFunctionOrTemplateParamDeclarator(Scope *S, Declarator &D);
  Decl *ActOnParamDeclarator(Scope *S, Declarator &D);
  ParmVarDecl *BuildParmVarDeclForTypedef(DeclContext *DC,
                                          SourceLocation Loc,
                                          QualType T);
  ParmVarDecl *CheckParameter(DeclContext *DC, SourceLocation StartLoc,
                              SourceLocation NameLoc, IdentifierInfo *Name,
                              QualType T, TypeSourceInfo *TSInfo,
                              StorageClass SC);
  void ActOnParamDefaultArgument(Decl *param,
                                 SourceLocation EqualLoc,
                                 Expr *defarg);
  void ActOnParamUnparsedDefaultArgument(Decl *param, SourceLocation EqualLoc,
                                         SourceLocation ArgLoc);
  void ActOnParamDefaultArgumentError(Decl *param, SourceLocation EqualLoc);
  ExprResult ConvertParamDefaultArgument(ParmVarDecl *Param, Expr *DefaultArg,
                                         SourceLocation EqualLoc);
  void SetParamDefaultArgument(ParmVarDecl *Param, Expr *DefaultArg,
                               SourceLocation EqualLoc);

  // Contexts where using non-trivial C union types can be disallowed. This is
  // passed to err_non_trivial_c_union_in_invalid_context.
  enum NonTrivialCUnionContext {
    // Function parameter.
    NTCUC_FunctionParam,
    // Function return.
    NTCUC_FunctionReturn,
    // Default-initialized object.
    NTCUC_DefaultInitializedObject,
    // Variable with automatic storage duration.
    NTCUC_AutoVar,
    // Initializer expression that might copy from another object.
    NTCUC_CopyInit,
    // Assignment.
    NTCUC_Assignment,
    // Compound literal.
    NTCUC_CompoundLiteral,
    // Block capture.
    NTCUC_BlockCapture,
    // lvalue-to-rvalue conversion of volatile type.
    NTCUC_LValueToRValueVolatile,
  };

  /// Emit diagnostics if the initializer or any of its explicit or
  /// implicitly-generated subexpressions require copying or
  /// default-initializing a type that is or contains a C union type that is
  /// non-trivial to copy or default-initialize.
  void checkNonTrivialCUnionInInitializer(const Expr *Init, SourceLocation Loc);

  // These flags are passed to checkNonTrivialCUnion.
  enum NonTrivialCUnionKind {
    NTCUK_Init = 0x1,
    NTCUK_Destruct = 0x2,
    NTCUK_Copy = 0x4,
  };

  /// Emit diagnostics if a non-trivial C union type or a struct that contains
  /// a non-trivial C union is used in an invalid context.
  void checkNonTrivialCUnion(QualType QT, SourceLocation Loc,
                             NonTrivialCUnionContext UseContext,
                             unsigned NonTrivialKind);

  void AddInitializerToDecl(Decl *dcl, Expr *init, bool DirectInit);
  void ActOnUninitializedDecl(Decl *dcl);
  void ActOnInitializerError(Decl *Dcl);

  void ActOnPureSpecifier(Decl *D, SourceLocation PureSpecLoc);
  void ActOnCXXForRangeDecl(Decl *D);
  StmtResult ActOnCXXForRangeIdentifier(Scope *S, SourceLocation IdentLoc,
                                        IdentifierInfo *Ident,
                                        ParsedAttributes &Attrs,
                                        SourceLocation AttrEnd);
  void SetDeclDeleted(Decl *dcl, SourceLocation DelLoc);
  void SetDeclDefaulted(Decl *dcl, SourceLocation DefaultLoc);
  void CheckStaticLocalForDllExport(VarDecl *VD);
  void FinalizeDeclaration(Decl *D);
  DeclGroupPtrTy FinalizeDeclaratorGroup(Scope *S, const DeclSpec &DS,
                                         ArrayRef<Decl *> Group);
  DeclGroupPtrTy BuildDeclaratorGroup(MutableArrayRef<Decl *> Group);

  /// Should be called on all declarations that might have attached
  /// documentation comments.
  void ActOnDocumentableDecl(Decl *D);
  void ActOnDocumentableDecls(ArrayRef<Decl *> Group);

  void ActOnFinishKNRParamDeclarations(Scope *S, Declarator &D,
                                       SourceLocation LocAfterDecls);
  void CheckForFunctionRedefinition(
      FunctionDecl *FD, const FunctionDecl *EffectiveDefinition = nullptr,
      SkipBodyInfo *SkipBody = nullptr);
  Decl *ActOnStartOfFunctionDef(Scope *S, Declarator &D,
                                MultiTemplateParamsArg TemplateParamLists,
                                SkipBodyInfo *SkipBody = nullptr);
  Decl *ActOnStartOfFunctionDef(Scope *S, Decl *D,
                                SkipBodyInfo *SkipBody = nullptr);
  void ActOnStartTrailingRequiresClause(Scope *S, Declarator &D);
  ExprResult ActOnFinishTrailingRequiresClause(ExprResult ConstraintExpr);
  ExprResult ActOnRequiresClause(ExprResult ConstraintExpr);
  void ActOnStartOfObjCMethodDef(Scope *S, Decl *D);
  bool isObjCMethodDecl(Decl *D) {
    return D && isa<ObjCMethodDecl>(D);
  }

  /// Determine whether we can delay parsing the body of a function or
  /// function template until it is used, assuming we don't care about emitting
  /// code for that function.
  ///
  /// This will be \c false if we may need the body of the function in the
  /// middle of parsing an expression (where it's impractical to switch to
  /// parsing a different function), for instance, if it's constexpr in C++11
  /// or has an 'auto' return type in C++14. These cases are essentially bugs.
  bool canDelayFunctionBody(const Declarator &D);

  /// Determine whether we can skip parsing the body of a function
  /// definition, assuming we don't care about analyzing its body or emitting
  /// code for that function.
  ///
  /// This will be \c false only if we may need the body of the function in
  /// order to parse the rest of the program (for instance, if it is
  /// \c constexpr in C++11 or has an 'auto' return type in C++14).
  bool canSkipFunctionBody(Decl *D);

  void computeNRVO(Stmt *Body, sema::FunctionScopeInfo *Scope);
  Decl *ActOnFinishFunctionBody(Decl *Decl, Stmt *Body);
  Decl *ActOnFinishFunctionBody(Decl *Decl, Stmt *Body, bool IsInstantiation);
  Decl *ActOnSkippedFunctionBody(Decl *Decl);
  void ActOnFinishInlineFunctionDef(FunctionDecl *D);

  /// ActOnFinishDelayedAttribute - Invoked when we have finished parsing an
  /// attribute for which parsing is delayed.
  void ActOnFinishDelayedAttribute(Scope *S, Decl *D, ParsedAttributes &Attrs);

  /// Diagnose any unused parameters in the given sequence of
  /// ParmVarDecl pointers.
  void DiagnoseUnusedParameters(ArrayRef<ParmVarDecl *> Parameters);

  /// Diagnose whether the size of parameters or return value of a
  /// function or obj-c method definition is pass-by-value and larger than a
  /// specified threshold.
  void
  DiagnoseSizeOfParametersAndReturnValue(ArrayRef<ParmVarDecl *> Parameters,
                                         QualType ReturnTy, NamedDecl *D);

  void DiagnoseInvalidJumps(Stmt *Body);
  Decl *ActOnFileScopeAsmDecl(Expr *expr,
                              SourceLocation AsmLoc,
                              SourceLocation RParenLoc);

  /// Handle a C++11 empty-declaration and attribute-declaration.
  Decl *ActOnEmptyDeclaration(Scope *S, const ParsedAttributesView &AttrList,
                              SourceLocation SemiLoc);

  enum class ModuleDeclKind {
    Interface,      ///< 'export module X;'
    Implementation, ///< 'module X;'
  };

  /// The parser has processed a module-declaration that begins the definition
  /// of a module interface or implementation.
  DeclGroupPtrTy ActOnModuleDecl(SourceLocation StartLoc,
                                 SourceLocation ModuleLoc, ModuleDeclKind MDK,
                                 ModuleIdPath Path, bool IsFirstDecl);

  /// The parser has processed a global-module-fragment declaration that begins
  /// the definition of the global module fragment of the current module unit.
  /// \param ModuleLoc The location of the 'module' keyword.
  DeclGroupPtrTy ActOnGlobalModuleFragmentDecl(SourceLocation ModuleLoc);

  /// The parser has processed a private-module-fragment declaration that begins
  /// the definition of the private module fragment of the current module unit.
  /// \param ModuleLoc The location of the 'module' keyword.
  /// \param PrivateLoc The location of the 'private' keyword.
  DeclGroupPtrTy ActOnPrivateModuleFragmentDecl(SourceLocation ModuleLoc,
                                                SourceLocation PrivateLoc);

  /// The parser has processed a module import declaration.
  ///
  /// \param StartLoc The location of the first token in the declaration. This
  ///        could be the location of an '@', 'export', or 'import'.
  /// \param ExportLoc The location of the 'export' keyword, if any.
  /// \param ImportLoc The location of the 'import' keyword.
  /// \param Path The module access path.
  DeclResult ActOnModuleImport(SourceLocation StartLoc,
                               SourceLocation ExportLoc,
                               SourceLocation ImportLoc, ModuleIdPath Path);
  DeclResult ActOnModuleImport(SourceLocation StartLoc,
                               SourceLocation ExportLoc,
                               SourceLocation ImportLoc, Module *M,
                               ModuleIdPath Path = {});

  /// The parser has processed a module import translated from a
  /// #include or similar preprocessing directive.
  void ActOnModuleInclude(SourceLocation DirectiveLoc, Module *Mod);
  void BuildModuleInclude(SourceLocation DirectiveLoc, Module *Mod);

  /// The parsed has entered a submodule.
  void ActOnModuleBegin(SourceLocation DirectiveLoc, Module *Mod);
  /// The parser has left a submodule.
  void ActOnModuleEnd(SourceLocation DirectiveLoc, Module *Mod);

  /// Create an implicit import of the given module at the given
  /// source location, for error recovery, if possible.
  ///
  /// This routine is typically used when an entity found by name lookup
  /// is actually hidden within a module that we know about but the user
  /// has forgotten to import.
  void createImplicitModuleImportForErrorRecovery(SourceLocation Loc,
                                                  Module *Mod);

  /// Kinds of missing import. Note, the values of these enumerators correspond
  /// to %select values in diagnostics.
  enum class MissingImportKind {
    Declaration,
    Definition,
    DefaultArgument,
    ExplicitSpecialization,
    PartialSpecialization
  };

  /// Diagnose that the specified declaration needs to be visible but
  /// isn't, and suggest a module import that would resolve the problem.
  void diagnoseMissingImport(SourceLocation Loc, NamedDecl *Decl,
                             MissingImportKind MIK, bool Recover = true);
  void diagnoseMissingImport(SourceLocation Loc, NamedDecl *Decl,
                             SourceLocation DeclLoc, ArrayRef<Module *> Modules,
                             MissingImportKind MIK, bool Recover);

  Decl *ActOnStartExportDecl(Scope *S, SourceLocation ExportLoc,
                             SourceLocation LBraceLoc);
  Decl *ActOnFinishExportDecl(Scope *S, Decl *ExportDecl,
                              SourceLocation RBraceLoc);

  /// We've found a use of a templated declaration that would trigger an
  /// implicit instantiation. Check that any relevant explicit specializations
  /// and partial specializations are visible, and diagnose if not.
  void checkSpecializationVisibility(SourceLocation Loc, NamedDecl *Spec);

  /// Retrieve a suitable printing policy for diagnostics.
  PrintingPolicy getPrintingPolicy() const {
    return getPrintingPolicy(Context, PP);
  }

  /// Retrieve a suitable printing policy for diagnostics.
  static PrintingPolicy getPrintingPolicy(const ASTContext &Ctx,
                                          const Preprocessor &PP);

  /// Scope actions.
  void ActOnPopScope(SourceLocation Loc, Scope *S);
  void ActOnTranslationUnitScope(Scope *S);

  Decl *ParsedFreeStandingDeclSpec(Scope *S, AccessSpecifier AS, DeclSpec &DS,
                                   RecordDecl *&AnonRecord);
  Decl *ParsedFreeStandingDeclSpec(Scope *S, AccessSpecifier AS, DeclSpec &DS,
                                   MultiTemplateParamsArg TemplateParams,
                                   bool IsExplicitInstantiation,
                                   RecordDecl *&AnonRecord);

  Decl *BuildAnonymousStructOrUnion(Scope *S, DeclSpec &DS,
                                    AccessSpecifier AS,
                                    RecordDecl *Record,
                                    const PrintingPolicy &Policy);

  Decl *BuildMicrosoftCAnonymousStruct(Scope *S, DeclSpec &DS,
                                       RecordDecl *Record);

  /// Common ways to introduce type names without a tag for use in diagnostics.
  /// Keep in sync with err_tag_reference_non_tag.
  enum NonTagKind {
    NTK_NonStruct,
    NTK_NonClass,
    NTK_NonUnion,
    NTK_NonEnum,
    NTK_Typedef,
    NTK_TypeAlias,
    NTK_Template,
    NTK_TypeAliasTemplate,
    NTK_TemplateTemplateArgument,
  };

  /// Given a non-tag type declaration, returns an enum useful for indicating
  /// what kind of non-tag type this is.
  NonTagKind getNonTagTypeDeclKind(const Decl *D, TagTypeKind TTK);

  bool isAcceptableTagRedeclaration(const TagDecl *Previous,
                                    TagTypeKind NewTag, bool isDefinition,
                                    SourceLocation NewTagLoc,
                                    const IdentifierInfo *Name);

  enum TagUseKind {
    TUK_Reference,   // Reference to a tag:  'struct foo *X;'
    TUK_Declaration, // Fwd decl of a tag:   'struct foo;'
    TUK_Definition,  // Definition of a tag: 'struct foo { int X; } Y;'
    TUK_Friend       // Friend declaration:  'friend struct foo;'
  };

  Decl *ActOnTag(Scope *S, unsigned TagSpec, TagUseKind TUK,
                 SourceLocation KWLoc, CXXScopeSpec &SS, IdentifierInfo *Name,
                 SourceLocation NameLoc, const ParsedAttributesView &Attr,
                 AccessSpecifier AS, SourceLocation ModulePrivateLoc,
                 MultiTemplateParamsArg TemplateParameterLists, bool &OwnedDecl,
                 bool &IsDependent, SourceLocation ScopedEnumKWLoc,
                 bool ScopedEnumUsesClassTag, TypeResult UnderlyingType,
                 bool IsTypeSpecifier, bool IsTemplateParamOrArg,
                 SkipBodyInfo *SkipBody = nullptr);

  Decl *ActOnTemplatedFriendTag(Scope *S, SourceLocation FriendLoc,
                                unsigned TagSpec, SourceLocation TagLoc,
                                CXXScopeSpec &SS, IdentifierInfo *Name,
                                SourceLocation NameLoc,
                                const ParsedAttributesView &Attr,
                                MultiTemplateParamsArg TempParamLists);

  TypeResult ActOnDependentTag(Scope *S,
                               unsigned TagSpec,
                               TagUseKind TUK,
                               const CXXScopeSpec &SS,
                               IdentifierInfo *Name,
                               SourceLocation TagLoc,
                               SourceLocation NameLoc);

  void ActOnDefs(Scope *S, Decl *TagD, SourceLocation DeclStart,
                 IdentifierInfo *ClassName,
                 SmallVectorImpl<Decl *> &Decls);
  Decl *ActOnField(Scope *S, Decl *TagD, SourceLocation DeclStart,
                   Declarator &D, Expr *BitfieldWidth);

  FieldDecl *HandleField(Scope *S, RecordDecl *TagD, SourceLocation DeclStart,
                         Declarator &D, Expr *BitfieldWidth,
                         InClassInitStyle InitStyle,
                         AccessSpecifier AS);
  MSPropertyDecl *HandleMSProperty(Scope *S, RecordDecl *TagD,
                                   SourceLocation DeclStart, Declarator &D,
                                   Expr *BitfieldWidth,
                                   InClassInitStyle InitStyle,
                                   AccessSpecifier AS,
                                   const ParsedAttr &MSPropertyAttr);

  FieldDecl *CheckFieldDecl(DeclarationName Name, QualType T,
                            TypeSourceInfo *TInfo,
                            RecordDecl *Record, SourceLocation Loc,
                            bool Mutable, Expr *BitfieldWidth,
                            InClassInitStyle InitStyle,
                            SourceLocation TSSL,
                            AccessSpecifier AS, NamedDecl *PrevDecl,
                            Declarator *D = nullptr);

  bool CheckNontrivialField(FieldDecl *FD);
  void DiagnoseNontrivial(const CXXRecordDecl *Record, CXXSpecialMember CSM);

  enum TrivialABIHandling {
    /// The triviality of a method unaffected by "trivial_abi".
    TAH_IgnoreTrivialABI,

    /// The triviality of a method affected by "trivial_abi".
    TAH_ConsiderTrivialABI
  };

  bool SpecialMemberIsTrivial(CXXMethodDecl *MD, CXXSpecialMember CSM,
                              TrivialABIHandling TAH = TAH_IgnoreTrivialABI,
                              bool Diagnose = false);

  /// For a defaulted function, the kind of defaulted function that it is.
  class DefaultedFunctionKind {
    CXXSpecialMember SpecialMember : 8;
    DefaultedComparisonKind Comparison : 8;

  public:
    DefaultedFunctionKind()
        : SpecialMember(CXXInvalid), Comparison(DefaultedComparisonKind::None) {
    }
    DefaultedFunctionKind(CXXSpecialMember CSM)
        : SpecialMember(CSM), Comparison(DefaultedComparisonKind::None) {}
    DefaultedFunctionKind(DefaultedComparisonKind Comp)
        : SpecialMember(CXXInvalid), Comparison(Comp) {}

    bool isSpecialMember() const { return SpecialMember != CXXInvalid; }
    bool isComparison() const {
      return Comparison != DefaultedComparisonKind::None;
    }

    explicit operator bool() const {
      return isSpecialMember() || isComparison();
    }

    CXXSpecialMember asSpecialMember() const { return SpecialMember; }
    DefaultedComparisonKind asComparison() const { return Comparison; }

    /// Get the index of this function kind for use in diagnostics.
    unsigned getDiagnosticIndex() const {
      static_assert(CXXInvalid > CXXDestructor,
                    "invalid should have highest index");
      static_assert((unsigned)DefaultedComparisonKind::None == 0,
                    "none should be equal to zero");
      return SpecialMember + (unsigned)Comparison;
    }
  };

  DefaultedFunctionKind getDefaultedFunctionKind(const FunctionDecl *FD);

  CXXSpecialMember getSpecialMember(const CXXMethodDecl *MD) {
    return getDefaultedFunctionKind(MD).asSpecialMember();
  }
  DefaultedComparisonKind getDefaultedComparisonKind(const FunctionDecl *FD) {
    return getDefaultedFunctionKind(FD).asComparison();
  }

  void ActOnLastBitfield(SourceLocation DeclStart,
                         SmallVectorImpl<Decl *> &AllIvarDecls);
  Decl *ActOnIvar(Scope *S, SourceLocation DeclStart,
                  Declarator &D, Expr *BitfieldWidth,
                  tok::ObjCKeywordKind visibility);

  // This is used for both record definitions and ObjC interface declarations.
  void ActOnFields(Scope *S, SourceLocation RecLoc, Decl *TagDecl,
                   ArrayRef<Decl *> Fields, SourceLocation LBrac,
                   SourceLocation RBrac, const ParsedAttributesView &AttrList);

  /// ActOnTagStartDefinition - Invoked when we have entered the
  /// scope of a tag's definition (e.g., for an enumeration, class,
  /// struct, or union).
  void ActOnTagStartDefinition(Scope *S, Decl *TagDecl);

  /// Perform ODR-like check for C/ObjC when merging tag types from modules.
  /// Differently from C++, actually parse the body and reject / error out
  /// in case of a structural mismatch.
  bool ActOnDuplicateDefinition(DeclSpec &DS, Decl *Prev,
                                SkipBodyInfo &SkipBody);

  typedef void *SkippedDefinitionContext;

  /// Invoked when we enter a tag definition that we're skipping.
  SkippedDefinitionContext ActOnTagStartSkippedDefinition(Scope *S, Decl *TD);

  Decl *ActOnObjCContainerStartDefinition(Decl *IDecl);

  /// ActOnStartCXXMemberDeclarations - Invoked when we have parsed a
  /// C++ record definition's base-specifiers clause and are starting its
  /// member declarations.
  void ActOnStartCXXMemberDeclarations(Scope *S, Decl *TagDecl,
                                       SourceLocation FinalLoc,
                                       bool IsFinalSpelledSealed,
                                       bool IsAbstract,
                                       SourceLocation LBraceLoc);

  /// ActOnTagFinishDefinition - Invoked once we have finished parsing
  /// the definition of a tag (enumeration, class, struct, or union).
  void ActOnTagFinishDefinition(Scope *S, Decl *TagDecl,
                                SourceRange BraceRange);

  void ActOnTagFinishSkippedDefinition(SkippedDefinitionContext Context);

  void ActOnObjCContainerFinishDefinition();

  /// Invoked when we must temporarily exit the objective-c container
  /// scope for parsing/looking-up C constructs.
  ///
  /// Must be followed by a call to \see ActOnObjCReenterContainerContext
  void ActOnObjCTemporaryExitContainerContext(DeclContext *DC);
  void ActOnObjCReenterContainerContext(DeclContext *DC);

  /// ActOnTagDefinitionError - Invoked when there was an unrecoverable
  /// error parsing the definition of a tag.
  void ActOnTagDefinitionError(Scope *S, Decl *TagDecl);

  EnumConstantDecl *CheckEnumConstant(EnumDecl *Enum,
                                      EnumConstantDecl *LastEnumConst,
                                      SourceLocation IdLoc,
                                      IdentifierInfo *Id,
                                      Expr *val);
  bool CheckEnumUnderlyingType(TypeSourceInfo *TI);
  bool CheckEnumRedeclaration(SourceLocation EnumLoc, bool IsScoped,
                              QualType EnumUnderlyingTy, bool IsFixed,
                              const EnumDecl *Prev);

  /// Determine whether the body of an anonymous enumeration should be skipped.
  /// \param II The name of the first enumerator.
  SkipBodyInfo shouldSkipAnonEnumBody(Scope *S, IdentifierInfo *II,
                                      SourceLocation IILoc);

  Decl *ActOnEnumConstant(Scope *S, Decl *EnumDecl, Decl *LastEnumConstant,
                          SourceLocation IdLoc, IdentifierInfo *Id,
                          const ParsedAttributesView &Attrs,
                          SourceLocation EqualLoc, Expr *Val);
  void ActOnEnumBody(SourceLocation EnumLoc, SourceRange BraceRange,
                     Decl *EnumDecl, ArrayRef<Decl *> Elements, Scope *S,
                     const ParsedAttributesView &Attr);

  /// Set the current declaration context until it gets popped.
  void PushDeclContext(Scope *S, DeclContext *DC);
  void PopDeclContext();

  /// EnterDeclaratorContext - Used when we must lookup names in the context
  /// of a declarator's nested name specifier.
  void EnterDeclaratorContext(Scope *S, DeclContext *DC);
  void ExitDeclaratorContext(Scope *S);

  /// Enter a template parameter scope, after it's been associated with a particular
  /// DeclContext. Causes lookup within the scope to chain through enclosing contexts
  /// in the correct order.
  void EnterTemplatedContext(Scope *S, DeclContext *DC);

  /// Push the parameters of D, which must be a function, into scope.
  void ActOnReenterFunctionContext(Scope* S, Decl* D);
  void ActOnExitFunctionContext();

  DeclContext *getFunctionLevelDeclContext();

  /// getCurFunctionDecl - If inside of a function body, this returns a pointer
  /// to the function decl for the function being parsed.  If we're currently
  /// in a 'block', this returns the containing context.
  FunctionDecl *getCurFunctionDecl();

  /// getCurMethodDecl - If inside of a method body, this returns a pointer to
  /// the method decl for the method being parsed.  If we're currently
  /// in a 'block', this returns the containing context.
  ObjCMethodDecl *getCurMethodDecl();

  /// getCurFunctionOrMethodDecl - Return the Decl for the current ObjC method
  /// or C function we're in, otherwise return null.  If we're currently
  /// in a 'block', this returns the containing context.
  NamedDecl *getCurFunctionOrMethodDecl();

  /// Add this decl to the scope shadowed decl chains.
  void PushOnScopeChains(NamedDecl *D, Scope *S, bool AddToContext = true);

  /// isDeclInScope - If 'Ctx' is a function/method, isDeclInScope returns true
  /// if 'D' is in Scope 'S', otherwise 'S' is ignored and isDeclInScope returns
  /// true if 'D' belongs to the given declaration context.
  ///
  /// \param AllowInlineNamespace If \c true, allow the declaration to be in the
  ///        enclosing namespace set of the context, rather than contained
  ///        directly within it.
  bool isDeclInScope(NamedDecl *D, DeclContext *Ctx, Scope *S = nullptr,
                     bool AllowInlineNamespace = false);

  /// Finds the scope corresponding to the given decl context, if it
  /// happens to be an enclosing scope.  Otherwise return NULL.
  static Scope *getScopeForDeclContext(Scope *S, DeclContext *DC);

  /// Subroutines of ActOnDeclarator().
  TypedefDecl *ParseTypedefDecl(Scope *S, Declarator &D, QualType T,
                                TypeSourceInfo *TInfo);
  bool isIncompatibleTypedef(TypeDecl *Old, TypedefNameDecl *New);

  /// Describes the kind of merge to perform for availability
  /// attributes (including "deprecated", "unavailable", and "availability").
  enum AvailabilityMergeKind {
    /// Don't merge availability attributes at all.
    AMK_None,
    /// Merge availability attributes for a redeclaration, which requires
    /// an exact match.
    AMK_Redeclaration,
    /// Merge availability attributes for an override, which requires
    /// an exact match or a weakening of constraints.
    AMK_Override,
    /// Merge availability attributes for an implementation of
    /// a protocol requirement.
    AMK_ProtocolImplementation,
    /// Merge availability attributes for an implementation of
    /// an optional protocol requirement.
    AMK_OptionalProtocolImplementation
  };

  /// Describes the kind of priority given to an availability attribute.
  ///
  /// The sum of priorities deteremines the final priority of the attribute.
  /// The final priority determines how the attribute will be merged.
  /// An attribute with a lower priority will always remove higher priority
  /// attributes for the specified platform when it is being applied. An
  /// attribute with a higher priority will not be applied if the declaration
  /// already has an availability attribute with a lower priority for the
  /// specified platform. The final prirority values are not expected to match
  /// the values in this enumeration, but instead should be treated as a plain
  /// integer value. This enumeration just names the priority weights that are
  /// used to calculate that final vaue.
  enum AvailabilityPriority : int {
    /// The availability attribute was specified explicitly next to the
    /// declaration.
    AP_Explicit = 0,

    /// The availability attribute was applied using '#pragma clang attribute'.
    AP_PragmaClangAttribute = 1,

    /// The availability attribute for a specific platform was inferred from
    /// an availability attribute for another platform.
    AP_InferredFromOtherPlatform = 2
  };

  /// Attribute merging methods. Return true if a new attribute was added.
  AvailabilityAttr *
  mergeAvailabilityAttr(NamedDecl *D, const AttributeCommonInfo &CI,
                        IdentifierInfo *Platform, bool Implicit,
                        VersionTuple Introduced, VersionTuple Deprecated,
                        VersionTuple Obsoleted, bool IsUnavailable,
                        StringRef Message, bool IsStrict, StringRef Replacement,
                        AvailabilityMergeKind AMK, int Priority);
  TypeVisibilityAttr *
  mergeTypeVisibilityAttr(Decl *D, const AttributeCommonInfo &CI,
                          TypeVisibilityAttr::VisibilityType Vis);
  VisibilityAttr *mergeVisibilityAttr(Decl *D, const AttributeCommonInfo &CI,
                                      VisibilityAttr::VisibilityType Vis);
  UuidAttr *mergeUuidAttr(Decl *D, const AttributeCommonInfo &CI,
                          StringRef UuidAsWritten, MSGuidDecl *GuidDecl);
  DLLImportAttr *mergeDLLImportAttr(Decl *D, const AttributeCommonInfo &CI);
  DLLExportAttr *mergeDLLExportAttr(Decl *D, const AttributeCommonInfo &CI);
  MSInheritanceAttr *mergeMSInheritanceAttr(Decl *D,
                                            const AttributeCommonInfo &CI,
                                            bool BestCase,
                                            MSInheritanceModel Model);
  ErrorAttr *mergeErrorAttr(Decl *D, const AttributeCommonInfo &CI,
                            StringRef NewUserDiagnostic);
  FormatAttr *mergeFormatAttr(Decl *D, const AttributeCommonInfo &CI,
                              IdentifierInfo *Format, int FormatIdx,
                              int FirstArg);
  SectionAttr *mergeSectionAttr(Decl *D, const AttributeCommonInfo &CI,
                                StringRef Name);
  CodeSegAttr *mergeCodeSegAttr(Decl *D, const AttributeCommonInfo &CI,
                                StringRef Name);
  AlwaysInlineAttr *mergeAlwaysInlineAttr(Decl *D,
                                          const AttributeCommonInfo &CI,
                                          const IdentifierInfo *Ident);
  MinSizeAttr *mergeMinSizeAttr(Decl *D, const AttributeCommonInfo &CI);
  SwiftNameAttr *mergeSwiftNameAttr(Decl *D, const SwiftNameAttr &SNA,
                                    StringRef Name);
  OptimizeNoneAttr *mergeOptimizeNoneAttr(Decl *D,
                                          const AttributeCommonInfo &CI);
  InternalLinkageAttr *mergeInternalLinkageAttr(Decl *D, const ParsedAttr &AL);
  InternalLinkageAttr *mergeInternalLinkageAttr(Decl *D,
                                                const InternalLinkageAttr &AL);
  WebAssemblyImportNameAttr *mergeImportNameAttr(
      Decl *D, const WebAssemblyImportNameAttr &AL);
  WebAssemblyImportModuleAttr *mergeImportModuleAttr(
      Decl *D, const WebAssemblyImportModuleAttr &AL);
  EnforceTCBAttr *mergeEnforceTCBAttr(Decl *D, const EnforceTCBAttr &AL);
  EnforceTCBLeafAttr *mergeEnforceTCBLeafAttr(Decl *D,
                                              const EnforceTCBLeafAttr &AL);
  BTFDeclTagAttr *mergeBTFDeclTagAttr(Decl *D, const BTFDeclTagAttr &AL);

  void mergeDeclAttributes(NamedDecl *New, Decl *Old,
                           AvailabilityMergeKind AMK = AMK_Redeclaration);
  void MergeTypedefNameDecl(Scope *S, TypedefNameDecl *New,
                            LookupResult &OldDecls);
  bool MergeFunctionDecl(FunctionDecl *New, NamedDecl *&Old, Scope *S,
                         bool MergeTypeWithOld);
  bool MergeCompatibleFunctionDecls(FunctionDecl *New, FunctionDecl *Old,
                                    Scope *S, bool MergeTypeWithOld);
  void mergeObjCMethodDecls(ObjCMethodDecl *New, ObjCMethodDecl *Old);
  void MergeVarDecl(VarDecl *New, LookupResult &Previous);
  void MergeVarDeclTypes(VarDecl *New, VarDecl *Old, bool MergeTypeWithOld);
  void MergeVarDeclExceptionSpecs(VarDecl *New, VarDecl *Old);
  bool checkVarDeclRedefinition(VarDecl *OldDefn, VarDecl *NewDefn);
  void notePreviousDefinition(const NamedDecl *Old, SourceLocation New);
  bool MergeCXXFunctionDecl(FunctionDecl *New, FunctionDecl *Old, Scope *S);

  // AssignmentAction - This is used by all the assignment diagnostic functions
  // to represent what is actually causing the operation
  enum AssignmentAction {
    AA_Assigning,
    AA_Passing,
    AA_Returning,
    AA_Converting,
    AA_Initializing,
    AA_Sending,
    AA_Casting,
    AA_Passing_CFAudited
  };

  /// C++ Overloading.
  enum OverloadKind {
    /// This is a legitimate overload: the existing declarations are
    /// functions or function templates with different signatures.
    Ovl_Overload,

    /// This is not an overload because the signature exactly matches
    /// an existing declaration.
    Ovl_Match,

    /// This is not an overload because the lookup results contain a
    /// non-function.
    Ovl_NonFunction
  };
  OverloadKind CheckOverload(Scope *S,
                             FunctionDecl *New,
                             const LookupResult &OldDecls,
                             NamedDecl *&OldDecl,
                             bool IsForUsingDecl);
  bool IsOverload(FunctionDecl *New, FunctionDecl *Old, bool IsForUsingDecl,
                  bool ConsiderCudaAttrs = true,
                  bool ConsiderRequiresClauses = true);

  enum class AllowedExplicit {
    /// Allow no explicit functions to be used.
    None,
    /// Allow explicit conversion functions but not explicit constructors.
    Conversions,
    /// Allow both explicit conversion functions and explicit constructors.
    All
  };

  ImplicitConversionSequence
  TryImplicitConversion(Expr *From, QualType ToType,
                        bool SuppressUserConversions,
                        AllowedExplicit AllowExplicit,
                        bool InOverloadResolution,
                        bool CStyle,
                        bool AllowObjCWritebackConversion);

  bool IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType);
  bool IsFloatingPointPromotion(QualType FromType, QualType ToType);
  bool IsComplexPromotion(QualType FromType, QualType ToType);
  bool IsPointerConversion(Expr *From, QualType FromType, QualType ToType,
                           bool InOverloadResolution,
                           QualType& ConvertedType, bool &IncompatibleObjC);
  bool isObjCPointerConversion(QualType FromType, QualType ToType,
                               QualType& ConvertedType, bool &IncompatibleObjC);
  bool isObjCWritebackConversion(QualType FromType, QualType ToType,
                                 QualType &ConvertedType);
  bool IsBlockPointerConversion(QualType FromType, QualType ToType,
                                QualType& ConvertedType);
  bool FunctionParamTypesAreEqual(const FunctionProtoType *OldType,
                                  const FunctionProtoType *NewType,
                                  unsigned *ArgPos = nullptr);
  void HandleFunctionTypeMismatch(PartialDiagnostic &PDiag,
                                  QualType FromType, QualType ToType);

  void maybeExtendBlockObject(ExprResult &E);
  CastKind PrepareCastToObjCObjectPointer(ExprResult &E);
  bool CheckPointerConversion(Expr *From, QualType ToType,
                              CastKind &Kind,
                              CXXCastPath& BasePath,
                              bool IgnoreBaseAccess,
                              bool Diagnose = true);
  bool IsMemberPointerConversion(Expr *From, QualType FromType, QualType ToType,
                                 bool InOverloadResolution,
                                 QualType &ConvertedType);
  bool CheckMemberPointerConversion(Expr *From, QualType ToType,
                                    CastKind &Kind,
                                    CXXCastPath &BasePath,
                                    bool IgnoreBaseAccess);
  bool IsQualificationConversion(QualType FromType, QualType ToType,
                                 bool CStyle, bool &ObjCLifetimeConversion);
  bool IsFunctionConversion(QualType FromType, QualType ToType,
                            QualType &ResultTy);
  bool DiagnoseMultipleUserDefinedConversion(Expr *From, QualType ToType);
  bool isSameOrCompatibleFunctionType(QualType Param, QualType Arg);

  bool CanPerformAggregateInitializationForOverloadResolution(
      const InitializedEntity &Entity, InitListExpr *From);

  bool IsStringInit(Expr *Init, const ArrayType *AT);

  bool CanPerformCopyInitialization(const InitializedEntity &Entity,
                                    ExprResult Init);
  ExprResult PerformCopyInitialization(const InitializedEntity &Entity,
                                       SourceLocation EqualLoc,
                                       ExprResult Init,
                                       bool TopLevelOfInitList = false,
                                       bool AllowExplicit = false);
  ExprResult PerformObjectArgumentInitialization(Expr *From,
                                                 NestedNameSpecifier *Qualifier,
                                                 NamedDecl *FoundDecl,
                                                 CXXMethodDecl *Method);

  /// Check that the lifetime of the initializer (and its subobjects) is
  /// sufficient for initializing the entity, and perform lifetime extension
  /// (when permitted) if not.
  void checkInitializerLifetime(const InitializedEntity &Entity, Expr *Init);

  ExprResult PerformContextuallyConvertToBool(Expr *From);
  ExprResult PerformContextuallyConvertToObjCPointer(Expr *From);

  /// Contexts in which a converted constant expression is required.
  enum CCEKind {
    CCEK_CaseValue,    ///< Expression in a case label.
    CCEK_Enumerator,   ///< Enumerator value with fixed underlying type.
    CCEK_TemplateArg,  ///< Value of a non-type template parameter.
    CCEK_ArrayBound,   ///< Array bound in array declarator or new-expression.
    CCEK_ExplicitBool, ///< Condition in an explicit(bool) specifier.
    CCEK_Noexcept      ///< Condition in a noexcept(bool) specifier.
  };
  ExprResult CheckConvertedConstantExpression(Expr *From, QualType T,
                                              llvm::APSInt &Value, CCEKind CCE);
  ExprResult CheckConvertedConstantExpression(Expr *From, QualType T,
                                              APValue &Value, CCEKind CCE,
                                              NamedDecl *Dest = nullptr);

  /// Abstract base class used to perform a contextual implicit
  /// conversion from an expression to any type passing a filter.
  class ContextualImplicitConverter {
  public:
    bool Suppress;
    bool SuppressConversion;

    ContextualImplicitConverter(bool Suppress = false,
                                bool SuppressConversion = false)
        : Suppress(Suppress), SuppressConversion(SuppressConversion) {}

    /// Determine whether the specified type is a valid destination type
    /// for this conversion.
    virtual bool match(QualType T) = 0;

    /// Emits a diagnostic complaining that the expression does not have
    /// integral or enumeration type.
    virtual SemaDiagnosticBuilder
    diagnoseNoMatch(Sema &S, SourceLocation Loc, QualType T) = 0;

    /// Emits a diagnostic when the expression has incomplete class type.
    virtual SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) = 0;

    /// Emits a diagnostic when the only matching conversion function
    /// is explicit.
    virtual SemaDiagnosticBuilder diagnoseExplicitConv(
        Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) = 0;

    /// Emits a note for the explicit conversion function.
    virtual SemaDiagnosticBuilder
    noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) = 0;

    /// Emits a diagnostic when there are multiple possible conversion
    /// functions.
    virtual SemaDiagnosticBuilder
    diagnoseAmbiguous(Sema &S, SourceLocation Loc, QualType T) = 0;

    /// Emits a note for one of the candidate conversions.
    virtual SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) = 0;

    /// Emits a diagnostic when we picked a conversion function
    /// (for cases when we are not allowed to pick a conversion function).
    virtual SemaDiagnosticBuilder diagnoseConversion(
        Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) = 0;

    virtual ~ContextualImplicitConverter() {}
  };

  class ICEConvertDiagnoser : public ContextualImplicitConverter {
    bool AllowScopedEnumerations;

  public:
    ICEConvertDiagnoser(bool AllowScopedEnumerations,
                        bool Suppress, bool SuppressConversion)
        : ContextualImplicitConverter(Suppress, SuppressConversion),
          AllowScopedEnumerations(AllowScopedEnumerations) {}

    /// Match an integral or (possibly scoped) enumeration type.
    bool match(QualType T) override;

    SemaDiagnosticBuilder
    diagnoseNoMatch(Sema &S, SourceLocation Loc, QualType T) override {
      return diagnoseNotInt(S, Loc, T);
    }

    /// Emits a diagnostic complaining that the expression does not have
    /// integral or enumeration type.
    virtual SemaDiagnosticBuilder
    diagnoseNotInt(Sema &S, SourceLocation Loc, QualType T) = 0;
  };

  /// Perform a contextual implicit conversion.
  ExprResult PerformContextualImplicitConversion(
      SourceLocation Loc, Expr *FromE, ContextualImplicitConverter &Converter);


  enum ObjCSubscriptKind {
    OS_Array,
    OS_Dictionary,
    OS_Error
  };
  ObjCSubscriptKind CheckSubscriptingKind(Expr *FromE);

  // Note that LK_String is intentionally after the other literals, as
  // this is used for diagnostics logic.
  enum ObjCLiteralKind {
    LK_Array,
    LK_Dictionary,
    LK_Numeric,
    LK_Boxed,
    LK_String,
    LK_Block,
    LK_None
  };
  ObjCLiteralKind CheckLiteralKind(Expr *FromE);

  ExprResult PerformObjectMemberConversion(Expr *From,
                                           NestedNameSpecifier *Qualifier,
                                           NamedDecl *FoundDecl,
                                           NamedDecl *Member);

  // Members have to be NamespaceDecl* or TranslationUnitDecl*.
  // TODO: make this is a typesafe union.
  typedef llvm::SmallSetVector<DeclContext   *, 16> AssociatedNamespaceSet;
  typedef llvm::SmallSetVector<CXXRecordDecl *, 16> AssociatedClassSet;

  using ADLCallKind = CallExpr::ADLCallKind;

  void AddOverloadCandidate(FunctionDecl *Function, DeclAccessPair FoundDecl,
                            ArrayRef<Expr *> Args,
                            OverloadCandidateSet &CandidateSet,
                            bool SuppressUserConversions = false,
                            bool PartialOverloading = false,
                            bool AllowExplicit = true,
                            bool AllowExplicitConversion = false,
                            ADLCallKind IsADLCandidate = ADLCallKind::NotADL,
                            ConversionSequenceList EarlyConversions = None,
                            OverloadCandidateParamOrder PO = {});
  void AddFunctionCandidates(const UnresolvedSetImpl &Functions,
                      ArrayRef<Expr *> Args,
                      OverloadCandidateSet &CandidateSet,
                      TemplateArgumentListInfo *ExplicitTemplateArgs = nullptr,
                      bool SuppressUserConversions = false,
                      bool PartialOverloading = false,
                      bool FirstArgumentIsBase = false);
  void AddMethodCandidate(DeclAccessPair FoundDecl,
                          QualType ObjectType,
                          Expr::Classification ObjectClassification,
                          ArrayRef<Expr *> Args,
                          OverloadCandidateSet& CandidateSet,
                          bool SuppressUserConversion = false,
                          OverloadCandidateParamOrder PO = {});
  void AddMethodCandidate(CXXMethodDecl *Method,
                          DeclAccessPair FoundDecl,
                          CXXRecordDecl *ActingContext, QualType ObjectType,
                          Expr::Classification ObjectClassification,
                          ArrayRef<Expr *> Args,
                          OverloadCandidateSet& CandidateSet,
                          bool SuppressUserConversions = false,
                          bool PartialOverloading = false,
                          ConversionSequenceList EarlyConversions = None,
                          OverloadCandidateParamOrder PO = {});
  void AddMethodTemplateCandidate(FunctionTemplateDecl *MethodTmpl,
                                  DeclAccessPair FoundDecl,
                                  CXXRecordDecl *ActingContext,
                                 TemplateArgumentListInfo *ExplicitTemplateArgs,
                                  QualType ObjectType,
                                  Expr::Classification ObjectClassification,
                                  ArrayRef<Expr *> Args,
                                  OverloadCandidateSet& CandidateSet,
                                  bool SuppressUserConversions = false,
                                  bool PartialOverloading = false,
                                  OverloadCandidateParamOrder PO = {});
  void AddTemplateOverloadCandidate(
      FunctionTemplateDecl *FunctionTemplate, DeclAccessPair FoundDecl,
      TemplateArgumentListInfo *ExplicitTemplateArgs, ArrayRef<Expr *> Args,
      OverloadCandidateSet &CandidateSet, bool SuppressUserConversions = false,
      bool PartialOverloading = false, bool AllowExplicit = true,
      ADLCallKind IsADLCandidate = ADLCallKind::NotADL,
      OverloadCandidateParamOrder PO = {});
  bool CheckNonDependentConversions(
      FunctionTemplateDecl *FunctionTemplate, ArrayRef<QualType> ParamTypes,
      ArrayRef<Expr *> Args, OverloadCandidateSet &CandidateSet,
      ConversionSequenceList &Conversions, bool SuppressUserConversions,
      CXXRecordDecl *ActingContext = nullptr, QualType ObjectType = QualType(),
      Expr::Classification ObjectClassification = {},
      OverloadCandidateParamOrder PO = {});
  void AddConversionCandidate(
      CXXConversionDecl *Conversion, DeclAccessPair FoundDecl,
      CXXRecordDecl *ActingContext, Expr *From, QualType ToType,
      OverloadCandidateSet &CandidateSet, bool AllowObjCConversionOnExplicit,
      bool AllowExplicit, bool AllowResultConversion = true);
  void AddTemplateConversionCandidate(
      FunctionTemplateDecl *FunctionTemplate, DeclAccessPair FoundDecl,
      CXXRecordDecl *ActingContext, Expr *From, QualType ToType,
      OverloadCandidateSet &CandidateSet, bool AllowObjCConversionOnExplicit,
      bool AllowExplicit, bool AllowResultConversion = true);
  void AddSurrogateCandidate(CXXConversionDecl *Conversion,
                             DeclAccessPair FoundDecl,
                             CXXRecordDecl *ActingContext,
                             const FunctionProtoType *Proto,
                             Expr *Object, ArrayRef<Expr *> Args,
                             OverloadCandidateSet& CandidateSet);
  void AddNonMemberOperatorCandidates(
      const UnresolvedSetImpl &Functions, ArrayRef<Expr *> Args,
      OverloadCandidateSet &CandidateSet,
      TemplateArgumentListInfo *ExplicitTemplateArgs = nullptr);
  void AddMemberOperatorCandidates(OverloadedOperatorKind Op,
                                   SourceLocation OpLoc, ArrayRef<Expr *> Args,
                                   OverloadCandidateSet &CandidateSet,
                                   OverloadCandidateParamOrder PO = {});
  void AddBuiltinCandidate(QualType *ParamTys, ArrayRef<Expr *> Args,
                           OverloadCandidateSet& CandidateSet,
                           bool IsAssignmentOperator = false,
                           unsigned NumContextualBoolArguments = 0);
  void AddBuiltinOperatorCandidates(OverloadedOperatorKind Op,
                                    SourceLocation OpLoc, ArrayRef<Expr *> Args,
                                    OverloadCandidateSet& CandidateSet);
  void AddArgumentDependentLookupCandidates(DeclarationName Name,
                                            SourceLocation Loc,
                                            ArrayRef<Expr *> Args,
                                TemplateArgumentListInfo *ExplicitTemplateArgs,
                                            OverloadCandidateSet& CandidateSet,
                                            bool PartialOverloading = false);

  // Emit as a 'note' the specific overload candidate
  void NoteOverloadCandidate(
      NamedDecl *Found, FunctionDecl *Fn,
      OverloadCandidateRewriteKind RewriteKind = OverloadCandidateRewriteKind(),
      QualType DestType = QualType(), bool TakingAddress = false);

  // Emit as a series of 'note's all template and non-templates identified by
  // the expression Expr
  void NoteAllOverloadCandidates(Expr *E, QualType DestType = QualType(),
                                 bool TakingAddress = false);

  /// Check the enable_if expressions on the given function. Returns the first
  /// failing attribute, or NULL if they were all successful.
  EnableIfAttr *CheckEnableIf(FunctionDecl *Function, SourceLocation CallLoc,
                              ArrayRef<Expr *> Args,
                              bool MissingImplicitThis = false);

  /// Find the failed Boolean condition within a given Boolean
  /// constant expression, and describe it with a string.
  std::pair<Expr *, std::string> findFailedBooleanCondition(Expr *Cond);

  /// Emit diagnostics for the diagnose_if attributes on Function, ignoring any
  /// non-ArgDependent DiagnoseIfAttrs.
  ///
  /// Argument-dependent diagnose_if attributes should be checked each time a
  /// function is used as a direct callee of a function call.
  ///
  /// Returns true if any errors were emitted.
  bool diagnoseArgDependentDiagnoseIfAttrs(const FunctionDecl *Function,
                                           const Expr *ThisArg,
                                           ArrayRef<const Expr *> Args,
                                           SourceLocation Loc);

  /// Emit diagnostics for the diagnose_if attributes on Function, ignoring any
  /// ArgDependent DiagnoseIfAttrs.
  ///
  /// Argument-independent diagnose_if attributes should be checked on every use
  /// of a function.
  ///
  /// Returns true if any errors were emitted.
  bool diagnoseArgIndependentDiagnoseIfAttrs(const NamedDecl *ND,
                                             SourceLocation Loc);

  /// Returns whether the given function's address can be taken or not,
  /// optionally emitting a diagnostic if the address can't be taken.
  ///
  /// Returns false if taking the address of the function is illegal.
  bool checkAddressOfFunctionIsAvailable(const FunctionDecl *Function,
                                         bool Complain = false,
                                         SourceLocation Loc = SourceLocation());

  // [PossiblyAFunctionType]  -->   [Return]
  // NonFunctionType --> NonFunctionType
  // R (A) --> R(A)
  // R (*)(A) --> R (A)
  // R (&)(A) --> R (A)
  // R (S::*)(A) --> R (A)
  QualType ExtractUnqualifiedFunctionType(QualType PossiblyAFunctionType);

  FunctionDecl *
  ResolveAddressOfOverloadedFunction(Expr *AddressOfExpr,
                                     QualType TargetType,
                                     bool Complain,
                                     DeclAccessPair &Found,
                                     bool *pHadMultipleCandidates = nullptr);

  FunctionDecl *
  resolveAddressOfSingleOverloadCandidate(Expr *E, DeclAccessPair &FoundResult);

  bool resolveAndFixAddressOfSingleOverloadCandidate(
      ExprResult &SrcExpr, bool DoFunctionPointerConversion = false);

  FunctionDecl *
  ResolveSingleFunctionTemplateSpecialization(OverloadExpr *ovl,
                                              bool Complain = false,
                                              DeclAccessPair *Found = nullptr);

  bool ResolveAndFixSingleFunctionTemplateSpecialization(
                      ExprResult &SrcExpr,
                      bool DoFunctionPointerConverion = false,
                      bool Complain = false,
                      SourceRange OpRangeForComplaining = SourceRange(),
                      QualType DestTypeForComplaining = QualType(),
                      unsigned DiagIDForComplaining = 0);


  Expr *FixOverloadedFunctionReference(Expr *E,
                                       DeclAccessPair FoundDecl,
                                       FunctionDecl *Fn);
  ExprResult FixOverloadedFunctionReference(ExprResult,
                                            DeclAccessPair FoundDecl,
                                            FunctionDecl *Fn);

  void AddOverloadedCallCandidates(UnresolvedLookupExpr *ULE,
                                   ArrayRef<Expr *> Args,
                                   OverloadCandidateSet &CandidateSet,
                                   bool PartialOverloading = false);
  void AddOverloadedCallCandidates(
      LookupResult &R, TemplateArgumentListInfo *ExplicitTemplateArgs,
      ArrayRef<Expr *> Args, OverloadCandidateSet &CandidateSet);

  // An enum used to represent the different possible results of building a
  // range-based for loop.
  enum ForRangeStatus {
    FRS_Success,
    FRS_NoViableFunction,
    FRS_DiagnosticIssued
  };

  ForRangeStatus BuildForRangeBeginEndCall(SourceLocation Loc,
                                           SourceLocation RangeLoc,
                                           const DeclarationNameInfo &NameInfo,
                                           LookupResult &MemberLookup,
                                           OverloadCandidateSet *CandidateSet,
                                           Expr *Range, ExprResult *CallExpr);

  ExprResult BuildOverloadedCallExpr(Scope *S, Expr *Fn,
                                     UnresolvedLookupExpr *ULE,
                                     SourceLocation LParenLoc,
                                     MultiExprArg Args,
                                     SourceLocation RParenLoc,
                                     Expr *ExecConfig,
                                     bool AllowTypoCorrection=true,
                                     bool CalleesAddressIsTaken=false);

  bool buildOverloadedCallSet(Scope *S, Expr *Fn, UnresolvedLookupExpr *ULE,
                              MultiExprArg Args, SourceLocation RParenLoc,
                              OverloadCandidateSet *CandidateSet,
                              ExprResult *Result);

  ExprResult CreateUnresolvedLookupExpr(CXXRecordDecl *NamingClass,
                                        NestedNameSpecifierLoc NNSLoc,
                                        DeclarationNameInfo DNI,
                                        const UnresolvedSetImpl &Fns,
                                        bool PerformADL = true);

  ExprResult CreateOverloadedUnaryOp(SourceLocation OpLoc,
                                     UnaryOperatorKind Opc,
                                     const UnresolvedSetImpl &Fns,
                                     Expr *input, bool RequiresADL = true);

  void LookupOverloadedBinOp(OverloadCandidateSet &CandidateSet,
                             OverloadedOperatorKind Op,
                             const UnresolvedSetImpl &Fns,
                             ArrayRef<Expr *> Args, bool RequiresADL = true);
  ExprResult CreateOverloadedBinOp(SourceLocation OpLoc,
                                   BinaryOperatorKind Opc,
                                   const UnresolvedSetImpl &Fns,
                                   Expr *LHS, Expr *RHS,
                                   bool RequiresADL = true,
                                   bool AllowRewrittenCandidates = true,
                                   FunctionDecl *DefaultedFn = nullptr);
  ExprResult BuildSynthesizedThreeWayComparison(SourceLocation OpLoc,
                                                const UnresolvedSetImpl &Fns,
                                                Expr *LHS, Expr *RHS,
                                                FunctionDecl *DefaultedFn);

  ExprResult CreateOverloadedArraySubscriptExpr(SourceLocation LLoc,
                                                SourceLocation RLoc,
                                                Expr *Base,Expr *Idx);

  ExprResult BuildCallToMemberFunction(Scope *S, Expr *MemExpr,
                                       SourceLocation LParenLoc,
                                       MultiExprArg Args,
                                       SourceLocation RParenLoc,
                                       Expr *ExecConfig = nullptr,
                                       bool IsExecConfig = false,
                                       bool AllowRecovery = false);
  ExprResult
  BuildCallToObjectOfClassType(Scope *S, Expr *Object, SourceLocation LParenLoc,
                               MultiExprArg Args,
                               SourceLocation RParenLoc);

  ExprResult BuildOverloadedArrowExpr(Scope *S, Expr *Base,
                                      SourceLocation OpLoc,
                                      bool *NoArrowOperatorFound = nullptr);

  /// CheckCallReturnType - Checks that a call expression's return type is
  /// complete. Returns true on failure. The location passed in is the location
  /// that best represents the call.
  bool CheckCallReturnType(QualType ReturnType, SourceLocation Loc,
                           CallExpr *CE, FunctionDecl *FD);

  /// Helpers for dealing with blocks and functions.
  bool CheckParmsForFunctionDef(ArrayRef<ParmVarDecl *> Parameters,
                                bool CheckParameterNames);
  void CheckCXXDefaultArguments(FunctionDecl *FD);
  void CheckExtraCXXDefaultArguments(Declarator &D);
  Scope *getNonFieldDeclScope(Scope *S);

  /// \name Name lookup
  ///
  /// These routines provide name lookup that is used during semantic
  /// analysis to resolve the various kinds of names (identifiers,
  /// overloaded operator names, constructor names, etc.) into zero or
  /// more declarations within a particular scope. The major entry
  /// points are LookupName, which performs unqualified name lookup,
  /// and LookupQualifiedName, which performs qualified name lookup.
  ///
  /// All name lookup is performed based on some specific criteria,
  /// which specify what names will be visible to name lookup and how
  /// far name lookup should work. These criteria are important both
  /// for capturing language semantics (certain lookups will ignore
  /// certain names, for example) and for performance, since name
  /// lookup is often a bottleneck in the compilation of C++. Name
  /// lookup criteria is specified via the LookupCriteria enumeration.
  ///
  /// The results of name lookup can vary based on the kind of name
  /// lookup performed, the current language, and the translation
  /// unit. In C, for example, name lookup will either return nothing
  /// (no entity found) or a single declaration. In C++, name lookup
  /// can additionally refer to a set of overloaded functions or
  /// result in an ambiguity. All of the possible results of name
  /// lookup are captured by the LookupResult class, which provides
  /// the ability to distinguish among them.
  //@{

  /// Describes the kind of name lookup to perform.
  enum LookupNameKind {
    /// Ordinary name lookup, which finds ordinary names (functions,
    /// variables, typedefs, etc.) in C and most kinds of names
    /// (functions, variables, members, types, etc.) in C++.
    LookupOrdinaryName = 0,
    /// Tag name lookup, which finds the names of enums, classes,
    /// structs, and unions.
    LookupTagName,
    /// Label name lookup.
    LookupLabel,
    /// Member name lookup, which finds the names of
    /// class/struct/union members.
    LookupMemberName,
    /// Look up of an operator name (e.g., operator+) for use with
    /// operator overloading. This lookup is similar to ordinary name
    /// lookup, but will ignore any declarations that are class members.
    LookupOperatorName,
    /// Look up a name following ~ in a destructor name. This is an ordinary
    /// lookup, but prefers tags to typedefs.
    LookupDestructorName,
    /// Look up of a name that precedes the '::' scope resolution
    /// operator in C++. This lookup completely ignores operator, object,
    /// function, and enumerator names (C++ [basic.lookup.qual]p1).
    LookupNestedNameSpecifierName,
    /// Look up a namespace name within a C++ using directive or
    /// namespace alias definition, ignoring non-namespace names (C++
    /// [basic.lookup.udir]p1).
    LookupNamespaceName,
    /// Look up all declarations in a scope with the given name,
    /// including resolved using declarations.  This is appropriate
    /// for checking redeclarations for a using declaration.
    LookupUsingDeclName,
    /// Look up an ordinary name that is going to be redeclared as a
    /// name with linkage. This lookup ignores any declarations that
    /// are outside of the current scope unless they have linkage. See
    /// C99 6.2.2p4-5 and C++ [basic.link]p6.
    LookupRedeclarationWithLinkage,
    /// Look up a friend of a local class. This lookup does not look
    /// outside the innermost non-class scope. See C++11 [class.friend]p11.
    LookupLocalFriendName,
    /// Look up the name of an Objective-C protocol.
    LookupObjCProtocolName,
    /// Look up implicit 'self' parameter of an objective-c method.
    LookupObjCImplicitSelfParam,
    /// Look up the name of an OpenMP user-defined reduction operation.
    LookupOMPReductionName,
    /// Look up the name of an OpenMP user-defined mapper.
    LookupOMPMapperName,
    /// Look up any declaration with any name.
    LookupAnyName
  };

  /// Specifies whether (or how) name lookup is being performed for a
  /// redeclaration (vs. a reference).
  enum RedeclarationKind {
    /// The lookup is a reference to this name that is not for the
    /// purpose of redeclaring the name.
    NotForRedeclaration = 0,
    /// The lookup results will be used for redeclaration of a name,
    /// if an entity by that name already exists and is visible.
    ForVisibleRedeclaration,
    /// The lookup results will be used for redeclaration of a name
    /// with external linkage; non-visible lookup results with external linkage
    /// may also be found.
    ForExternalRedeclaration
  };

  RedeclarationKind forRedeclarationInCurContext() {
    // A declaration with an owning module for linkage can never link against
    // anything that is not visible. We don't need to check linkage here; if
    // the context has internal linkage, redeclaration lookup won't find things
    // from other TUs, and we can't safely compute linkage yet in general.
    if (cast<Decl>(CurContext)
            ->getOwningModuleForLinkage(/*IgnoreLinkage*/true))
      return ForVisibleRedeclaration;
    return ForExternalRedeclaration;
  }

  /// The possible outcomes of name lookup for a literal operator.
  enum LiteralOperatorLookupResult {
    /// The lookup resulted in an error.
    LOLR_Error,
    /// The lookup found no match but no diagnostic was issued.
    LOLR_ErrorNoDiagnostic,
    /// The lookup found a single 'cooked' literal operator, which
    /// expects a normal literal to be built and passed to it.
    LOLR_Cooked,
    /// The lookup found a single 'raw' literal operator, which expects
    /// a string literal containing the spelling of the literal token.
    LOLR_Raw,
    /// The lookup found an overload set of literal operator templates,
    /// which expect the characters of the spelling of the literal token to be
    /// passed as a non-type template argument pack.
    LOLR_Template,
    /// The lookup found an overload set of literal operator templates,
    /// which expect the character type and characters of the spelling of the
    /// string literal token to be passed as template arguments.
    LOLR_StringTemplatePack,
  };

  SpecialMemberOverloadResult LookupSpecialMember(CXXRecordDecl *D,
                                                  CXXSpecialMember SM,
                                                  bool ConstArg,
                                                  bool VolatileArg,
                                                  bool RValueThis,
                                                  bool ConstThis,
                                                  bool VolatileThis);

  typedef std::function<void(const TypoCorrection &)> TypoDiagnosticGenerator;
  typedef std::function<ExprResult(Sema &, TypoExpr *, TypoCorrection)>
      TypoRecoveryCallback;

private:
  bool CppLookupName(LookupResult &R, Scope *S);

  struct TypoExprState {
    std::unique_ptr<TypoCorrectionConsumer> Consumer;
    TypoDiagnosticGenerator DiagHandler;
    TypoRecoveryCallback RecoveryHandler;
    TypoExprState();
    TypoExprState(TypoExprState &&other) noexcept;
    TypoExprState &operator=(TypoExprState &&other) noexcept;
  };

  /// The set of unhandled TypoExprs and their associated state.
  llvm::MapVector<TypoExpr *, TypoExprState> DelayedTypos;

  /// Creates a new TypoExpr AST node.
  TypoExpr *createDelayedTypo(std::unique_ptr<TypoCorrectionConsumer> TCC,
                              TypoDiagnosticGenerator TDG,
                              TypoRecoveryCallback TRC, SourceLocation TypoLoc);

  // The set of known/encountered (unique, canonicalized) NamespaceDecls.
  //
  // The boolean value will be true to indicate that the namespace was loaded
  // from an AST/PCH file, or false otherwise.
  llvm::MapVector<NamespaceDecl*, bool> KnownNamespaces;

  /// Whether we have already loaded known namespaces from an extenal
  /// source.
  bool LoadedExternalKnownNamespaces;

  /// Helper for CorrectTypo and CorrectTypoDelayed used to create and
  /// populate a new TypoCorrectionConsumer. Returns nullptr if typo correction
  /// should be skipped entirely.
  std::unique_ptr<TypoCorrectionConsumer>
  makeTypoCorrectionConsumer(const DeclarationNameInfo &Typo,
                             Sema::LookupNameKind LookupKind, Scope *S,
                             CXXScopeSpec *SS,
                             CorrectionCandidateCallback &CCC,
                             DeclContext *MemberContext, bool EnteringContext,
                             const ObjCObjectPointerType *OPT,
                             bool ErrorRecovery);

public:
  const TypoExprState &getTypoExprState(TypoExpr *TE) const;

  /// Clears the state of the given TypoExpr.
  void clearDelayedTypo(TypoExpr *TE);

  /// Look up a name, looking for a single declaration.  Return
  /// null if the results were absent, ambiguous, or overloaded.
  ///
  /// It is preferable to use the elaborated form and explicitly handle
  /// ambiguity and overloaded.
  NamedDecl *LookupSingleName(Scope *S, DeclarationName Name,
                              SourceLocation Loc,
                              LookupNameKind NameKind,
                              RedeclarationKind Redecl
                                = NotForRedeclaration);
  bool LookupBuiltin(LookupResult &R);
  void LookupNecessaryTypesForBuiltin(Scope *S, unsigned ID);
  bool LookupName(LookupResult &R, Scope *S,
                  bool AllowBuiltinCreation = false);
  bool LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx,
                           bool InUnqualifiedLookup = false);
  bool LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx,
                           CXXScopeSpec &SS);
  bool LookupParsedName(LookupResult &R, Scope *S, CXXScopeSpec *SS,
                        bool AllowBuiltinCreation = false,
                        bool EnteringContext = false);
  ObjCProtocolDecl *LookupProtocol(IdentifierInfo *II, SourceLocation IdLoc,
                                   RedeclarationKind Redecl
                                     = NotForRedeclaration);
  bool LookupInSuper(LookupResult &R, CXXRecordDecl *Class);

  void LookupOverloadedOperatorName(OverloadedOperatorKind Op, Scope *S,
                                    UnresolvedSetImpl &Functions);

  LabelDecl *LookupOrCreateLabel(IdentifierInfo *II, SourceLocation IdentLoc,
                                 SourceLocation GnuLabelLoc = SourceLocation());

  DeclContextLookupResult LookupConstructors(CXXRecordDecl *Class);
  CXXConstructorDecl *LookupDefaultConstructor(CXXRecordDecl *Class);
  CXXConstructorDecl *LookupCopyingConstructor(CXXRecordDecl *Class,
                                               unsigned Quals);
  CXXMethodDecl *LookupCopyingAssignment(CXXRecordDecl *Class, unsigned Quals,
                                         bool RValueThis, unsigned ThisQuals);
  CXXConstructorDecl *LookupMovingConstructor(CXXRecordDecl *Class,
                                              unsigned Quals);
  CXXMethodDecl *LookupMovingAssignment(CXXRecordDecl *Class, unsigned Quals,
                                        bool RValueThis, unsigned ThisQuals);
  CXXDestructorDecl *LookupDestructor(CXXRecordDecl *Class);

  bool checkLiteralOperatorId(const CXXScopeSpec &SS, const UnqualifiedId &Id,
                              bool IsUDSuffix);
  LiteralOperatorLookupResult
  LookupLiteralOperator(Scope *S, LookupResult &R, ArrayRef<QualType> ArgTys,
                        bool AllowRaw, bool AllowTemplate,
                        bool AllowStringTemplate, bool DiagnoseMissing,
                        StringLiteral *StringLit = nullptr);
  bool isKnownName(StringRef name);

  /// Status of the function emission on the CUDA/HIP/OpenMP host/device attrs.
  enum class FunctionEmissionStatus {
    Emitted,
    CUDADiscarded,     // Discarded due to CUDA/HIP hostness
    OMPDiscarded,      // Discarded due to OpenMP hostness
    TemplateDiscarded, // Discarded due to uninstantiated templates
    Unknown,
  };
  FunctionEmissionStatus getEmissionStatus(FunctionDecl *Decl,
                                           bool Final = false);

  // Whether the callee should be ignored in CUDA/HIP/OpenMP host/device check.
  bool shouldIgnoreInHostDeviceCheck(FunctionDecl *Callee);

  void ArgumentDependentLookup(DeclarationName Name, SourceLocation Loc,
                               ArrayRef<Expr *> Args, ADLResult &Functions);

  void LookupVisibleDecls(Scope *S, LookupNameKind Kind,
                          VisibleDeclConsumer &Consumer,
                          bool IncludeGlobalScope = true,
                          bool LoadExternal = true);
  void LookupVisibleDecls(DeclContext *Ctx, LookupNameKind Kind,
                          VisibleDeclConsumer &Consumer,
                          bool IncludeGlobalScope = true,
                          bool IncludeDependentBases = false,
                          bool LoadExternal = true);

  enum CorrectTypoKind {
    CTK_NonError,     // CorrectTypo used in a non error recovery situation.
    CTK_ErrorRecovery // CorrectTypo used in normal error recovery.
  };

  TypoCorrection CorrectTypo(const DeclarationNameInfo &Typo,
                             Sema::LookupNameKind LookupKind,
                             Scope *S, CXXScopeSpec *SS,
                             CorrectionCandidateCallback &CCC,
                             CorrectTypoKind Mode,
                             DeclContext *MemberContext = nullptr,
                             bool EnteringContext = false,
                             const ObjCObjectPointerType *OPT = nullptr,
                             bool RecordFailure = true);

  TypoExpr *CorrectTypoDelayed(const DeclarationNameInfo &Typo,
                               Sema::LookupNameKind LookupKind, Scope *S,
                               CXXScopeSpec *SS,
                               CorrectionCandidateCallback &CCC,
                               TypoDiagnosticGenerator TDG,
                               TypoRecoveryCallback TRC, CorrectTypoKind Mode,
                               DeclContext *MemberContext = nullptr,
                               bool EnteringContext = false,
                               const ObjCObjectPointerType *OPT = nullptr);

  /// Process any TypoExprs in the given Expr and its children,
  /// generating diagnostics as appropriate and returning a new Expr if there
  /// were typos that were all successfully corrected and ExprError if one or
  /// more typos could not be corrected.
  ///
  /// \param E The Expr to check for TypoExprs.
  ///
  /// \param InitDecl A VarDecl to avoid because the Expr being corrected is its
  /// initializer.
  ///
  /// \param RecoverUncorrectedTypos If true, when typo correction fails, it
  /// will rebuild the given Expr with all TypoExprs degraded to RecoveryExprs.
  ///
  /// \param Filter A function applied to a newly rebuilt Expr to determine if
  /// it is an acceptable/usable result from a single combination of typo
  /// corrections. As long as the filter returns ExprError, different
  /// combinations of corrections will be tried until all are exhausted.
  ExprResult CorrectDelayedTyposInExpr(
      Expr *E, VarDecl *InitDecl = nullptr,
      bool RecoverUncorrectedTypos = false,
      llvm::function_ref<ExprResult(Expr *)> Filter =
          [](Expr *E) -> ExprResult { return E; });

  ExprResult CorrectDelayedTyposInExpr(
      ExprResult ER, VarDecl *InitDecl = nullptr,
      bool RecoverUncorrectedTypos = false,
      llvm::function_ref<ExprResult(Expr *)> Filter =
          [](Expr *E) -> ExprResult { return E; }) {
    return ER.isInvalid()
               ? ER
               : CorrectDelayedTyposInExpr(ER.get(), InitDecl,
                                           RecoverUncorrectedTypos, Filter);
  }

  void diagnoseTypo(const TypoCorrection &Correction,
                    const PartialDiagnostic &TypoDiag,
                    bool ErrorRecovery = true);

  void diagnoseTypo(const TypoCorrection &Correction,
                    const PartialDiagnostic &TypoDiag,
                    const PartialDiagnostic &PrevNote,
                    bool ErrorRecovery = true);

  void MarkTypoCorrectedFunctionDefinition(const NamedDecl *F);

  void FindAssociatedClassesAndNamespaces(SourceLocation InstantiationLoc,
                                          ArrayRef<Expr *> Args,
                                   AssociatedNamespaceSet &AssociatedNamespaces,
                                   AssociatedClassSet &AssociatedClasses);

  void FilterLookupForScope(LookupResult &R, DeclContext *Ctx, Scope *S,
                            bool ConsiderLinkage, bool AllowInlineNamespace);

  bool CheckRedeclarationModuleOwnership(NamedDecl *New, NamedDecl *Old);

  void DiagnoseAmbiguousLookup(LookupResult &Result);
  //@}

  /// Attempts to produce a RecoveryExpr after some AST node cannot be created.
  ExprResult CreateRecoveryExpr(SourceLocation Begin, SourceLocation End,
                                ArrayRef<Expr *> SubExprs,
                                QualType T = QualType());

  ObjCInterfaceDecl *getObjCInterfaceDecl(IdentifierInfo *&Id,
                                          SourceLocation IdLoc,
                                          bool TypoCorrection = false);
  FunctionDecl *CreateBuiltin(IdentifierInfo *II, QualType Type, unsigned ID,
                              SourceLocation Loc);
  NamedDecl *LazilyCreateBuiltin(IdentifierInfo *II, unsigned ID,
                                 Scope *S, bool ForRedeclaration,
                                 SourceLocation Loc);
  NamedDecl *ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                      Scope *S);
  void AddKnownFunctionAttributesForReplaceableGlobalAllocationFunction(
      FunctionDecl *FD);
  void AddKnownFunctionAttributes(FunctionDecl *FD);

  // More parsing and symbol table subroutines.

  void ProcessPragmaWeak(Scope *S, Decl *D);
  // Decl attributes - this routine is the top level dispatcher.
  void ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD);
  // Helper for delayed processing of attributes.
  void ProcessDeclAttributeDelayed(Decl *D,
                                   const ParsedAttributesView &AttrList);
  void ProcessDeclAttributeList(Scope *S, Decl *D, const ParsedAttributesView &AL,
                             bool IncludeCXX11Attributes = true);
  bool ProcessAccessDeclAttributeList(AccessSpecDecl *ASDecl,
                                   const ParsedAttributesView &AttrList);

  void checkUnusedDeclAttributes(Declarator &D);

  /// Handles semantic checking for features that are common to all attributes,
  /// such as checking whether a parameter was properly specified, or the
  /// correct number of arguments were passed, etc. Returns true if the
  /// attribute has been diagnosed.
  bool checkCommonAttributeFeatures(const Decl *D, const ParsedAttr &A);
  bool checkCommonAttributeFeatures(const Stmt *S, const ParsedAttr &A);

  /// Determine if type T is a valid subject for a nonnull and similar
  /// attributes. By default, we look through references (the behavior used by
  /// nonnull), but if the second parameter is true, then we treat a reference
  /// type as valid.
  bool isValidPointerAttrType(QualType T, bool RefOkay = false);

  bool CheckRegparmAttr(const ParsedAttr &attr, unsigned &value);
  bool CheckCallingConvAttr(const ParsedAttr &attr, CallingConv &CC,
                            const FunctionDecl *FD = nullptr);
  bool CheckAttrTarget(const ParsedAttr &CurrAttr);
  bool CheckAttrNoArgs(const ParsedAttr &CurrAttr);
  bool checkStringLiteralArgumentAttr(const ParsedAttr &Attr, unsigned ArgNum,
                                      StringRef &Str,
                                      SourceLocation *ArgLocation = nullptr);
  llvm::Error isValidSectionSpecifier(StringRef Str);
  bool checkSectionName(SourceLocation LiteralLoc, StringRef Str);
  bool checkTargetAttr(SourceLocation LiteralLoc, StringRef Str);
  bool checkTargetClonesAttrString(SourceLocation LiteralLoc, StringRef Str,
                                   const StringLiteral *Literal,
                                   bool &HasDefault, bool &HasCommas,
                                   SmallVectorImpl<StringRef> &Strings);
  bool checkMSInheritanceAttrOnDefinition(
      CXXRecordDecl *RD, SourceRange Range, bool BestCase,
      MSInheritanceModel SemanticSpelling);

  void CheckAlignasUnderalignment(Decl *D);

  /// Adjust the calling convention of a method to be the ABI default if it
  /// wasn't specified explicitly.  This handles method types formed from
  /// function type typedefs and typename template arguments.
  void adjustMemberFunctionCC(QualType &T, bool IsStatic, bool IsCtorOrDtor,
                              SourceLocation Loc);

  // Check if there is an explicit attribute, but only look through parens.
  // The intent is to look for an attribute on the current declarator, but not
  // one that came from a typedef.
  bool hasExplicitCallingConv(QualType T);

  /// Get the outermost AttributedType node that sets a calling convention.
  /// Valid types should not have multiple attributes with different CCs.
  const AttributedType *getCallingConvAttributedType(QualType T) const;

  /// Process the attributes before creating an attributed statement. Returns
  /// the semantic attributes that have been processed.
  void ProcessStmtAttributes(Stmt *Stmt,
                             const ParsedAttributesWithRange &InAttrs,
                             SmallVectorImpl<const Attr *> &OutAttrs);

  void WarnConflictingTypedMethods(ObjCMethodDecl *Method,
                                   ObjCMethodDecl *MethodDecl,
                                   bool IsProtocolMethodDecl);

  void CheckConflictingOverridingMethod(ObjCMethodDecl *Method,
                                   ObjCMethodDecl *Overridden,
                                   bool IsProtocolMethodDecl);

  /// WarnExactTypedMethods - This routine issues a warning if method
  /// implementation declaration matches exactly that of its declaration.
  void WarnExactTypedMethods(ObjCMethodDecl *Method,
                             ObjCMethodDecl *MethodDecl,
                             bool IsProtocolMethodDecl);

  typedef llvm::SmallPtrSet<Selector, 8> SelectorSet;

  /// CheckImplementationIvars - This routine checks if the instance variables
  /// listed in the implelementation match those listed in the interface.
  void CheckImplementationIvars(ObjCImplementationDecl *ImpDecl,
                                ObjCIvarDecl **Fields, unsigned nIvars,
                                SourceLocation Loc);

  /// ImplMethodsVsClassMethods - This is main routine to warn if any method
  /// remains unimplemented in the class or category \@implementation.
  void ImplMethodsVsClassMethods(Scope *S, ObjCImplDecl* IMPDecl,
                                 ObjCContainerDecl* IDecl,
                                 bool IncompleteImpl = false);

  /// DiagnoseUnimplementedProperties - This routine warns on those properties
  /// which must be implemented by this implementation.
  void DiagnoseUnimplementedProperties(Scope *S, ObjCImplDecl* IMPDecl,
                                       ObjCContainerDecl *CDecl,
                                       bool SynthesizeProperties);

  /// Diagnose any null-resettable synthesized setters.
  void diagnoseNullResettableSynthesizedSetters(const ObjCImplDecl *impDecl);

  /// DefaultSynthesizeProperties - This routine default synthesizes all
  /// properties which must be synthesized in the class's \@implementation.
  void DefaultSynthesizeProperties(Scope *S, ObjCImplDecl *IMPDecl,
                                   ObjCInterfaceDecl *IDecl,
                                   SourceLocation AtEnd);
  void DefaultSynthesizeProperties(Scope *S, Decl *D, SourceLocation AtEnd);

  /// IvarBacksCurrentMethodAccessor - This routine returns 'true' if 'IV' is
  /// an ivar synthesized for 'Method' and 'Method' is a property accessor
  /// declared in class 'IFace'.
  bool IvarBacksCurrentMethodAccessor(ObjCInterfaceDecl *IFace,
                                      ObjCMethodDecl *Method, ObjCIvarDecl *IV);

  /// DiagnoseUnusedBackingIvarInAccessor - Issue an 'unused' warning if ivar which
  /// backs the property is not used in the property's accessor.
  void DiagnoseUnusedBackingIvarInAccessor(Scope *S,
                                           const ObjCImplementationDecl *ImplD);

  /// GetIvarBackingPropertyAccessor - If method is a property setter/getter and
  /// it property has a backing ivar, returns this ivar; otherwise, returns NULL.
  /// It also returns ivar's property on success.
  ObjCIvarDecl *GetIvarBackingPropertyAccessor(const ObjCMethodDecl *Method,
                                               const ObjCPropertyDecl *&PDecl) const;

  /// Called by ActOnProperty to handle \@property declarations in
  /// class extensions.
  ObjCPropertyDecl *HandlePropertyInClassExtension(Scope *S,
                      SourceLocation AtLoc,
                      SourceLocation LParenLoc,
                      FieldDeclarator &FD,
                      Selector GetterSel,
                      SourceLocation GetterNameLoc,
                      Selector SetterSel,
                      SourceLocation SetterNameLoc,
                      const bool isReadWrite,
                      unsigned &Attributes,
                      const unsigned AttributesAsWritten,
                      QualType T,
                      TypeSourceInfo *TSI,
                      tok::ObjCKeywordKind MethodImplKind);

  /// Called by ActOnProperty and HandlePropertyInClassExtension to
  /// handle creating the ObjcPropertyDecl for a category or \@interface.
  ObjCPropertyDecl *CreatePropertyDecl(Scope *S,
                                       ObjCContainerDecl *CDecl,
                                       SourceLocation AtLoc,
                                       SourceLocation LParenLoc,
                                       FieldDeclarator &FD,
                                       Selector GetterSel,
                                       SourceLocation GetterNameLoc,
                                       Selector SetterSel,
                                       SourceLocation SetterNameLoc,
                                       const bool isReadWrite,
                                       const unsigned Attributes,
                                       const unsigned AttributesAsWritten,
                                       QualType T,
                                       TypeSourceInfo *TSI,
                                       tok::ObjCKeywordKind MethodImplKind,
                                       DeclContext *lexicalDC = nullptr);

  /// AtomicPropertySetterGetterRules - This routine enforces the rule (via
  /// warning) when atomic property has one but not the other user-declared
  /// setter or getter.
  void AtomicPropertySetterGetterRules(ObjCImplDecl* IMPDecl,
                                       ObjCInterfaceDecl* IDecl);

  void DiagnoseOwningPropertyGetterSynthesis(const ObjCImplementationDecl *D);

  void DiagnoseMissingDesignatedInitOverrides(
                                          const ObjCImplementationDecl *ImplD,
                                          const ObjCInterfaceDecl *IFD);

  void DiagnoseDuplicateIvars(ObjCInterfaceDecl *ID, ObjCInterfaceDecl *SID);

  enum MethodMatchStrategy {
    MMS_loose,
    MMS_strict
  };

  /// MatchTwoMethodDeclarations - Checks if two methods' type match and returns
  /// true, or false, accordingly.
  bool MatchTwoMethodDeclarations(const ObjCMethodDecl *Method,
                                  const ObjCMethodDecl *PrevMethod,
                                  MethodMatchStrategy strategy = MMS_strict);

  /// MatchAllMethodDeclarations - Check methods declaraed in interface or
  /// or protocol against those declared in their implementations.
  void MatchAllMethodDeclarations(const SelectorSet &InsMap,
                                  const SelectorSet &ClsMap,
                                  SelectorSet &InsMapSeen,
                                  SelectorSet &ClsMapSeen,
                                  ObjCImplDecl* IMPDecl,
                                  ObjCContainerDecl* IDecl,
                                  bool &IncompleteImpl,
                                  bool ImmediateClass,
                                  bool WarnCategoryMethodImpl=false);

  /// CheckCategoryVsClassMethodMatches - Checks that methods implemented in
  /// category matches with those implemented in its primary class and
  /// warns each time an exact match is found.
  void CheckCategoryVsClassMethodMatches(ObjCCategoryImplDecl *CatIMP);

  /// Add the given method to the list of globally-known methods.
  void addMethodToGlobalList(ObjCMethodList *List, ObjCMethodDecl *Method);

  /// Returns default addr space for method qualifiers.
  LangAS getDefaultCXXMethodAddrSpace() const;

private:
  /// AddMethodToGlobalPool - Add an instance or factory method to the global
  /// pool. See descriptoin of AddInstanceMethodToGlobalPool.
  void AddMethodToGlobalPool(ObjCMethodDecl *Method, bool impl, bool instance);

  /// LookupMethodInGlobalPool - Returns the instance or factory method and
  /// optionally warns if there are multiple signatures.
  ObjCMethodDecl *LookupMethodInGlobalPool(Selector Sel, SourceRange R,
                                           bool receiverIdOrClass,
                                           bool instance);

public:
  /// - Returns instance or factory methods in global method pool for
  /// given selector. It checks the desired kind first, if none is found, and
  /// parameter checkTheOther is set, it then checks the other kind. If no such
  /// method or only one method is found, function returns false; otherwise, it
  /// returns true.
  bool
  CollectMultipleMethodsInGlobalPool(Selector Sel,
                                     SmallVectorImpl<ObjCMethodDecl*>& Methods,
                                     bool InstanceFirst, bool CheckTheOther,
                                     const ObjCObjectType *TypeBound = nullptr);

  bool
  AreMultipleMethodsInGlobalPool(Selector Sel, ObjCMethodDecl *BestMethod,
                                 SourceRange R, bool receiverIdOrClass,
                                 SmallVectorImpl<ObjCMethodDecl*>& Methods);

  void
  DiagnoseMultipleMethodInGlobalPool(SmallVectorImpl<ObjCMethodDecl*> &Methods,
                                     Selector Sel, SourceRange R,
                                     bool receiverIdOrClass);

private:
  /// - Returns a selector which best matches given argument list or
  /// nullptr if none could be found
  ObjCMethodDecl *SelectBestMethod(Selector Sel, MultiExprArg Args,
                                   bool IsInstance,
                                   SmallVectorImpl<ObjCMethodDecl*>& Methods);


  /// Record the typo correction failure and return an empty correction.
  TypoCorrection FailedCorrection(IdentifierInfo *Typo, SourceLocation TypoLoc,
                                  bool RecordFailure = true) {
    if (RecordFailure)
      TypoCorrectionFailures[Typo].insert(TypoLoc);
    return TypoCorrection();
  }

public:
  /// AddInstanceMethodToGlobalPool - All instance methods in a translation
  /// unit are added to a global pool. This allows us to efficiently associate
  /// a selector with a method declaraation for purposes of typechecking
  /// messages sent to "id" (where the class of the object is unknown).
  void AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method, bool impl=false) {
    AddMethodToGlobalPool(Method, impl, /*instance*/true);
  }

  /// AddFactoryMethodToGlobalPool - Same as above, but for factory methods.
  void AddFactoryMethodToGlobalPool(ObjCMethodDecl *Method, bool impl=false) {
    AddMethodToGlobalPool(Method, impl, /*instance*/false);
  }

  /// AddAnyMethodToGlobalPool - Add any method, instance or factory to global
  /// pool.
  void AddAnyMethodToGlobalPool(Decl *D);

  /// LookupInstanceMethodInGlobalPool - Returns the method and warns if
  /// there are multiple signatures.
  ObjCMethodDecl *LookupInstanceMethodInGlobalPool(Selector Sel, SourceRange R,
                                                   bool receiverIdOrClass=false) {
    return LookupMethodInGlobalPool(Sel, R, receiverIdOrClass,
                                    /*instance*/true);
  }

  /// LookupFactoryMethodInGlobalPool - Returns the method and warns if
  /// there are multiple signatures.
  ObjCMethodDecl *LookupFactoryMethodInGlobalPool(Selector Sel, SourceRange R,
                                                  bool receiverIdOrClass=false) {
    return LookupMethodInGlobalPool(Sel, R, receiverIdOrClass,
                                    /*instance*/false);
  }

  const ObjCMethodDecl *SelectorsForTypoCorrection(Selector Sel,
                              QualType ObjectType=QualType());
  /// LookupImplementedMethodInGlobalPool - Returns the method which has an
  /// implementation.
  ObjCMethodDecl *LookupImplementedMethodInGlobalPool(Selector Sel);

  /// CollectIvarsToConstructOrDestruct - Collect those ivars which require
  /// initialization.
  void CollectIvarsToConstructOrDestruct(ObjCInterfaceDecl *OI,
                                  SmallVectorImpl<ObjCIvarDecl*> &Ivars);

  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks: SemaStmt.cpp.
public:
  class FullExprArg {
  public:
    FullExprArg() : E(nullptr) { }
    FullExprArg(Sema &actions) : E(nullptr) { }

    ExprResult release() {
      return E;
    }

    Expr *get() const { return E; }

    Expr *operator->() {
      return E;
    }

  private:
    // FIXME: No need to make the entire Sema class a friend when it's just
    // Sema::MakeFullExpr that needs access to the constructor below.
    friend class Sema;

    explicit FullExprArg(Expr *expr) : E(expr) {}

    Expr *E;
  };

  FullExprArg MakeFullExpr(Expr *Arg) {
    return MakeFullExpr(Arg, Arg ? Arg->getExprLoc() : SourceLocation());
  }
  FullExprArg MakeFullExpr(Expr *Arg, SourceLocation CC) {
    return FullExprArg(
        ActOnFinishFullExpr(Arg, CC, /*DiscardedValue*/ false).get());
  }
  FullExprArg MakeFullDiscardedValueExpr(Expr *Arg) {
    ExprResult FE =
        ActOnFinishFullExpr(Arg, Arg ? Arg->getExprLoc() : SourceLocation(),
                            /*DiscardedValue*/ true);
    return FullExprArg(FE.get());
  }

  StmtResult ActOnExprStmt(ExprResult Arg, bool DiscardedValue = true);
  StmtResult ActOnExprStmtError();

  StmtResult ActOnNullStmt(SourceLocation SemiLoc,
                           bool HasLeadingEmptyMacro = false);

  void ActOnStartOfCompoundStmt(bool IsStmtExpr);
  void ActOnAfterCompoundStatementLeadingPragmas();
  void ActOnFinishOfCompoundStmt();
  StmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                               ArrayRef<Stmt *> Elts, bool isStmtExpr);

  /// A RAII object to enter scope of a compound statement.
  class CompoundScopeRAII {
  public:
    CompoundScopeRAII(Sema &S, bool IsStmtExpr = false) : S(S) {
      S.ActOnStartOfCompoundStmt(IsStmtExpr);
    }

    ~CompoundScopeRAII() {
      S.ActOnFinishOfCompoundStmt();
    }

  private:
    Sema &S;
  };

  /// An RAII helper that pops function a function scope on exit.
  struct FunctionScopeRAII {
    Sema &S;
    bool Active;
    FunctionScopeRAII(Sema &S) : S(S), Active(true) {}
    ~FunctionScopeRAII() {
      if (Active)
        S.PopFunctionScopeInfo();
    }
    void disable() { Active = false; }
  };

  StmtResult ActOnDeclStmt(DeclGroupPtrTy Decl,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc);
  void ActOnForEachDeclStmt(DeclGroupPtrTy Decl);
  StmtResult ActOnForEachLValueExpr(Expr *E);
  ExprResult ActOnCaseExpr(SourceLocation CaseLoc, ExprResult Val);
  StmtResult ActOnCaseStmt(SourceLocation CaseLoc, ExprResult LHS,
                           SourceLocation DotDotDotLoc, ExprResult RHS,
                           SourceLocation ColonLoc);
  void ActOnCaseStmtBody(Stmt *CaseStmt, Stmt *SubStmt);

  StmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc,
                                      Stmt *SubStmt, Scope *CurScope);
  StmtResult ActOnLabelStmt(SourceLocation IdentLoc, LabelDecl *TheDecl,
                            SourceLocation ColonLoc, Stmt *SubStmt);

  StmtResult BuildAttributedStmt(SourceLocation AttrsLoc,
                                 ArrayRef<const Attr *> Attrs, Stmt *SubStmt);
  StmtResult ActOnAttributedStmt(const ParsedAttributesWithRange &AttrList,
                                 Stmt *SubStmt);

  class ConditionResult;

  StmtResult ActOnIfStmt(SourceLocation IfLoc, IfStatementKind StatementKind,
                         SourceLocation LParenLoc, Stmt *InitStmt,
                         ConditionResult Cond, SourceLocation RParenLoc,
                         Stmt *ThenVal, SourceLocation ElseLoc, Stmt *ElseVal);
  StmtResult BuildIfStmt(SourceLocation IfLoc, IfStatementKind StatementKind,
                         SourceLocation LParenLoc, Stmt *InitStmt,
                         ConditionResult Cond, SourceLocation RParenLoc,
                         Stmt *ThenVal, SourceLocation ElseLoc, Stmt *ElseVal);
  StmtResult ActOnStartOfSwitchStmt(SourceLocation SwitchLoc,
                                    SourceLocation LParenLoc, Stmt *InitStmt,
                                    ConditionResult Cond,
                                    SourceLocation RParenLoc);
  StmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc,
                                           Stmt *Switch, Stmt *Body);
  StmtResult ActOnWhileStmt(SourceLocation WhileLoc, SourceLocation LParenLoc,
                            ConditionResult Cond, SourceLocation RParenLoc,
                            Stmt *Body);
  StmtResult ActOnDoStmt(SourceLocation DoLoc, Stmt *Body,
                         SourceLocation WhileLoc, SourceLocation CondLParen,
                         Expr *Cond, SourceLocation CondRParen);

  StmtResult ActOnForStmt(SourceLocation ForLoc,
                          SourceLocation LParenLoc,
                          Stmt *First,
                          ConditionResult Second,
                          FullExprArg Third,
                          SourceLocation RParenLoc,
                          Stmt *Body);
  ExprResult CheckObjCForCollectionOperand(SourceLocation forLoc,
                                           Expr *collection);
  StmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc,
                                        Stmt *First, Expr *collection,
                                        SourceLocation RParenLoc);
  StmtResult FinishObjCForCollectionStmt(Stmt *ForCollection, Stmt *Body);

  enum BuildForRangeKind {
    /// Initial building of a for-range statement.
    BFRK_Build,
    /// Instantiation or recovery rebuild of a for-range statement. Don't
    /// attempt any typo-correction.
    BFRK_Rebuild,
    /// Determining whether a for-range statement could be built. Avoid any
    /// unnecessary or irreversible actions.
    BFRK_Check
  };

  StmtResult ActOnCXXForRangeStmt(Scope *S, SourceLocation ForLoc,
                                  SourceLocation CoawaitLoc,
                                  Stmt *InitStmt,
                                  Stmt *LoopVar,
                                  SourceLocation ColonLoc, Expr *Collection,
                                  SourceLocation RParenLoc,
                                  BuildForRangeKind Kind);
  StmtResult BuildCXXForRangeStmt(SourceLocation ForLoc,
                                  SourceLocation CoawaitLoc,
                                  Stmt *InitStmt,
                                  SourceLocation ColonLoc,
                                  Stmt *RangeDecl, Stmt *Begin, Stmt *End,
                                  Expr *Cond, Expr *Inc,
                                  Stmt *LoopVarDecl,
                                  SourceLocation RParenLoc,
                                  BuildForRangeKind Kind);
  StmtResult FinishCXXForRangeStmt(Stmt *ForRange, Stmt *Body);

  StmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                           SourceLocation LabelLoc,
                           LabelDecl *TheDecl);
  StmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation StarLoc,
                                   Expr *DestExp);
  StmtResult ActOnContinueStmt(SourceLocation ContinueLoc, Scope *CurScope);
  StmtResult ActOnBreakStmt(SourceLocation BreakLoc, Scope *CurScope);

  void ActOnCapturedRegionStart(SourceLocation Loc, Scope *CurScope,
                                CapturedRegionKind Kind, unsigned NumParams);
  typedef std::pair<StringRef, QualType> CapturedParamNameType;
  void ActOnCapturedRegionStart(SourceLocation Loc, Scope *CurScope,
                                CapturedRegionKind Kind,
                                ArrayRef<CapturedParamNameType> Params,
                                unsigned OpenMPCaptureLevel = 0);
  StmtResult ActOnCapturedRegionEnd(Stmt *S);
  void ActOnCapturedRegionError();
  RecordDecl *CreateCapturedStmtRecordDecl(CapturedDecl *&CD,
                                           SourceLocation Loc,
                                           unsigned NumParams);

  struct NamedReturnInfo {
    const VarDecl *Candidate;

    enum Status : uint8_t { None, MoveEligible, MoveEligibleAndCopyElidable };
    Status S;

    bool isMoveEligible() const { return S != None; };
    bool isCopyElidable() const { return S == MoveEligibleAndCopyElidable; }
  };
  enum class SimplerImplicitMoveMode { ForceOff, Normal, ForceOn };
  NamedReturnInfo getNamedReturnInfo(
      Expr *&E, SimplerImplicitMoveMode Mode = SimplerImplicitMoveMode::Normal);
  NamedReturnInfo getNamedReturnInfo(const VarDecl *VD);
  const VarDecl *getCopyElisionCandidate(NamedReturnInfo &Info,
                                         QualType ReturnType);

  ExprResult
  PerformMoveOrCopyInitialization(const InitializedEntity &Entity,
                                  const NamedReturnInfo &NRInfo, Expr *Value,
                                  bool SupressSimplerImplicitMoves = false);

  StmtResult ActOnReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp,
                             Scope *CurScope);
  StmtResult BuildReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp,
                             bool AllowRecovery = false);
  StmtResult ActOnCapScopeReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp,
                                     NamedReturnInfo &NRInfo,
                                     bool SupressSimplerImplicitMoves);

  StmtResult ActOnGCCAsmStmt(SourceLocation AsmLoc, bool IsSimple,
                             bool IsVolatile, unsigned NumOutputs,
                             unsigned NumInputs, IdentifierInfo **Names,
                             MultiExprArg Constraints, MultiExprArg Exprs,
                             Expr *AsmString, MultiExprArg Clobbers,
                             unsigned NumLabels,
                             SourceLocation RParenLoc);

  void FillInlineAsmIdentifierInfo(Expr *Res,
                                   llvm::InlineAsmIdentifierInfo &Info);
  ExprResult LookupInlineAsmIdentifier(CXXScopeSpec &SS,
                                       SourceLocation TemplateKWLoc,
                                       UnqualifiedId &Id,
                                       bool IsUnevaluatedContext);
  bool LookupInlineAsmField(StringRef Base, StringRef Member,
                            unsigned &Offset, SourceLocation AsmLoc);
  ExprResult LookupInlineAsmVarDeclField(Expr *RefExpr, StringRef Member,
                                         SourceLocation AsmLoc);
  StmtResult ActOnMSAsmStmt(SourceLocation AsmLoc, SourceLocation LBraceLoc,
                            ArrayRef<Token> AsmToks,
                            StringRef AsmString,
                            unsigned NumOutputs, unsigned NumInputs,
                            ArrayRef<StringRef> Constraints,
                            ArrayRef<StringRef> Clobbers,
                            ArrayRef<Expr*> Exprs,
                            SourceLocation EndLoc);
  LabelDecl *GetOrCreateMSAsmLabel(StringRef ExternalLabelName,
                                   SourceLocation Location,
                                   bool AlwaysCreate);

  VarDecl *BuildObjCExceptionDecl(TypeSourceInfo *TInfo, QualType ExceptionType,
                                  SourceLocation StartLoc,
                                  SourceLocation IdLoc, IdentifierInfo *Id,
                                  bool Invalid = false);

  Decl *ActOnObjCExceptionDecl(Scope *S, Declarator &D);

  StmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc, SourceLocation RParen,
                                  Decl *Parm, Stmt *Body);

  StmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc, Stmt *Body);

  StmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc, Stmt *Try,
                                MultiStmtArg Catch, Stmt *Finally);

  StmtResult BuildObjCAtThrowStmt(SourceLocation AtLoc, Expr *Throw);
  StmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc, Expr *Throw,
                                  Scope *CurScope);
  ExprResult ActOnObjCAtSynchronizedOperand(SourceLocation atLoc,
                                            Expr *operand);
  StmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc,
                                         Expr *SynchExpr,
                                         Stmt *SynchBody);

  StmtResult ActOnObjCAutoreleasePoolStmt(SourceLocation AtLoc, Stmt *Body);

  VarDecl *BuildExceptionDeclaration(Scope *S, TypeSourceInfo *TInfo,
                                     SourceLocation StartLoc,
                                     SourceLocation IdLoc,
                                     IdentifierInfo *Id);

  Decl *ActOnExceptionDeclarator(Scope *S, Declarator &D);

  StmtResult ActOnCXXCatchBlock(SourceLocation CatchLoc,
                                Decl *ExDecl, Stmt *HandlerBlock);
  StmtResult ActOnCXXTryBlock(SourceLocation TryLoc, Stmt *TryBlock,
                              ArrayRef<Stmt *> Handlers);

  StmtResult ActOnSEHTryBlock(bool IsCXXTry, // try (true) or __try (false) ?
                              SourceLocation TryLoc, Stmt *TryBlock,
                              Stmt *Handler);
  StmtResult ActOnSEHExceptBlock(SourceLocation Loc,
                                 Expr *FilterExpr,
                                 Stmt *Block);
  void ActOnStartSEHFinallyBlock();
  void ActOnAbortSEHFinallyBlock();
  StmtResult ActOnFinishSEHFinallyBlock(SourceLocation Loc, Stmt *Block);
  StmtResult ActOnSEHLeaveStmt(SourceLocation Loc, Scope *CurScope);

  void DiagnoseReturnInConstructorExceptionHandler(CXXTryStmt *TryBlock);

  bool ShouldWarnIfUnusedFileScopedDecl(const DeclaratorDecl *D) const;

  /// If it's a file scoped decl that must warn if not used, keep track
  /// of it.
  void MarkUnusedFileScopedDecl(const DeclaratorDecl *D);

  /// DiagnoseUnusedExprResult - If the statement passed in is an expression
  /// whose result is unused, warn.
  void DiagnoseUnusedExprResult(const Stmt *S, unsigned DiagID);
  void DiagnoseUnusedNestedTypedefs(const RecordDecl *D);
  void DiagnoseUnusedDecl(const NamedDecl *ND);

  /// If VD is set but not otherwise used, diagnose, for a parameter or a
  /// variable.
  void DiagnoseUnusedButSetDecl(const VarDecl *VD);

  /// Emit \p DiagID if statement located on \p StmtLoc has a suspicious null
  /// statement as a \p Body, and it is located on the same line.
  ///
  /// This helps prevent bugs due to typos, such as:
  ///     if (condition);
  ///       do_stuff();
  void DiagnoseEmptyStmtBody(SourceLocation StmtLoc,
                             const Stmt *Body,
                             unsigned DiagID);

  /// Warn if a for/while loop statement \p S, which is followed by
  /// \p PossibleBody, has a suspicious null statement as a body.
  void DiagnoseEmptyLoopBody(const Stmt *S,
                             const Stmt *PossibleBody);

  /// Warn if a value is moved to itself.
  void DiagnoseSelfMove(const Expr *LHSExpr, const Expr *RHSExpr,
                        SourceLocation OpLoc);

  /// Warn if we're implicitly casting from a _Nullable pointer type to a
  /// _Nonnull one.
  void diagnoseNullableToNonnullConversion(QualType DstType, QualType SrcType,
                                           SourceLocation Loc);

  /// Warn when implicitly casting 0 to nullptr.
  void diagnoseZeroToNullptrConversion(CastKind Kind, const Expr *E);

  ParsingDeclState PushParsingDeclaration(sema::DelayedDiagnosticPool &pool) {
    return DelayedDiagnostics.push(pool);
  }
  void PopParsingDeclaration(ParsingDeclState state, Decl *decl);

  typedef ProcessingContextState ParsingClassState;
  ParsingClassState PushParsingClass() {
    ParsingClassDepth++;
    return DelayedDiagnostics.pushUndelayed();
  }
  void PopParsingClass(ParsingClassState state) {
    ParsingClassDepth--;
    DelayedDiagnostics.popUndelayed(state);
  }

  void redelayDiagnostics(sema::DelayedDiagnosticPool &pool);

  void DiagnoseAvailabilityOfDecl(NamedDecl *D, ArrayRef<SourceLocation> Locs,
                                  const ObjCInterfaceDecl *UnknownObjCClass,
                                  bool ObjCPropertyAccess,
                                  bool AvoidPartialAvailabilityChecks = false,
                                  ObjCInterfaceDecl *ClassReceiver = nullptr);

  bool makeUnavailableInSystemHeader(SourceLocation loc,
                                     UnavailableAttr::ImplicitReason reason);

  /// Issue any -Wunguarded-availability warnings in \c FD
  void DiagnoseUnguardedAvailabilityViolations(Decl *FD);

  void handleDelayedAvailabilityCheck(sema::DelayedDiagnostic &DD, Decl *Ctx);

  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks: SemaExpr.cpp.

  bool CanUseDecl(NamedDecl *D, bool TreatUnavailableAsInvalid);
  bool DiagnoseUseOfDecl(NamedDecl *D, ArrayRef<SourceLocation> Locs,
                         const ObjCInterfaceDecl *UnknownObjCClass = nullptr,
                         bool ObjCPropertyAccess = false,
                         bool AvoidPartialAvailabilityChecks = false,
                         ObjCInterfaceDecl *ClassReciever = nullptr);
  void NoteDeletedFunction(FunctionDecl *FD);
  void NoteDeletedInheritingConstructor(CXXConstructorDecl *CD);
  bool DiagnosePropertyAccessorMismatch(ObjCPropertyDecl *PD,
                                        ObjCMethodDecl *Getter,
                                        SourceLocation Loc);
  void DiagnoseSentinelCalls(NamedDecl *D, SourceLocation Loc,
                             ArrayRef<Expr *> Args);

  void PushExpressionEvaluationContext(
      ExpressionEvaluationContext NewContext, Decl *LambdaContextDecl = nullptr,
      ExpressionEvaluationContextRecord::ExpressionKind Type =
          ExpressionEvaluationContextRecord::EK_Other);
  enum ReuseLambdaContextDecl_t { ReuseLambdaContextDecl };
  void PushExpressionEvaluationContext(
      ExpressionEvaluationContext NewContext, ReuseLambdaContextDecl_t,
      ExpressionEvaluationContextRecord::ExpressionKind Type =
          ExpressionEvaluationContextRecord::EK_Other);
  void PopExpressionEvaluationContext();

  void DiscardCleanupsInEvaluationContext();

  ExprResult TransformToPotentiallyEvaluated(Expr *E);
  ExprResult HandleExprEvaluationContextForTypeof(Expr *E);

  ExprResult CheckUnevaluatedOperand(Expr *E);
  void CheckUnusedVolatileAssignment(Expr *E);

  ExprResult ActOnConstantExpression(ExprResult Res);

  // Functions for marking a declaration referenced.  These functions also
  // contain the relevant logic for marking if a reference to a function or
  // variable is an odr-use (in the C++11 sense).  There are separate variants
  // for expressions referring to a decl; these exist because odr-use marking
  // needs to be delayed for some constant variables when we build one of the
  // named expressions.
  //
  // MightBeOdrUse indicates whether the use could possibly be an odr-use, and
  // should usually be true. This only needs to be set to false if the lack of
  // odr-use cannot be determined from the current context (for instance,
  // because the name denotes a virtual function and was written without an
  // explicit nested-name-specifier).
  void MarkAnyDeclReferenced(SourceLocation Loc, Decl *D, bool MightBeOdrUse);
  void MarkFunctionReferenced(SourceLocation Loc, FunctionDecl *Func,
                              bool MightBeOdrUse = true);
  void MarkVariableReferenced(SourceLocation Loc, VarDecl *Var);
  void MarkDeclRefReferenced(DeclRefExpr *E, const Expr *Base = nullptr);
  void MarkMemberReferenced(MemberExpr *E);
  void MarkFunctionParmPackReferenced(FunctionParmPackExpr *E);
  void MarkCaptureUsedInEnclosingContext(VarDecl *Capture, SourceLocation Loc,
                                         unsigned CapturingScopeIndex);

  ExprResult CheckLValueToRValueConversionOperand(Expr *E);
  void CleanupVarDeclMarking();

  enum TryCaptureKind {
    TryCapture_Implicit, TryCapture_ExplicitByVal, TryCapture_ExplicitByRef
  };

  /// Try to capture the given variable.
  ///
  /// \param Var The variable to capture.
  ///
  /// \param Loc The location at which the capture occurs.
  ///
  /// \param Kind The kind of capture, which may be implicit (for either a
  /// block or a lambda), or explicit by-value or by-reference (for a lambda).
  ///
  /// \param EllipsisLoc The location of the ellipsis, if one is provided in
  /// an explicit lambda capture.
  ///
  /// \param BuildAndDiagnose Whether we are actually supposed to add the
  /// captures or diagnose errors. If false, this routine merely check whether
  /// the capture can occur without performing the capture itself or complaining
  /// if the variable cannot be captured.
  ///
  /// \param CaptureType Will be set to the type of the field used to capture
  /// this variable in the innermost block or lambda. Only valid when the
  /// variable can be captured.
  ///
  /// \param DeclRefType Will be set to the type of a reference to the capture
  /// from within the current scope. Only valid when the variable can be
  /// captured.
  ///
  /// \param FunctionScopeIndexToStopAt If non-null, it points to the index
  /// of the FunctionScopeInfo stack beyond which we do not attempt to capture.
  /// This is useful when enclosing lambdas must speculatively capture
  /// variables that may or may not be used in certain specializations of
  /// a nested generic lambda.
  ///
  /// \returns true if an error occurred (i.e., the variable cannot be
  /// captured) and false if the capture succeeded.
  bool tryCaptureVariable(VarDecl *Var, SourceLocation Loc, TryCaptureKind Kind,
                          SourceLocation EllipsisLoc, bool BuildAndDiagnose,
                          QualType &CaptureType,
                          QualType &DeclRefType,
                          const unsigned *const FunctionScopeIndexToStopAt);

  /// Try to capture the given variable.
  bool tryCaptureVariable(VarDecl *Var, SourceLocation Loc,
                          TryCaptureKind Kind = TryCapture_Implicit,
                          SourceLocation EllipsisLoc = SourceLocation());

  /// Checks if the variable must be captured.
  bool NeedToCaptureVariable(VarDecl *Var, SourceLocation Loc);

  /// Given a variable, determine the type that a reference to that
  /// variable will have in the given scope.
  QualType getCapturedDeclRefType(VarDecl *Var, SourceLocation Loc);

  /// Mark all of the declarations referenced within a particular AST node as
  /// referenced. Used when template instantiation instantiates a non-dependent
  /// type -- entities referenced by the type are now referenced.
  void MarkDeclarationsReferencedInType(SourceLocation Loc, QualType T);
  void MarkDeclarationsReferencedInExpr(Expr *E,
                                        bool SkipLocalVariables = false,
                                        ArrayRef<const Expr *> StopAt = None);

  /// Try to recover by turning the given expression into a
  /// call.  Returns true if recovery was attempted or an error was
  /// emitted; this may also leave the ExprResult invalid.
  bool tryToRecoverWithCall(ExprResult &E, const PartialDiagnostic &PD,
                            bool ForceComplain = false,
                            bool (*IsPlausibleResult)(QualType) = nullptr);

  /// Figure out if an expression could be turned into a call.
  bool tryExprAsCall(Expr &E, QualType &ZeroArgCallReturnTy,
                     UnresolvedSetImpl &NonTemplateOverloads);

  /// Try to convert an expression \p E to type \p Ty. Returns the result of the
  /// conversion.
  ExprResult tryConvertExprToType(Expr *E, QualType Ty);

  /// Conditionally issue a diagnostic based on the statements's reachability
  /// analysis.
  ///
  /// \param Stmts If Stmts is non-empty, delay reporting the diagnostic until
  /// the function body is parsed, and then do a basic reachability analysis to
  /// determine if the statement is reachable. If it is unreachable, the
  /// diagnostic will not be emitted.
  bool DiagIfReachable(SourceLocation Loc, ArrayRef<const Stmt *> Stmts,
                       const PartialDiagnostic &PD);

  /// Conditionally issue a diagnostic based on the current
  /// evaluation context.
  ///
  /// \param Statement If Statement is non-null, delay reporting the
  /// diagnostic until the function body is parsed, and then do a basic
  /// reachability analysis to determine if the statement is reachable.
  /// If it is unreachable, the diagnostic will not be emitted.
  bool DiagRuntimeBehavior(SourceLocation Loc, const Stmt *Statement,
                           const PartialDiagnostic &PD);
  /// Similar, but diagnostic is only produced if all the specified statements
  /// are reachable.
  bool DiagRuntimeBehavior(SourceLocation Loc, ArrayRef<const Stmt*> Stmts,
                           const PartialDiagnostic &PD);

  // Primary Expressions.
  SourceRange getExprRange(Expr *E) const;

  ExprResult ActOnIdExpression(
      Scope *S, CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
      UnqualifiedId &Id, bool HasTrailingLParen, bool IsAddressOfOperand,
      CorrectionCandidateCallback *CCC = nullptr,
      bool IsInlineAsmIdentifier = false, Token *KeywordReplacement = nullptr);

  void DecomposeUnqualifiedId(const UnqualifiedId &Id,
                              TemplateArgumentListInfo &Buffer,
                              DeclarationNameInfo &NameInfo,
                              const TemplateArgumentListInfo *&TemplateArgs);

  bool DiagnoseDependentMemberLookup(LookupResult &R);

  bool
  DiagnoseEmptyLookup(Scope *S, CXXScopeSpec &SS, LookupResult &R,
                      CorrectionCandidateCallback &CCC,
                      TemplateArgumentListInfo *ExplicitTemplateArgs = nullptr,
                      ArrayRef<Expr *> Args = None, TypoExpr **Out = nullptr);

  DeclResult LookupIvarInObjCMethod(LookupResult &Lookup, Scope *S,
                                    IdentifierInfo *II);
  ExprResult BuildIvarRefExpr(Scope *S, SourceLocation Loc, ObjCIvarDecl *IV);

  ExprResult LookupInObjCMethod(LookupResult &LookUp, Scope *S,
                                IdentifierInfo *II,
                                bool AllowBuiltinCreation=false);

  ExprResult ActOnDependentIdExpression(const CXXScopeSpec &SS,
                                        SourceLocation TemplateKWLoc,
                                        const DeclarationNameInfo &NameInfo,
                                        bool isAddressOfOperand,
                                const TemplateArgumentListInfo *TemplateArgs);

  /// If \p D cannot be odr-used in the current expression evaluation context,
  /// return a reason explaining why. Otherwise, return NOUR_None.
  NonOdrUseReason getNonOdrUseReasonInCurrentContext(ValueDecl *D);

  DeclRefExpr *BuildDeclRefExpr(ValueDecl *D, QualType Ty, ExprValueKind VK,
                                SourceLocation Loc,
                                const CXXScopeSpec *SS = nullptr);
  DeclRefExpr *
  BuildDeclRefExpr(ValueDecl *D, QualType Ty, ExprValueKind VK,
                   const DeclarationNameInfo &NameInfo,
                   const CXXScopeSpec *SS = nullptr,
                   NamedDecl *FoundD = nullptr,
                   SourceLocation TemplateKWLoc = SourceLocation(),
                   const TemplateArgumentListInfo *TemplateArgs = nullptr);
  DeclRefExpr *
  BuildDeclRefExpr(ValueDecl *D, QualType Ty, ExprValueKind VK,
                   const DeclarationNameInfo &NameInfo,
                   NestedNameSpecifierLoc NNS,
                   NamedDecl *FoundD = nullptr,
                   SourceLocation TemplateKWLoc = SourceLocation(),
                   const TemplateArgumentListInfo *TemplateArgs = nullptr);

  ExprResult
  BuildAnonymousStructUnionMemberReference(
      const CXXScopeSpec &SS,
      SourceLocation nameLoc,
      IndirectFieldDecl *indirectField,
      DeclAccessPair FoundDecl = DeclAccessPair::make(nullptr, AS_none),
      Expr *baseObjectExpr = nullptr,
      SourceLocation opLoc = SourceLocation());

  ExprResult BuildPossibleImplicitMemberExpr(
      const CXXScopeSpec &SS, SourceLocation TemplateKWLoc, LookupResult &R,
      const TemplateArgumentListInfo *TemplateArgs, const Scope *S,
      UnresolvedLookupExpr *AsULE = nullptr);
  ExprResult BuildImplicitMemberExpr(const CXXScopeSpec &SS,
                                     SourceLocation TemplateKWLoc,
                                     LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs,
                                     bool IsDefiniteInstance,
                                     const Scope *S);
  bool UseArgumentDependentLookup(const CXXScopeSpec &SS,
                                  const LookupResult &R,
                                  bool HasTrailingLParen);

  ExprResult
  BuildQualifiedDeclarationNameExpr(CXXScopeSpec &SS,
                                    const DeclarationNameInfo &NameInfo,
                                    bool IsAddressOfOperand, const Scope *S,
                                    TypeSourceInfo **RecoveryTSI = nullptr);

  ExprResult BuildDependentDeclRefExpr(const CXXScopeSpec &SS,
                                       SourceLocation TemplateKWLoc,
                                const DeclarationNameInfo &NameInfo,
                                const TemplateArgumentListInfo *TemplateArgs);

  ExprResult BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                      LookupResult &R,
                                      bool NeedsADL,
                                      bool AcceptInvalidDecl = false);
  ExprResult BuildDeclarationNameExpr(
      const CXXScopeSpec &SS, const DeclarationNameInfo &NameInfo, NamedDecl *D,
      NamedDecl *FoundD = nullptr,
      const TemplateArgumentListInfo *TemplateArgs = nullptr,
      bool AcceptInvalidDecl = false);

  ExprResult BuildLiteralOperatorCall(LookupResult &R,
                      DeclarationNameInfo &SuffixInfo,
                      ArrayRef<Expr *> Args,
                      SourceLocation LitEndLoc,
                      TemplateArgumentListInfo *ExplicitTemplateArgs = nullptr);

  ExprResult BuildPredefinedExpr(SourceLocation Loc,
                                 PredefinedExpr::IdentKind IK);
  ExprResult ActOnPredefinedExpr(SourceLocation Loc, tok::TokenKind Kind);
  ExprResult ActOnIntegerConstant(SourceLocation Loc, uint64_t Val);

  ExprResult BuildSYCLUniqueStableNameExpr(SourceLocation OpLoc,
                                           SourceLocation LParen,
                                           SourceLocation RParen,
                                           TypeSourceInfo *TSI);
  ExprResult ActOnSYCLUniqueStableNameExpr(SourceLocation OpLoc,
                                           SourceLocation LParen,
                                           SourceLocation RParen,
                                           ParsedType ParsedTy);

  bool CheckLoopHintExpr(Expr *E, SourceLocation Loc);

  ExprResult ActOnNumericConstant(const Token &Tok, Scope *UDLScope = nullptr);
  ExprResult ActOnCharacterConstant(const Token &Tok,
                                    Scope *UDLScope = nullptr);
  ExprResult ActOnParenExpr(SourceLocation L, SourceLocation R, Expr *E);
  ExprResult ActOnParenListExpr(SourceLocation L,
                                SourceLocation R,
                                MultiExprArg Val);

  /// ActOnStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  ExprResult ActOnStringLiteral(ArrayRef<Token> StringToks,
                                Scope *UDLScope = nullptr);

  ExprResult ActOnGenericSelectionExpr(SourceLocation KeyLoc,
                                       SourceLocation DefaultLoc,
                                       SourceLocation RParenLoc,
                                       Expr *ControllingExpr,
                                       ArrayRef<ParsedType> ArgTypes,
                                       ArrayRef<Expr *> ArgExprs);
  ExprResult CreateGenericSelectionExpr(SourceLocation KeyLoc,
                                        SourceLocation DefaultLoc,
                                        SourceLocation RParenLoc,
                                        Expr *ControllingExpr,
                                        ArrayRef<TypeSourceInfo *> Types,
                                        ArrayRef<Expr *> Exprs);

  // Binary/Unary Operators.  'Tok' is the token for the operator.
  ExprResult CreateBuiltinUnaryOp(SourceLocation OpLoc, UnaryOperatorKind Opc,
                                  Expr *InputExpr);
  ExprResult BuildUnaryOp(Scope *S, SourceLocation OpLoc,
                          UnaryOperatorKind Opc, Expr *Input);
  ExprResult ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                          tok::TokenKind Op, Expr *Input);

  bool isQualifiedMemberAccess(Expr *E);
  QualType CheckAddressOfOperand(ExprResult &Operand, SourceLocation OpLoc);

  ExprResult CreateUnaryExprOrTypeTraitExpr(TypeSourceInfo *TInfo,
                                            SourceLocation OpLoc,
                                            UnaryExprOrTypeTrait ExprKind,
                                            SourceRange R);
  ExprResult CreateUnaryExprOrTypeTraitExpr(Expr *E, SourceLocation OpLoc,
                                            UnaryExprOrTypeTrait ExprKind);
  ExprResult
    ActOnUnaryExprOrTypeTraitExpr(SourceLocation OpLoc,
                                  UnaryExprOrTypeTrait ExprKind,
                                  bool IsType, void *TyOrEx,
                                  SourceRange ArgRange);

  ExprResult CheckPlaceholderExpr(Expr *E);
  bool CheckVecStepExpr(Expr *E);

  bool CheckUnaryExprOrTypeTraitOperand(Expr *E, UnaryExprOrTypeTrait ExprKind);
  bool CheckUnaryExprOrTypeTraitOperand(QualType ExprType, SourceLocation OpLoc,
                                        SourceRange ExprRange,
                                        UnaryExprOrTypeTrait ExprKind);
  ExprResult ActOnSizeofParameterPackExpr(Scope *S,
                                          SourceLocation OpLoc,
                                          IdentifierInfo &Name,
                                          SourceLocation NameLoc,
                                          SourceLocation RParenLoc);
  ExprResult ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
                                 tok::TokenKind Kind, Expr *Input);

  ExprResult ActOnArraySubscriptExpr(Scope *S, Expr *Base, SourceLocation LLoc,
                                     Expr *Idx, SourceLocation RLoc);
  ExprResult CreateBuiltinArraySubscriptExpr(Expr *Base, SourceLocation LLoc,
                                             Expr *Idx, SourceLocation RLoc);

  ExprResult CreateBuiltinMatrixSubscriptExpr(Expr *Base, Expr *RowIdx,
                                              Expr *ColumnIdx,
                                              SourceLocation RBLoc);

  ExprResult ActOnOMPArraySectionExpr(Expr *Base, SourceLocation LBLoc,
                                      Expr *LowerBound,
                                      SourceLocation ColonLocFirst,
                                      SourceLocation ColonLocSecond,
                                      Expr *Length, Expr *Stride,
                                      SourceLocation RBLoc);
  ExprResult ActOnOMPArrayShapingExpr(Expr *Base, SourceLocation LParenLoc,
                                      SourceLocation RParenLoc,
                                      ArrayRef<Expr *> Dims,
                                      ArrayRef<SourceRange> Brackets);

  /// Data structure for iterator expression.
  struct OMPIteratorData {
    IdentifierInfo *DeclIdent = nullptr;
    SourceLocation DeclIdentLoc;
    ParsedType Type;
    OMPIteratorExpr::IteratorRange Range;
    SourceLocation AssignLoc;
    SourceLocation ColonLoc;
    SourceLocation SecColonLoc;
  };

  ExprResult ActOnOMPIteratorExpr(Scope *S, SourceLocation IteratorKwLoc,
                                  SourceLocation LLoc, SourceLocation RLoc,
                                  ArrayRef<OMPIteratorData> Data);

  // This struct is for use by ActOnMemberAccess to allow
  // BuildMemberReferenceExpr to be able to reinvoke ActOnMemberAccess after
  // changing the access operator from a '.' to a '->' (to see if that is the
  // change needed to fix an error about an unknown member, e.g. when the class
  // defines a custom operator->).
  struct ActOnMemberAccessExtraArgs {
    Scope *S;
    UnqualifiedId &Id;
    Decl *ObjCImpDecl;
  };

  ExprResult BuildMemberReferenceExpr(
      Expr *Base, QualType BaseType, SourceLocation OpLoc, bool IsArrow,
      CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
      NamedDecl *FirstQualifierInScope, const DeclarationNameInfo &NameInfo,
      const TemplateArgumentListInfo *TemplateArgs,
      const Scope *S,
      ActOnMemberAccessExtraArgs *ExtraArgs = nullptr);

  ExprResult
  BuildMemberReferenceExpr(Expr *Base, QualType BaseType, SourceLocation OpLoc,
                           bool IsArrow, const CXXScopeSpec &SS,
                           SourceLocation TemplateKWLoc,
                           NamedDecl *FirstQualifierInScope, LookupResult &R,
                           const TemplateArgumentListInfo *TemplateArgs,
                           const Scope *S,
                           bool SuppressQualifierCheck = false,
                           ActOnMemberAccessExtraArgs *ExtraArgs = nullptr);

  ExprResult BuildFieldReferenceExpr(Expr *BaseExpr, bool IsArrow,
                                     SourceLocation OpLoc,
                                     const CXXScopeSpec &SS, FieldDecl *Field,
                                     DeclAccessPair FoundDecl,
                                     const DeclarationNameInfo &MemberNameInfo);

  ExprResult PerformMemberExprBaseConversion(Expr *Base, bool IsArrow);

  bool CheckQualifiedMemberReference(Expr *BaseExpr, QualType BaseType,
                                     const CXXScopeSpec &SS,
                                     const LookupResult &R);

  ExprResult ActOnDependentMemberExpr(Expr *Base, QualType BaseType,
                                      bool IsArrow, SourceLocation OpLoc,
                                      const CXXScopeSpec &SS,
                                      SourceLocation TemplateKWLoc,
                                      NamedDecl *FirstQualifierInScope,
                               const DeclarationNameInfo &NameInfo,
                               const TemplateArgumentListInfo *TemplateArgs);

  ExprResult ActOnMemberAccessExpr(Scope *S, Expr *Base,
                                   SourceLocation OpLoc,
                                   tok::TokenKind OpKind,
                                   CXXScopeSpec &SS,
                                   SourceLocation TemplateKWLoc,
                                   UnqualifiedId &Member,
                                   Decl *ObjCImpDecl);

  MemberExpr *
  BuildMemberExpr(Expr *Base, bool IsArrow, SourceLocation OpLoc,
                  const CXXScopeSpec *SS, SourceLocation TemplateKWLoc,
                  ValueDecl *Member, DeclAccessPair FoundDecl,
                  bool HadMultipleCandidates,
                  const DeclarationNameInfo &MemberNameInfo, QualType Ty,
                  ExprValueKind VK, ExprObjectKind OK,
                  const TemplateArgumentListInfo *TemplateArgs = nullptr);
  MemberExpr *
  BuildMemberExpr(Expr *Base, bool IsArrow, SourceLocation OpLoc,
                  NestedNameSpecifierLoc NNS, SourceLocation TemplateKWLoc,
                  ValueDecl *Member, DeclAccessPair FoundDecl,
                  bool HadMultipleCandidates,
                  const DeclarationNameInfo &MemberNameInfo, QualType Ty,
                  ExprValueKind VK, ExprObjectKind OK,
                  const TemplateArgumentListInfo *TemplateArgs = nullptr);

  void ActOnDefaultCtorInitializers(Decl *CDtorDecl);
  bool ConvertArgumentsForCall(CallExpr *Call, Expr *Fn,
                               FunctionDecl *FDecl,
                               const FunctionProtoType *Proto,
                               ArrayRef<Expr *> Args,
                               SourceLocation RParenLoc,
                               bool ExecConfig = false);
  void CheckStaticArrayArgument(SourceLocation CallLoc,
                                ParmVarDecl *Param,
                                const Expr *ArgExpr);

  /// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.
  ExprResult ActOnCallExpr(Scope *S, Expr *Fn, SourceLocation LParenLoc,
                           MultiExprArg ArgExprs, SourceLocation RParenLoc,
                           Expr *ExecConfig = nullptr);
  ExprResult BuildCallExpr(Scope *S, Expr *Fn, SourceLocation LParenLoc,
                           MultiExprArg ArgExprs, SourceLocation RParenLoc,
                           Expr *ExecConfig = nullptr,
                           bool IsExecConfig = false,
                           bool AllowRecovery = false);
  Expr *BuildBuiltinCallExpr(SourceLocation Loc, Builtin::ID Id,
                             MultiExprArg CallArgs);
  enum class AtomicArgumentOrder { API, AST };
  ExprResult
  BuildAtomicExpr(SourceRange CallRange, SourceRange ExprRange,
                  SourceLocation RParenLoc, MultiExprArg Args,
                  AtomicExpr::AtomicOp Op,
                  AtomicArgumentOrder ArgOrder = AtomicArgumentOrder::API);
  ExprResult
  BuildResolvedCallExpr(Expr *Fn, NamedDecl *NDecl, SourceLocation LParenLoc,
                        ArrayRef<Expr *> Arg, SourceLocation RParenLoc,
                        Expr *Config = nullptr, bool IsExecConfig = false,
                        ADLCallKind UsesADL = ADLCallKind::NotADL);

  ExprResult ActOnCUDAExecConfigExpr(Scope *S, SourceLocation LLLLoc,
                                     MultiExprArg ExecConfig,
                                     SourceLocation GGGLoc);

  ExprResult ActOnCastExpr(Scope *S, SourceLocation LParenLoc,
                           Declarator &D, ParsedType &Ty,
                           SourceLocation RParenLoc, Expr *CastExpr);
  ExprResult BuildCStyleCastExpr(SourceLocation LParenLoc,
                                 TypeSourceInfo *Ty,
                                 SourceLocation RParenLoc,
                                 Expr *Op);
  CastKind PrepareScalarCast(ExprResult &src, QualType destType);

  /// Build an altivec or OpenCL literal.
  ExprResult BuildVectorLiteral(SourceLocation LParenLoc,
                                SourceLocation RParenLoc, Expr *E,
                                TypeSourceInfo *TInfo);

  ExprResult MaybeConvertParenListExprToParenExpr(Scope *S, Expr *ME);

  ExprResult ActOnCompoundLiteral(SourceLocation LParenLoc,
                                  ParsedType Ty,
                                  SourceLocation RParenLoc,
                                  Expr *InitExpr);

  ExprResult BuildCompoundLiteralExpr(SourceLocation LParenLoc,
                                      TypeSourceInfo *TInfo,
                                      SourceLocation RParenLoc,
                                      Expr *LiteralExpr);

  ExprResult ActOnInitList(SourceLocation LBraceLoc,
                           MultiExprArg InitArgList,
                           SourceLocation RBraceLoc);

  ExprResult BuildInitList(SourceLocation LBraceLoc,
                           MultiExprArg InitArgList,
                           SourceLocation RBraceLoc);

  ExprResult ActOnDesignatedInitializer(Designation &Desig,
                                        SourceLocation EqualOrColonLoc,
                                        bool GNUSyntax,
                                        ExprResult Init);

private:
  static BinaryOperatorKind ConvertTokenKindToBinaryOpcode(tok::TokenKind Kind);

public:
  ExprResult ActOnBinOp(Scope *S, SourceLocation TokLoc,
                        tok::TokenKind Kind, Expr *LHSExpr, Expr *RHSExpr);
  ExprResult BuildBinOp(Scope *S, SourceLocation OpLoc,
                        BinaryOperatorKind Opc, Expr *LHSExpr, Expr *RHSExpr);
  ExprResult CreateBuiltinBinOp(SourceLocation OpLoc, BinaryOperatorKind Opc,
                                Expr *LHSExpr, Expr *RHSExpr);
  void LookupBinOp(Scope *S, SourceLocation OpLoc, BinaryOperatorKind Opc,
                   UnresolvedSetImpl &Functions);

  void DiagnoseCommaOperator(const Expr *LHS, SourceLocation Loc);

  /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  ExprResult ActOnConditionalOp(SourceLocation QuestionLoc,
                                SourceLocation ColonLoc,
                                Expr *CondExpr, Expr *LHSExpr, Expr *RHSExpr);

  /// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
  ExprResult ActOnAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
                            LabelDecl *TheDecl);

  void ActOnStartStmtExpr();
  ExprResult ActOnStmtExpr(Scope *S, SourceLocation LPLoc, Stmt *SubStmt,
                           SourceLocation RPLoc);
  ExprResult BuildStmtExpr(SourceLocation LPLoc, Stmt *SubStmt,
                           SourceLocation RPLoc, unsigned TemplateDepth);
  // Handle the final expression in a statement expression.
  ExprResult ActOnStmtExprResult(ExprResult E);
  void ActOnStmtExprError();

  // __builtin_offsetof(type, identifier(.identifier|[expr])*)
  struct OffsetOfComponent {
    SourceLocation LocStart, LocEnd;
    bool isBrackets;  // true if [expr], false if .ident
    union {
      IdentifierInfo *IdentInfo;
      Expr *E;
    } U;
  };

  /// __builtin_offsetof(type, a.b[123][456].c)
  ExprResult BuildBuiltinOffsetOf(SourceLocation BuiltinLoc,
                                  TypeSourceInfo *TInfo,
                                  ArrayRef<OffsetOfComponent> Components,
                                  SourceLocation RParenLoc);
  ExprResult ActOnBuiltinOffsetOf(Scope *S,
                                  SourceLocation BuiltinLoc,
                                  SourceLocation TypeLoc,
                                  ParsedType ParsedArgTy,
                                  ArrayRef<OffsetOfComponent> Components,
                                  SourceLocation RParenLoc);

  // __builtin_choose_expr(constExpr, expr1, expr2)
  ExprResult ActOnChooseExpr(SourceLocation BuiltinLoc,
                             Expr *CondExpr, Expr *LHSExpr,
                             Expr *RHSExpr, SourceLocation RPLoc);

  // __builtin_va_arg(expr, type)
  ExprResult ActOnVAArg(SourceLocation BuiltinLoc, Expr *E, ParsedType Ty,
                        SourceLocation RPLoc);
  ExprResult BuildVAArgExpr(SourceLocation BuiltinLoc, Expr *E,
                            TypeSourceInfo *TInfo, SourceLocation RPLoc);

  // __builtin_LINE(), __builtin_FUNCTION(), __builtin_FILE(),
  // __builtin_COLUMN()
  ExprResult ActOnSourceLocExpr(SourceLocExpr::IdentKind Kind,
                                SourceLocation BuiltinLoc,
                                SourceLocation RPLoc);

  // Build a potentially resolved SourceLocExpr.
  ExprResult BuildSourceLocExpr(SourceLocExpr::IdentKind Kind,
                                SourceLocation BuiltinLoc, SourceLocation RPLoc,
                                DeclContext *ParentContext);

  // __null
  ExprResult ActOnGNUNullExpr(SourceLocation TokenLoc);

  bool CheckCaseExpression(Expr *E);

  /// Describes the result of an "if-exists" condition check.
  enum IfExistsResult {
    /// The symbol exists.
    IER_Exists,

    /// The symbol does not exist.
    IER_DoesNotExist,

    /// The name is a dependent name, so the results will differ
    /// from one instantiation to the next.
    IER_Dependent,

    /// An error occurred.
    IER_Error
  };

  IfExistsResult
  CheckMicrosoftIfExistsSymbol(Scope *S, CXXScopeSpec &SS,
                               const DeclarationNameInfo &TargetNameInfo);

  IfExistsResult
  CheckMicrosoftIfExistsSymbol(Scope *S, SourceLocation KeywordLoc,
                               bool IsIfExists, CXXScopeSpec &SS,
                               UnqualifiedId &Name);

  StmtResult BuildMSDependentExistsStmt(SourceLocation KeywordLoc,
                                        bool IsIfExists,
                                        NestedNameSpecifierLoc QualifierLoc,
                                        DeclarationNameInfo NameInfo,
                                        Stmt *Nested);
  StmtResult ActOnMSDependentExistsStmt(SourceLocation KeywordLoc,
                                        bool IsIfExists,
                                        CXXScopeSpec &SS, UnqualifiedId &Name,
                                        Stmt *Nested);

  //===------------------------- "Block" Extension ------------------------===//

  /// ActOnBlockStart - This callback is invoked when a block literal is
  /// started.
  void ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope);

  /// ActOnBlockArguments - This callback allows processing of block arguments.
  /// If there are no arguments, this is still invoked.
  void ActOnBlockArguments(SourceLocation CaretLoc, Declarator &ParamInfo,
                           Scope *CurScope);

  /// ActOnBlockError - If there is an error parsing a block, this callback
  /// is invoked to pop the information about the block from the action impl.
  void ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope);

  /// ActOnBlockStmtExpr - This is called when the body of a block statement
  /// literal was successfully completed.  ^(int x){...}
  ExprResult ActOnBlockStmtExpr(SourceLocation CaretLoc, Stmt *Body,
                                Scope *CurScope);

  //===---------------------------- Clang Extensions ----------------------===//

  /// __builtin_convertvector(...)
  ExprResult ActOnConvertVectorExpr(Expr *E, ParsedType ParsedDestTy,
                                    SourceLocation BuiltinLoc,
                                    SourceLocation RParenLoc);

  //===---------------------------- OpenCL Features -----------------------===//

  /// __builtin_astype(...)
  ExprResult ActOnAsTypeExpr(Expr *E, ParsedType ParsedDestTy,
                             SourceLocation BuiltinLoc,
                             SourceLocation RParenLoc);
  ExprResult BuildAsTypeExpr(Expr *E, QualType DestTy,
                             SourceLocation BuiltinLoc,
                             SourceLocation RParenLoc);

  //===---------------------------- C++ Features --------------------------===//

  // Act on C++ namespaces
  Decl *ActOnStartNamespaceDef(Scope *S, SourceLocation InlineLoc,
                               SourceLocation NamespaceLoc,
                               SourceLocation IdentLoc, IdentifierInfo *Ident,
                               SourceLocation LBrace,
                               const ParsedAttributesView &AttrList,
                               UsingDirectiveDecl *&UsingDecl);
  void ActOnFinishNamespaceDef(Decl *Dcl, SourceLocation RBrace);

  NamespaceDecl *getStdNamespace() const;
  NamespaceDecl *getOrCreateStdNamespace();

  NamespaceDecl *lookupStdExperimentalNamespace();
  NamespaceDecl *getCachedCoroNamespace() { return CoroTraitsNamespaceCache; }

  CXXRecordDecl *getStdBadAlloc() const;
  EnumDecl *getStdAlignValT() const;

private:
  // A cache representing if we've fully checked the various comparison category
  // types stored in ASTContext. The bit-index corresponds to the integer value
  // of a ComparisonCategoryType enumerator.
  llvm::SmallBitVector FullyCheckedComparisonCategories;

  ValueDecl *tryLookupCtorInitMemberDecl(CXXRecordDecl *ClassDecl,
                                         CXXScopeSpec &SS,
                                         ParsedType TemplateTypeTy,
                                         IdentifierInfo *MemberOrBase);

public:
  enum class ComparisonCategoryUsage {
    /// The '<=>' operator was used in an expression and a builtin operator
    /// was selected.
    OperatorInExpression,
    /// A defaulted 'operator<=>' needed the comparison category. This
    /// typically only applies to 'std::strong_ordering', due to the implicit
    /// fallback return value.
    DefaultedOperator,
  };

  /// Lookup the specified comparison category types in the standard
  ///   library, an check the VarDecls possibly returned by the operator<=>
  ///   builtins for that type.
  ///
  /// \return The type of the comparison category type corresponding to the
  ///   specified Kind, or a null type if an error occurs
  QualType CheckComparisonCategoryType(ComparisonCategoryType Kind,
                                       SourceLocation Loc,
                                       ComparisonCategoryUsage Usage);

  /// Tests whether Ty is an instance of std::initializer_list and, if
  /// it is and Element is not NULL, assigns the element type to Element.
  bool isStdInitializerList(QualType Ty, QualType *Element);

  /// Looks for the std::initializer_list template and instantiates it
  /// with Element, or emits an error if it's not found.
  ///
  /// \returns The instantiated template, or null on error.
  QualType BuildStdInitializerList(QualType Element, SourceLocation Loc);

  /// Determine whether Ctor is an initializer-list constructor, as
  /// defined in [dcl.init.list]p2.
  bool isInitListConstructor(const FunctionDecl *Ctor);

  Decl *ActOnUsingDirective(Scope *CurScope, SourceLocation UsingLoc,
                            SourceLocation NamespcLoc, CXXScopeSpec &SS,
                            SourceLocation IdentLoc,
                            IdentifierInfo *NamespcName,
                            const ParsedAttributesView &AttrList);

  void PushUsingDirective(Scope *S, UsingDirectiveDecl *UDir);

  Decl *ActOnNamespaceAliasDef(Scope *CurScope,
                               SourceLocation NamespaceLoc,
                               SourceLocation AliasLoc,
                               IdentifierInfo *Alias,
                               CXXScopeSpec &SS,
                               SourceLocation IdentLoc,
                               IdentifierInfo *Ident);

  void FilterUsingLookup(Scope *S, LookupResult &lookup);
  void HideUsingShadowDecl(Scope *S, UsingShadowDecl *Shadow);
  bool CheckUsingShadowDecl(BaseUsingDecl *BUD, NamedDecl *Target,
                            const LookupResult &PreviousDecls,
                            UsingShadowDecl *&PrevShadow);
  UsingShadowDecl *BuildUsingShadowDecl(Scope *S, BaseUsingDecl *BUD,
                                        NamedDecl *Target,
                                        UsingShadowDecl *PrevDecl);

  bool CheckUsingDeclRedeclaration(SourceLocation UsingLoc,
                                   bool HasTypenameKeyword,
                                   const CXXScopeSpec &SS,
                                   SourceLocation NameLoc,
                                   const LookupResult &Previous);
  bool CheckUsingDeclQualifier(SourceLocation UsingLoc, bool HasTypename,
                               const CXXScopeSpec &SS,
                               const DeclarationNameInfo &NameInfo,
                               SourceLocation NameLoc,
                               const LookupResult *R = nullptr,
                               const UsingDecl *UD = nullptr);

  NamedDecl *BuildUsingDeclaration(
      Scope *S, AccessSpecifier AS, SourceLocation UsingLoc,
      bool HasTypenameKeyword, SourceLocation TypenameLoc, CXXScopeSpec &SS,
      DeclarationNameInfo NameInfo, SourceLocation EllipsisLoc,
      const ParsedAttributesView &AttrList, bool IsInstantiation,
      bool IsUsingIfExists);
  NamedDecl *BuildUsingEnumDeclaration(Scope *S, AccessSpecifier AS,
                                       SourceLocation UsingLoc,
                                       SourceLocation EnumLoc,
                                       SourceLocation NameLoc, EnumDecl *ED);
  NamedDecl *BuildUsingPackDecl(NamedDecl *InstantiatedFrom,
                                ArrayRef<NamedDecl *> Expansions);

  bool CheckInheritingConstructorUsingDecl(UsingDecl *UD);

  /// Given a derived-class using shadow declaration for a constructor and the
  /// correspnding base class constructor, find or create the implicit
  /// synthesized derived class constructor to use for this initialization.
  CXXConstructorDecl *
  findInheritingConstructor(SourceLocation Loc, CXXConstructorDecl *BaseCtor,
                            ConstructorUsingShadowDecl *DerivedShadow);

  Decl *ActOnUsingDeclaration(Scope *CurScope, AccessSpecifier AS,
                              SourceLocation UsingLoc,
                              SourceLocation TypenameLoc, CXXScopeSpec &SS,
                              UnqualifiedId &Name, SourceLocation EllipsisLoc,
                              const ParsedAttributesView &AttrList);
  Decl *ActOnUsingEnumDeclaration(Scope *CurScope, AccessSpecifier AS,
                                  SourceLocation UsingLoc,
                                  SourceLocation EnumLoc, const DeclSpec &);
  Decl *ActOnAliasDeclaration(Scope *CurScope, AccessSpecifier AS,
                              MultiTemplateParamsArg TemplateParams,
                              SourceLocation UsingLoc, UnqualifiedId &Name,
                              const ParsedAttributesView &AttrList,
                              TypeResult Type, Decl *DeclFromDeclSpec);

  /// BuildCXXConstructExpr - Creates a complete call to a constructor,
  /// including handling of its default argument expressions.
  ///
  /// \param ConstructKind - a CXXConstructExpr::ConstructionKind
  ExprResult
  BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
                        NamedDecl *FoundDecl,
                        CXXConstructorDecl *Constructor, MultiExprArg Exprs,
                        bool HadMultipleCandidates, bool IsListInitialization,
                        bool IsStdInitListInitialization,
                        bool RequiresZeroInit, unsigned ConstructKind,
                        SourceRange ParenRange);

  /// Build a CXXConstructExpr whose constructor has already been resolved if
  /// it denotes an inherited constructor.
  ExprResult
  BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
                        CXXConstructorDecl *Constructor, bool Elidable,
                        MultiExprArg Exprs,
                        bool HadMultipleCandidates, bool IsListInitialization,
                        bool IsStdInitListInitialization,
                        bool RequiresZeroInit, unsigned ConstructKind,
                        SourceRange ParenRange);

  // FIXME: Can we remove this and have the above BuildCXXConstructExpr check if
  // the constructor can be elidable?
  ExprResult
  BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
                        NamedDecl *FoundDecl,
                        CXXConstructorDecl *Constructor, bool Elidable,
                        MultiExprArg Exprs, bool HadMultipleCandidates,
                        bool IsListInitialization,
                        bool IsStdInitListInitialization, bool RequiresZeroInit,
                        unsigned ConstructKind, SourceRange ParenRange);

  ExprResult BuildCXXDefaultInitExpr(SourceLocation Loc, FieldDecl *Field);


  /// Instantiate or parse a C++ default argument expression as necessary.
  /// Return true on error.
  bool CheckCXXDefaultArgExpr(SourceLocation CallLoc, FunctionDecl *FD,
                              ParmVarDecl *Param);

  /// BuildCXXDefaultArgExpr - Creates a CXXDefaultArgExpr, instantiating
  /// the default expr if needed.
  ExprResult BuildCXXDefaultArgExpr(SourceLocation CallLoc,
                                    FunctionDecl *FD,
                                    ParmVarDecl *Param);

  /// FinalizeVarWithDestructor - Prepare for calling destructor on the
  /// constructed variable.
  void FinalizeVarWithDestructor(VarDecl *VD, const RecordType *DeclInitType);

  /// Helper class that collects exception specifications for
  /// implicitly-declared special member functions.
  class ImplicitExceptionSpecification {
    // Pointer to allow copying
    Sema *Self;
    // We order exception specifications thus:
    // noexcept is the most restrictive, but is only used in C++11.
    // throw() comes next.
    // Then a throw(collected exceptions)
    // Finally no specification, which is expressed as noexcept(false).
    // throw(...) is used instead if any called function uses it.
    ExceptionSpecificationType ComputedEST;
    llvm::SmallPtrSet<CanQualType, 4> ExceptionsSeen;
    SmallVector<QualType, 4> Exceptions;

    void ClearExceptions() {
      ExceptionsSeen.clear();
      Exceptions.clear();
    }

  public:
    explicit ImplicitExceptionSpecification(Sema &Self)
      : Self(&Self), ComputedEST(EST_BasicNoexcept) {
      if (!Self.getLangOpts().CPlusPlus11)
        ComputedEST = EST_DynamicNone;
    }

    /// Get the computed exception specification type.
    ExceptionSpecificationType getExceptionSpecType() const {
      assert(!isComputedNoexcept(ComputedEST) &&
             "noexcept(expr) should not be a possible result");
      return ComputedEST;
    }

    /// The number of exceptions in the exception specification.
    unsigned size() const { return Exceptions.size(); }

    /// The set of exceptions in the exception specification.
    const QualType *data() const { return Exceptions.data(); }

    /// Integrate another called method into the collected data.
    void CalledDecl(SourceLocation CallLoc, const CXXMethodDecl *Method);

    /// Integrate an invoked expression into the collected data.
    void CalledExpr(Expr *E) { CalledStmt(E); }

    /// Integrate an invoked statement into the collected data.
    void CalledStmt(Stmt *S);

    /// Overwrite an EPI's exception specification with this
    /// computed exception specification.
    FunctionProtoType::ExceptionSpecInfo getExceptionSpec() const {
      FunctionProtoType::ExceptionSpecInfo ESI;
      ESI.Type = getExceptionSpecType();
      if (ESI.Type == EST_Dynamic) {
        ESI.Exceptions = Exceptions;
      } else if (ESI.Type == EST_None) {
        /// C++11 [except.spec]p14:
        ///   The exception-specification is noexcept(false) if the set of
        ///   potential exceptions of the special member function contains "any"
        ESI.Type = EST_NoexceptFalse;
        ESI.NoexceptExpr = Self->ActOnCXXBoolLiteral(SourceLocation(),
                                                     tok::kw_false).get();
      }
      return ESI;
    }
  };

  /// Evaluate the implicit exception specification for a defaulted
  /// special member function.
  void EvaluateImplicitExceptionSpec(SourceLocation Loc, FunctionDecl *FD);

  /// Check the given noexcept-specifier, convert its expression, and compute
  /// the appropriate ExceptionSpecificationType.
  ExprResult ActOnNoexceptSpec(Expr *NoexceptExpr,
                               ExceptionSpecificationType &EST);

  /// Check the given exception-specification and update the
  /// exception specification information with the results.
  void checkExceptionSpecification(bool IsTopLevel,
                                   ExceptionSpecificationType EST,
                                   ArrayRef<ParsedType> DynamicExceptions,
                                   ArrayRef<SourceRange> DynamicExceptionRanges,
                                   Expr *NoexceptExpr,
                                   SmallVectorImpl<QualType> &Exceptions,
                                   FunctionProtoType::ExceptionSpecInfo &ESI);

  /// Determine if we're in a case where we need to (incorrectly) eagerly
  /// parse an exception specification to work around a libstdc++ bug.
  bool isLibstdcxxEagerExceptionSpecHack(const Declarator &D);

  /// Add an exception-specification to the given member function
  /// (or member function template). The exception-specification was parsed
  /// after the method itself was declared.
  void actOnDelayedExceptionSpecification(Decl *Method,
         ExceptionSpecificationType EST,
         SourceRange SpecificationRange,
         ArrayRef<ParsedType> DynamicExceptions,
         ArrayRef<SourceRange> DynamicExceptionRanges,
         Expr *NoexceptExpr);

  class InheritedConstructorInfo;

  /// Determine if a special member function should have a deleted
  /// definition when it is defaulted.
  bool ShouldDeleteSpecialMember(CXXMethodDecl *MD, CXXSpecialMember CSM,
                                 InheritedConstructorInfo *ICI = nullptr,
                                 bool Diagnose = false);

  /// Produce notes explaining why a defaulted function was defined as deleted.
  void DiagnoseDeletedDefaultedFunction(FunctionDecl *FD);

  /// Declare the implicit default constructor for the given class.
  ///
  /// \param ClassDecl The class declaration into which the implicit
  /// default constructor will be added.
  ///
  /// \returns The implicitly-declared default constructor.
  CXXConstructorDecl *DeclareImplicitDefaultConstructor(
                                                     CXXRecordDecl *ClassDecl);

  /// DefineImplicitDefaultConstructor - Checks for feasibility of
  /// defining this constructor as the default constructor.
  void DefineImplicitDefaultConstructor(SourceLocation CurrentLocation,
                                        CXXConstructorDecl *Constructor);

  /// Declare the implicit destructor for the given class.
  ///
  /// \param ClassDecl The class declaration into which the implicit
  /// destructor will be added.
  ///
  /// \returns The implicitly-declared destructor.
  CXXDestructorDecl *DeclareImplicitDestructor(CXXRecordDecl *ClassDecl);

  /// DefineImplicitDestructor - Checks for feasibility of
  /// defining this destructor as the default destructor.
  void DefineImplicitDestructor(SourceLocation CurrentLocation,
                                CXXDestructorDecl *Destructor);

  /// Build an exception spec for destructors that don't have one.
  ///
  /// C++11 says that user-defined destructors with no exception spec get one
  /// that looks as if the destructor was implicitly declared.
  void AdjustDestructorExceptionSpec(CXXDestructorDecl *Destructor);

  /// Define the specified inheriting constructor.
  void DefineInheritingConstructor(SourceLocation UseLoc,
                                   CXXConstructorDecl *Constructor);

  /// Declare the implicit copy constructor for the given class.
  ///
  /// \param ClassDecl The class declaration into which the implicit
  /// copy constructor will be added.
  ///
  /// \returns The implicitly-declared copy constructor.
  CXXConstructorDecl *DeclareImplicitCopyConstructor(CXXRecordDecl *ClassDecl);

  /// DefineImplicitCopyConstructor - Checks for feasibility of
  /// defining this constructor as the copy constructor.
  void DefineImplicitCopyConstructor(SourceLocation CurrentLocation,
                                     CXXConstructorDecl *Constructor);

  /// Declare the implicit move constructor for the given class.
  ///
  /// \param ClassDecl The Class declaration into which the implicit
  /// move constructor will be added.
  ///
  /// \returns The implicitly-declared move constructor, or NULL if it wasn't
  /// declared.
  CXXConstructorDecl *DeclareImplicitMoveConstructor(CXXRecordDecl *ClassDecl);

  /// DefineImplicitMoveConstructor - Checks for feasibility of
  /// defining this constructor as the move constructor.
  void DefineImplicitMoveConstructor(SourceLocation CurrentLocation,
                                     CXXConstructorDecl *Constructor);

  /// Declare the implicit copy assignment operator for the given class.
  ///
  /// \param ClassDecl The class declaration into which the implicit
  /// copy assignment operator will be added.
  ///
  /// \returns The implicitly-declared copy assignment operator.
  CXXMethodDecl *DeclareImplicitCopyAssignment(CXXRecordDecl *ClassDecl);

  /// Defines an implicitly-declared copy assignment operator.
  void DefineImplicitCopyAssignment(SourceLocation CurrentLocation,
                                    CXXMethodDecl *MethodDecl);

  /// Declare the implicit move assignment operator for the given class.
  ///
  /// \param ClassDecl The Class declaration into which the implicit
  /// move assignment operator will be added.
  ///
  /// \returns The implicitly-declared move assignment operator, or NULL if it
  /// wasn't declared.
  CXXMethodDecl *DeclareImplicitMoveAssignment(CXXRecordDecl *ClassDecl);

  /// Defines an implicitly-declared move assignment operator.
  void DefineImplicitMoveAssignment(SourceLocation CurrentLocation,
                                    CXXMethodDecl *MethodDecl);

  /// Force the declaration of any implicitly-declared members of this
  /// class.
  void ForceDeclarationOfImplicitMembers(CXXRecordDecl *Class);

  /// Check a completed declaration of an implicit special member.
  void CheckImplicitSpecialMemberDeclaration(Scope *S, FunctionDecl *FD);

  /// Determine whether the given function is an implicitly-deleted
  /// special member function.
  bool isImplicitlyDeleted(FunctionDecl *FD);

  /// Check whether 'this' shows up in the type of a static member
  /// function after the (naturally empty) cv-qualifier-seq would be.
  ///
  /// \returns true if an error occurred.
  bool checkThisInStaticMemberFunctionType(CXXMethodDecl *Method);

  /// Whether this' shows up in the exception specification of a static
  /// member function.
  bool checkThisInStaticMemberFunctionExceptionSpec(CXXMethodDecl *Method);

  /// Check whether 'this' shows up in the attributes of the given
  /// static member function.
  ///
  /// \returns true if an error occurred.
  bool checkThisInStaticMemberFunctionAttributes(CXXMethodDecl *Method);

  /// MaybeBindToTemporary - If the passed in expression has a record type with
  /// a non-trivial destructor, this will return CXXBindTemporaryExpr. Otherwise
  /// it simply returns the passed in expression.
  ExprResult MaybeBindToTemporary(Expr *E);

  /// Wrap the expression in a ConstantExpr if it is a potential immediate
  /// invocation.
  ExprResult CheckForImmediateInvocation(ExprResult E, FunctionDecl *Decl);

  bool CompleteConstructorCall(CXXConstructorDecl *Constructor,
                               QualType DeclInitType, MultiExprArg ArgsPtr,
                               SourceLocation Loc,
                               SmallVectorImpl<Expr *> &ConvertedArgs,
                               bool AllowExplicit = false,
                               bool IsListInitialization = false);

  ParsedType getInheritingConstructorName(CXXScopeSpec &SS,
                                          SourceLocation NameLoc,
                                          IdentifierInfo &Name);

  ParsedType getConstructorName(IdentifierInfo &II, SourceLocation NameLoc,
                                Scope *S, CXXScopeSpec &SS,
                                bool EnteringContext);
  ParsedType getDestructorName(SourceLocation TildeLoc,
                               IdentifierInfo &II, SourceLocation NameLoc,
                               Scope *S, CXXScopeSpec &SS,
                               ParsedType ObjectType,
                               bool EnteringContext);

  ParsedType getDestructorTypeForDecltype(const DeclSpec &DS,
                                          ParsedType ObjectType);

  // Checks that reinterpret casts don't have undefined behavior.
  void CheckCompatibleReinterpretCast(QualType SrcType, QualType DestType,
                                      bool IsDereference, SourceRange Range);

  // Checks that the vector type should be initialized from a scalar
  // by splatting the value rather than populating a single element.
  // This is the case for AltiVecVector types as well as with
  // AltiVecPixel and AltiVecBool when -faltivec-src-compat=xl is specified.
  bool ShouldSplatAltivecScalarInCast(const VectorType *VecTy);

  // Checks if the -faltivec-src-compat=gcc option is specified.
  // If so, AltiVecVector, AltiVecBool and AltiVecPixel types are
  // treated the same way as they are when trying to initialize
  // these vectors on gcc (an error is emitted).
  bool CheckAltivecInitFromScalar(SourceRange R, QualType VecTy,
                                  QualType SrcTy);

  /// ActOnCXXNamedCast - Parse
  /// {dynamic,static,reinterpret,const,addrspace}_cast's.
  ExprResult ActOnCXXNamedCast(SourceLocation OpLoc,
                               tok::TokenKind Kind,
                               SourceLocation LAngleBracketLoc,
                               Declarator &D,
                               SourceLocation RAngleBracketLoc,
                               SourceLocation LParenLoc,
                               Expr *E,
                               SourceLocation RParenLoc);

  ExprResult BuildCXXNamedCast(SourceLocation OpLoc,
                               tok::TokenKind Kind,
                               TypeSourceInfo *Ty,
                               Expr *E,
                               SourceRange AngleBrackets,
                               SourceRange Parens);

  ExprResult ActOnBuiltinBitCastExpr(SourceLocation KWLoc, Declarator &Dcl,
                                     ExprResult Operand,
                                     SourceLocation RParenLoc);

  ExprResult BuildBuiltinBitCastExpr(SourceLocation KWLoc, TypeSourceInfo *TSI,
                                     Expr *Operand, SourceLocation RParenLoc);

  ExprResult BuildCXXTypeId(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            TypeSourceInfo *Operand,
                            SourceLocation RParenLoc);
  ExprResult BuildCXXTypeId(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            Expr *Operand,
                            SourceLocation RParenLoc);

  /// ActOnCXXTypeid - Parse typeid( something ).
  ExprResult ActOnCXXTypeid(SourceLocation OpLoc,
                            SourceLocation LParenLoc, bool isType,
                            void *TyOrExpr,
                            SourceLocation RParenLoc);

  ExprResult BuildCXXUuidof(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            TypeSourceInfo *Operand,
                            SourceLocation RParenLoc);
  ExprResult BuildCXXUuidof(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            Expr *Operand,
                            SourceLocation RParenLoc);

  /// ActOnCXXUuidof - Parse __uuidof( something ).
  ExprResult ActOnCXXUuidof(SourceLocation OpLoc,
                            SourceLocation LParenLoc, bool isType,
                            void *TyOrExpr,
                            SourceLocation RParenLoc);

  /// Handle a C++1z fold-expression: ( expr op ... op expr ).
  ExprResult ActOnCXXFoldExpr(Scope *S, SourceLocation LParenLoc, Expr *LHS,
                              tok::TokenKind Operator,
                              SourceLocation EllipsisLoc, Expr *RHS,
                              SourceLocation RParenLoc);
  ExprResult BuildCXXFoldExpr(UnresolvedLookupExpr *Callee,
                              SourceLocation LParenLoc, Expr *LHS,
                              BinaryOperatorKind Operator,
                              SourceLocation EllipsisLoc, Expr *RHS,
                              SourceLocation RParenLoc,
                              Optional<unsigned> NumExpansions);
  ExprResult BuildEmptyCXXFoldExpr(SourceLocation EllipsisLoc,
                                   BinaryOperatorKind Operator);

  //// ActOnCXXThis -  Parse 'this' pointer.
  ExprResult ActOnCXXThis(SourceLocation loc);

  /// Build a CXXThisExpr and mark it referenced in the current context.
  Expr *BuildCXXThisExpr(SourceLocation Loc, QualType Type, bool IsImplicit);
  void MarkThisReferenced(CXXThisExpr *This);

  /// Try to retrieve the type of the 'this' pointer.
  ///
  /// \returns The type of 'this', if possible. Otherwise, returns a NULL type.
  QualType getCurrentThisType();

  /// When non-NULL, the C++ 'this' expression is allowed despite the
  /// current context not being a non-static member function. In such cases,
  /// this provides the type used for 'this'.
  QualType CXXThisTypeOverride;

  /// RAII object used to temporarily allow the C++ 'this' expression
  /// to be used, with the given qualifiers on the current class type.
  class CXXThisScopeRAII {
    Sema &S;
    QualType OldCXXThisTypeOverride;
    bool Enabled;

  public:
    /// Introduce a new scope where 'this' may be allowed (when enabled),
    /// using the given declaration (which is either a class template or a
    /// class) along with the given qualifiers.
    /// along with the qualifiers placed on '*this'.
    CXXThisScopeRAII(Sema &S, Decl *ContextDecl, Qualifiers CXXThisTypeQuals,
                     bool Enabled = true);

    ~CXXThisScopeRAII();
  };

  /// Make sure the value of 'this' is actually available in the current
  /// context, if it is a potentially evaluated context.
  ///
  /// \param Loc The location at which the capture of 'this' occurs.
  ///
  /// \param Explicit Whether 'this' is explicitly captured in a lambda
  /// capture list.
  ///
  /// \param FunctionScopeIndexToStopAt If non-null, it points to the index
  /// of the FunctionScopeInfo stack beyond which we do not attempt to capture.
  /// This is useful when enclosing lambdas must speculatively capture
  /// 'this' that may or may not be used in certain specializations of
  /// a nested generic lambda (depending on whether the name resolves to
  /// a non-static member function or a static function).
  /// \return returns 'true' if failed, 'false' if success.
  bool CheckCXXThisCapture(SourceLocation Loc, bool Explicit = false,
      bool BuildAndDiagnose = true,
      const unsigned *const FunctionScopeIndexToStopAt = nullptr,
      bool ByCopy = false);

  /// Determine whether the given type is the type of *this that is used
  /// outside of the body of a member function for a type that is currently
  /// being defined.
  bool isThisOutsideMemberFunctionBody(QualType BaseType);

  /// ActOnCXXBoolLiteral - Parse {true,false} literals.
  ExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind);


  /// ActOnObjCBoolLiteral - Parse {__objc_yes,__objc_no} literals.
  ExprResult ActOnObjCBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind);

  ExprResult
  ActOnObjCAvailabilityCheckExpr(llvm::ArrayRef<AvailabilitySpec> AvailSpecs,
                                 SourceLocation AtLoc, SourceLocation RParen);

  /// ActOnCXXNullPtrLiteral - Parse 'nullptr'.
  ExprResult ActOnCXXNullPtrLiteral(SourceLocation Loc);

  //// ActOnCXXThrow -  Parse throw expressions.
  ExprResult ActOnCXXThrow(Scope *S, SourceLocation OpLoc, Expr *expr);
  ExprResult BuildCXXThrow(SourceLocation OpLoc, Expr *Ex,
                           bool IsThrownVarInScope);
  bool CheckCXXThrowOperand(SourceLocation ThrowLoc, QualType ThrowTy, Expr *E);

  /// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
  /// Can be interpreted either as function-style casting ("int(x)")
  /// or class type construction ("ClassType(x,y,z)")
  /// or creation of a value-initialized type ("int()").
  ExprResult ActOnCXXTypeConstructExpr(ParsedType TypeRep,
                                       SourceLocation LParenOrBraceLoc,
                                       MultiExprArg Exprs,
                                       SourceLocation RParenOrBraceLoc,
                                       bool ListInitialization);

  ExprResult BuildCXXTypeConstructExpr(TypeSourceInfo *Type,
                                       SourceLocation LParenLoc,
                                       MultiExprArg Exprs,
                                       SourceLocation RParenLoc,
                                       bool ListInitialization);

  /// ActOnCXXNew - Parsed a C++ 'new' expression.
  ExprResult ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                         SourceLocation PlacementLParen,
                         MultiExprArg PlacementArgs,
                         SourceLocation PlacementRParen,
                         SourceRange TypeIdParens, Declarator &D,
                         Expr *Initializer);
  ExprResult BuildCXXNew(SourceRange Range, bool UseGlobal,
                         SourceLocation PlacementLParen,
                         MultiExprArg PlacementArgs,
                         SourceLocation PlacementRParen,
                         SourceRange TypeIdParens,
                         QualType AllocType,
                         TypeSourceInfo *AllocTypeInfo,
                         Optional<Expr *> ArraySize,
                         SourceRange DirectInitRange,
                         Expr *Initializer);

  /// Determine whether \p FD is an aligned allocation or deallocation
  /// function that is unavailable.
  bool isUnavailableAlignedAllocationFunction(const FunctionDecl &FD) const;

  /// Produce diagnostics if \p FD is an aligned allocation or deallocation
  /// function that is unavailable.
  void diagnoseUnavailableAlignedAllocation(const FunctionDecl &FD,
                                            SourceLocation Loc);

  bool CheckAllocatedType(QualType AllocType, SourceLocation Loc,
                          SourceRange R);

  /// The scope in which to find allocation functions.
  enum AllocationFunctionScope {
    /// Only look for allocation functions in the global scope.
    AFS_Global,
    /// Only look for allocation functions in the scope of the
    /// allocated class.
    AFS_Class,
    /// Look for allocation functions in both the global scope
    /// and in the scope of the allocated class.
    AFS_Both
  };

  /// Finds the overloads of operator new and delete that are appropriate
  /// for the allocation.
  bool FindAllocationFunctions(SourceLocation StartLoc, SourceRange Range,
                               AllocationFunctionScope NewScope,
                               AllocationFunctionScope DeleteScope,
                               QualType AllocType, bool IsArray,
                               bool &PassAlignment, MultiExprArg PlaceArgs,
                               FunctionDecl *&OperatorNew,
                               FunctionDecl *&OperatorDelete,
                               bool Diagnose = true);
  void DeclareGlobalNewDelete();
  void DeclareGlobalAllocationFunction(DeclarationName Name, QualType Return,
                                       ArrayRef<QualType> Params);

  bool FindDeallocationFunction(SourceLocation StartLoc, CXXRecordDecl *RD,
                                DeclarationName Name, FunctionDecl* &Operator,
                                bool Diagnose = true);
  FunctionDecl *FindUsualDeallocationFunction(SourceLocation StartLoc,
                                              bool CanProvideSize,
                                              bool Overaligned,
                                              DeclarationName Name);
  FunctionDecl *FindDeallocationFunctionForDestructor(SourceLocation StartLoc,
                                                      CXXRecordDecl *RD);

  /// ActOnCXXDelete - Parsed a C++ 'delete' expression
  ExprResult ActOnCXXDelete(SourceLocation StartLoc,
                            bool UseGlobal, bool ArrayForm,
                            Expr *Operand);
  void CheckVirtualDtorCall(CXXDestructorDecl *dtor, SourceLocation Loc,
                            bool IsDelete, bool CallCanBeVirtual,
                            bool WarnOnNonAbstractTypes,
                            SourceLocation DtorLoc);

  ExprResult ActOnNoexceptExpr(SourceLocation KeyLoc, SourceLocation LParen,
                               Expr *Operand, SourceLocation RParen);
  ExprResult BuildCXXNoexceptExpr(SourceLocation KeyLoc, Expr *Operand,
                                  SourceLocation RParen);

  /// Parsed one of the type trait support pseudo-functions.
  ExprResult ActOnTypeTrait(TypeTrait Kind, SourceLocation KWLoc,
                            ArrayRef<ParsedType> Args,
                            SourceLocation RParenLoc);
  ExprResult BuildTypeTrait(TypeTrait Kind, SourceLocation KWLoc,
                            ArrayRef<TypeSourceInfo *> Args,
                            SourceLocation RParenLoc);

  /// ActOnArrayTypeTrait - Parsed one of the binary type trait support
  /// pseudo-functions.
  ExprResult ActOnArrayTypeTrait(ArrayTypeTrait ATT,
                                 SourceLocation KWLoc,
                                 ParsedType LhsTy,
                                 Expr *DimExpr,
                                 SourceLocation RParen);

  ExprResult BuildArrayTypeTrait(ArrayTypeTrait ATT,
                                 SourceLocation KWLoc,
                                 TypeSourceInfo *TSInfo,
                                 Expr *DimExpr,
                                 SourceLocation RParen);

  /// ActOnExpressionTrait - Parsed one of the unary type trait support
  /// pseudo-functions.
  ExprResult ActOnExpressionTrait(ExpressionTrait OET,
                                  SourceLocation KWLoc,
                                  Expr *Queried,
                                  SourceLocation RParen);

  ExprResult BuildExpressionTrait(ExpressionTrait OET,
                                  SourceLocation KWLoc,
                                  Expr *Queried,
                                  SourceLocation RParen);

  ExprResult ActOnStartCXXMemberReference(Scope *S,
                                          Expr *Base,
                                          SourceLocation OpLoc,
                                          tok::TokenKind OpKind,
                                          ParsedType &ObjectType,
                                          bool &MayBePseudoDestructor);

  ExprResult BuildPseudoDestructorExpr(Expr *Base,
                                       SourceLocation OpLoc,
                                       tok::TokenKind OpKind,
                                       const CXXScopeSpec &SS,
                                       TypeSourceInfo *ScopeType,
                                       SourceLocation CCLoc,
                                       SourceLocation TildeLoc,
                                     PseudoDestructorTypeStorage DestroyedType);

  ExprResult ActOnPseudoDestructorExpr(Scope *S, Expr *Base,
                                       SourceLocation OpLoc,
                                       tok::TokenKind OpKind,
                                       CXXScopeSpec &SS,
                                       UnqualifiedId &FirstTypeName,
                                       SourceLocation CCLoc,
                                       SourceLocation TildeLoc,
                                       UnqualifiedId &SecondTypeName);

  ExprResult ActOnPseudoDestructorExpr(Scope *S, Expr *Base,
                                       SourceLocation OpLoc,
                                       tok::TokenKind OpKind,
                                       SourceLocation TildeLoc,
                                       const DeclSpec& DS);

  /// MaybeCreateExprWithCleanups - If the current full-expression
  /// requires any cleanups, surround it with a ExprWithCleanups node.
  /// Otherwise, just returns the passed-in expression.
  Expr *MaybeCreateExprWithCleanups(Expr *SubExpr);
  Stmt *MaybeCreateStmtWithCleanups(Stmt *SubStmt);
  ExprResult MaybeCreateExprWithCleanups(ExprResult SubExpr);

  MaterializeTemporaryExpr *
  CreateMaterializeTemporaryExpr(QualType T, Expr *Temporary,
                                 bool BoundToLvalueReference);

  ExprResult ActOnFinishFullExpr(Expr *Expr, bool DiscardedValue) {
    return ActOnFinishFullExpr(
        Expr, Expr ? Expr->getExprLoc() : SourceLocation(), DiscardedValue);
  }
  ExprResult ActOnFinishFullExpr(Expr *Expr, SourceLocation CC,
                                 bool DiscardedValue, bool IsConstexpr = false);
  StmtResult ActOnFinishFullStmt(Stmt *Stmt);

  // Marks SS invalid if it represents an incomplete type.
  bool RequireCompleteDeclContext(CXXScopeSpec &SS, DeclContext *DC);
  // Complete an enum decl, maybe without a scope spec.
  bool RequireCompleteEnumDecl(EnumDecl *D, SourceLocation L,
                               CXXScopeSpec *SS = nullptr);

  DeclContext *computeDeclContext(QualType T);
  DeclContext *computeDeclContext(const CXXScopeSpec &SS,
                                  bool EnteringContext = false);
  bool isDependentScopeSpecifier(const CXXScopeSpec &SS);
  CXXRecordDecl *getCurrentInstantiationOf(NestedNameSpecifier *NNS);

  /// The parser has parsed a global nested-name-specifier '::'.
  ///
  /// \param CCLoc The location of the '::'.
  ///
  /// \param SS The nested-name-specifier, which will be updated in-place
  /// to reflect the parsed nested-name-specifier.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ActOnCXXGlobalScopeSpecifier(SourceLocation CCLoc, CXXScopeSpec &SS);

  /// The parser has parsed a '__super' nested-name-specifier.
  ///
  /// \param SuperLoc The location of the '__super' keyword.
  ///
  /// \param ColonColonLoc The location of the '::'.
  ///
  /// \param SS The nested-name-specifier, which will be updated in-place
  /// to reflect the parsed nested-name-specifier.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ActOnSuperScopeSpecifier(SourceLocation SuperLoc,
                                SourceLocation ColonColonLoc, CXXScopeSpec &SS);

  bool isAcceptableNestedNameSpecifier(const NamedDecl *SD,
                                       bool *CanCorrect = nullptr);
  NamedDecl *FindFirstQualifierInScope(Scope *S, NestedNameSpecifier *NNS);

  /// Keeps information about an identifier in a nested-name-spec.
  ///
  struct NestedNameSpecInfo {
    /// The type of the object, if we're parsing nested-name-specifier in
    /// a member access expression.
    ParsedType ObjectType;

    /// The identifier preceding the '::'.
    IdentifierInfo *Identifier;

    /// The location of the identifier.
    SourceLocation IdentifierLoc;

    /// The location of the '::'.
    SourceLocation CCLoc;

    /// Creates info object for the most typical case.
    NestedNameSpecInfo(IdentifierInfo *II, SourceLocation IdLoc,
             SourceLocation ColonColonLoc, ParsedType ObjectType = ParsedType())
      : ObjectType(ObjectType), Identifier(II), IdentifierLoc(IdLoc),
        CCLoc(ColonColonLoc) {
    }

    NestedNameSpecInfo(IdentifierInfo *II, SourceLocation IdLoc,
                       SourceLocation ColonColonLoc, QualType ObjectType)
      : ObjectType(ParsedType::make(ObjectType)), Identifier(II),
        IdentifierLoc(IdLoc), CCLoc(ColonColonLoc) {
    }
  };

  bool isNonTypeNestedNameSpecifier(Scope *S, CXXScopeSpec &SS,
                                    NestedNameSpecInfo &IdInfo);

  bool BuildCXXNestedNameSpecifier(Scope *S,
                                   NestedNameSpecInfo &IdInfo,
                                   bool EnteringContext,
                                   CXXScopeSpec &SS,
                                   NamedDecl *ScopeLookupResult,
                                   bool ErrorRecoveryLookup,
                                   bool *IsCorrectedToColon = nullptr,
                                   bool OnlyNamespace = false);

  /// The parser has parsed a nested-name-specifier 'identifier::'.
  ///
  /// \param S The scope in which this nested-name-specifier occurs.
  ///
  /// \param IdInfo Parser information about an identifier in the
  /// nested-name-spec.
  ///
  /// \param EnteringContext Whether we're entering the context nominated by
  /// this nested-name-specifier.
  ///
  /// \param SS The nested-name-specifier, which is both an input
  /// parameter (the nested-name-specifier before this type) and an
  /// output parameter (containing the full nested-name-specifier,
  /// including this new type).
  ///
  /// \param ErrorRecoveryLookup If true, then this method is called to improve
  /// error recovery. In this case do not emit error message.
  ///
  /// \param IsCorrectedToColon If not null, suggestions to replace '::' -> ':'
  /// are allowed.  The bool value pointed by this parameter is set to 'true'
  /// if the identifier is treated as if it was followed by ':', not '::'.
  ///
  /// \param OnlyNamespace If true, only considers namespaces in lookup.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ActOnCXXNestedNameSpecifier(Scope *S,
                                   NestedNameSpecInfo &IdInfo,
                                   bool EnteringContext,
                                   CXXScopeSpec &SS,
                                   bool ErrorRecoveryLookup = false,
                                   bool *IsCorrectedToColon = nullptr,
                                   bool OnlyNamespace = false);

  ExprResult ActOnDecltypeExpression(Expr *E);

  bool ActOnCXXNestedNameSpecifierDecltype(CXXScopeSpec &SS,
                                           const DeclSpec &DS,
                                           SourceLocation ColonColonLoc);

  bool IsInvalidUnlessNestedName(Scope *S, CXXScopeSpec &SS,
                                 NestedNameSpecInfo &IdInfo,
                                 bool EnteringContext);

  /// The parser has parsed a nested-name-specifier
  /// 'template[opt] template-name < template-args >::'.
  ///
  /// \param S The scope in which this nested-name-specifier occurs.
  ///
  /// \param SS The nested-name-specifier, which is both an input
  /// parameter (the nested-name-specifier before this type) and an
  /// output parameter (containing the full nested-name-specifier,
  /// including this new type).
  ///
  /// \param TemplateKWLoc the location of the 'template' keyword, if any.
  /// \param TemplateName the template name.
  /// \param TemplateNameLoc The location of the template name.
  /// \param LAngleLoc The location of the opening angle bracket  ('<').
  /// \param TemplateArgs The template arguments.
  /// \param RAngleLoc The location of the closing angle bracket  ('>').
  /// \param CCLoc The location of the '::'.
  ///
  /// \param EnteringContext Whether we're entering the context of the
  /// nested-name-specifier.
  ///
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ActOnCXXNestedNameSpecifier(Scope *S,
                                   CXXScopeSpec &SS,
                                   SourceLocation TemplateKWLoc,
                                   TemplateTy TemplateName,
                                   SourceLocation TemplateNameLoc,
                                   SourceLocation LAngleLoc,
                                   ASTTemplateArgsPtr TemplateArgs,
                                   SourceLocation RAngleLoc,
                                   SourceLocation CCLoc,
                                   bool EnteringContext);

  /// Given a C++ nested-name-specifier, produce an annotation value
  /// that the parser can use later to reconstruct the given
  /// nested-name-specifier.
  ///
  /// \param SS A nested-name-specifier.
  ///
  /// \returns A pointer containing all of the information in the
  /// nested-name-specifier \p SS.
  void *SaveNestedNameSpecifierAnnotation(CXXScopeSpec &SS);

  /// Given an annotation pointer for a nested-name-specifier, restore
  /// the nested-name-specifier structure.
  ///
  /// \param Annotation The annotation pointer, produced by
  /// \c SaveNestedNameSpecifierAnnotation().
  ///
  /// \param AnnotationRange The source range corresponding to the annotation.
  ///
  /// \param SS The nested-name-specifier that will be updated with the contents
  /// of the annotation pointer.
  void RestoreNestedNameSpecifierAnnotation(void *Annotation,
                                            SourceRange AnnotationRange,
                                            CXXScopeSpec &SS);

  bool ShouldEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

  /// ActOnCXXEnterDeclaratorScope - Called when a C++ scope specifier (global
  /// scope or nested-name-specifier) is parsed, part of a declarator-id.
  /// After this method is called, according to [C++ 3.4.3p3], names should be
  /// looked up in the declarator-id's scope, until the declarator is parsed and
  /// ActOnCXXExitDeclaratorScope is called.
  /// The 'SS' should be a non-empty valid CXXScopeSpec.
  bool ActOnCXXEnterDeclaratorScope(Scope *S, CXXScopeSpec &SS);

  /// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
  /// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
  /// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
  /// Used to indicate that names should revert to being looked up in the
  /// defining scope.
  void ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

  /// ActOnCXXEnterDeclInitializer - Invoked when we are about to parse an
  /// initializer for the declaration 'Dcl'.
  /// After this method is called, according to [C++ 3.4.1p13], if 'Dcl' is a
  /// static data member of class X, names should be looked up in the scope of
  /// class X.
  void ActOnCXXEnterDeclInitializer(Scope *S, Decl *Dcl);

  /// ActOnCXXExitDeclInitializer - Invoked after we are finished parsing an
  /// initializer for the declaration 'Dcl'.
  void ActOnCXXExitDeclInitializer(Scope *S, Decl *Dcl);

  /// Create a new lambda closure type.
  CXXRecordDecl *createLambdaClosureType(SourceRange IntroducerRange,
                                         TypeSourceInfo *Info,
                                         bool KnownDependent,
                                         LambdaCaptureDefault CaptureDefault);

  /// Start the definition of a lambda expression.
  CXXMethodDecl *startLambdaDefinition(CXXRecordDecl *Class,
                                       SourceRange IntroducerRange,
                                       TypeSourceInfo *MethodType,
                                       SourceLocation EndLoc,
                                       ArrayRef<ParmVarDecl *> Params,
                                       ConstexprSpecKind ConstexprKind,
                                       Expr *TrailingRequiresClause);

  /// Number lambda for linkage purposes if necessary.
  void handleLambdaNumbering(
      CXXRecordDecl *Class, CXXMethodDecl *Method,
      Optional<std::tuple<bool, unsigned, unsigned, Decl *>> Mangling = None);

  /// Endow the lambda scope info with the relevant properties.
  void buildLambdaScope(sema::LambdaScopeInfo *LSI,
                        CXXMethodDecl *CallOperator,
                        SourceRange IntroducerRange,
                        LambdaCaptureDefault CaptureDefault,
                        SourceLocation CaptureDefaultLoc,
                        bool ExplicitParams,
                        bool ExplicitResultType,
                        bool Mutable);

  /// Perform initialization analysis of the init-capture and perform
  /// any implicit conversions such as an lvalue-to-rvalue conversion if
  /// not being used to initialize a reference.
  ParsedType actOnLambdaInitCaptureInitialization(
      SourceLocation Loc, bool ByRef, SourceLocation EllipsisLoc,
      IdentifierInfo *Id, LambdaCaptureInitKind InitKind, Expr *&Init) {
    return ParsedType::make(buildLambdaInitCaptureInitialization(
        Loc, ByRef, EllipsisLoc, None, Id,
        InitKind != LambdaCaptureInitKind::CopyInit, Init));
  }
  QualType buildLambdaInitCaptureInitialization(
      SourceLocation Loc, bool ByRef, SourceLocation EllipsisLoc,
      Optional<unsigned> NumExpansions, IdentifierInfo *Id, bool DirectInit,
      Expr *&Init);

  /// Create a dummy variable within the declcontext of the lambda's
  ///  call operator, for name lookup purposes for a lambda init capture.
  ///
  ///  CodeGen handles emission of lambda captures, ignoring these dummy
  ///  variables appropriately.
  VarDecl *createLambdaInitCaptureVarDecl(SourceLocation Loc,
                                          QualType InitCaptureType,
                                          SourceLocation EllipsisLoc,
                                          IdentifierInfo *Id,
                                          unsigned InitStyle, Expr *Init);

  /// Add an init-capture to a lambda scope.
  void addInitCapture(sema::LambdaScopeInfo *LSI, VarDecl *Var);

  /// Note that we have finished the explicit captures for the
  /// given lambda.
  void finishLambdaExplicitCaptures(sema::LambdaScopeInfo *LSI);

  /// \brief This is called after parsing the explicit template parameter list
  /// on a lambda (if it exists) in C++2a.
  void ActOnLambdaExplicitTemplateParameterList(SourceLocation LAngleLoc,
                                                ArrayRef<NamedDecl *> TParams,
                                                SourceLocation RAngleLoc,
                                                ExprResult RequiresClause);

  /// Introduce the lambda parameters into scope.
  void addLambdaParameters(
      ArrayRef<LambdaIntroducer::LambdaCapture> Captures,
      CXXMethodDecl *CallOperator, Scope *CurScope);

  /// Deduce a block or lambda's return type based on the return
  /// statements present in the body.
  void deduceClosureReturnType(sema::CapturingScopeInfo &CSI);

  /// ActOnStartOfLambdaDefinition - This is called just before we start
  /// parsing the body of a lambda; it analyzes the explicit captures and
  /// arguments, and sets up various data-structures for the body of the
  /// lambda.
  void ActOnStartOfLambdaDefinition(LambdaIntroducer &Intro,
                                    Declarator &ParamInfo, Scope *CurScope);

  /// ActOnLambdaError - If there is an error parsing a lambda, this callback
  /// is invoked to pop the information about the lambda.
  void ActOnLambdaError(SourceLocation StartLoc, Scope *CurScope,
                        bool IsInstantiation = false);

  /// ActOnLambdaExpr - This is called when the body of a lambda expression
  /// was successfully completed.
  ExprResult ActOnLambdaExpr(SourceLocation StartLoc, Stmt *Body,
                             Scope *CurScope);

  /// Does copying/destroying the captured variable have side effects?
  bool CaptureHasSideEffects(const sema::Capture &From);

  /// Diagnose if an explicit lambda capture is unused. Returns true if a
  /// diagnostic is emitted.
  bool DiagnoseUnusedLambdaCapture(SourceRange CaptureRange,
                                   const sema::Capture &From);

  /// Build a FieldDecl suitable to hold the given capture.
  FieldDecl *BuildCaptureField(RecordDecl *RD, const sema::Capture &Capture);

  /// Initialize the given capture with a suitable expression.
  ExprResult BuildCaptureInit(const sema::Capture &Capture,
                              SourceLocation ImplicitCaptureLoc,
                              bool IsOpenMPMapping = false);

  /// Complete a lambda-expression having processed and attached the
  /// lambda body.
  ExprResult BuildLambdaExpr(SourceLocation StartLoc, SourceLocation EndLoc,
                             sema::LambdaScopeInfo *LSI);

  /// Get the return type to use for a lambda's conversion function(s) to
  /// function pointer type, given the type of the call operator.
  QualType
  getLambdaConversionFunctionResultType(const FunctionProtoType *CallOpType,
                                        CallingConv CC);

  /// Define the "body" of the conversion from a lambda object to a
  /// function pointer.
  ///
  /// This routine doesn't actually define a sensible body; rather, it fills
  /// in the initialization expression needed to copy the lambda object into
  /// the block, and IR generation actually generates the real body of the
  /// block pointer conversion.
  void DefineImplicitLambdaToFunctionPointerConversion(
         SourceLocation CurrentLoc, CXXConversionDecl *Conv);

  /// Define the "body" of the conversion from a lambda object to a
  /// block pointer.
  ///
  /// This routine doesn't actually define a sensible body; rather, it fills
  /// in the initialization expression needed to copy the lambda object into
  /// the block, and IR generation actually generates the real body of the
  /// block pointer conversion.
  void DefineImplicitLambdaToBlockPointerConversion(SourceLocation CurrentLoc,
                                                    CXXConversionDecl *Conv);

  ExprResult BuildBlockForLambdaConversion(SourceLocation CurrentLocation,
                                           SourceLocation ConvLocation,
                                           CXXConversionDecl *Conv,
                                           Expr *Src);

  /// Check whether the given expression is a valid constraint expression.
  /// A diagnostic is emitted if it is not, false is returned, and
  /// PossibleNonPrimary will be set to true if the failure might be due to a
  /// non-primary expression being used as an atomic constraint.
  bool CheckConstraintExpression(const Expr *CE, Token NextToken = Token(),
                                 bool *PossibleNonPrimary = nullptr,
                                 bool IsTrailingRequiresClause = false);

private:
  /// Caches pairs of template-like decls whose associated constraints were
  /// checked for subsumption and whether or not the first's constraints did in
  /// fact subsume the second's.
  llvm::DenseMap<std::pair<NamedDecl *, NamedDecl *>, bool> SubsumptionCache;
  /// Caches the normalized associated constraints of declarations (concepts or
  /// constrained declarations). If an error occurred while normalizing the
  /// associated constraints of the template or concept, nullptr will be cached
  /// here.
  llvm::DenseMap<NamedDecl *, NormalizedConstraint *>
      NormalizationCache;

  llvm::ContextualFoldingSet<ConstraintSatisfaction, const ASTContext &>
      SatisfactionCache;

public:
  const NormalizedConstraint *
  getNormalizedAssociatedConstraints(
      NamedDecl *ConstrainedDecl, ArrayRef<const Expr *> AssociatedConstraints);

  /// \brief Check whether the given declaration's associated constraints are
  /// at least as constrained than another declaration's according to the
  /// partial ordering of constraints.
  ///
  /// \param Result If no error occurred, receives the result of true if D1 is
  /// at least constrained than D2, and false otherwise.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool IsAtLeastAsConstrained(NamedDecl *D1, ArrayRef<const Expr *> AC1,
                              NamedDecl *D2, ArrayRef<const Expr *> AC2,
                              bool &Result);

  /// If D1 was not at least as constrained as D2, but would've been if a pair
  /// of atomic constraints involved had been declared in a concept and not
  /// repeated in two separate places in code.
  /// \returns true if such a diagnostic was emitted, false otherwise.
  bool MaybeEmitAmbiguousAtomicConstraintsDiagnostic(NamedDecl *D1,
      ArrayRef<const Expr *> AC1, NamedDecl *D2, ArrayRef<const Expr *> AC2);

  /// \brief Check whether the given list of constraint expressions are
  /// satisfied (as if in a 'conjunction') given template arguments.
  /// \param Template the template-like entity that triggered the constraints
  /// check (either a concept or a constrained entity).
  /// \param ConstraintExprs a list of constraint expressions, treated as if
  /// they were 'AND'ed together.
  /// \param TemplateArgs the list of template arguments to substitute into the
  /// constraint expression.
  /// \param TemplateIDRange The source range of the template id that
  /// caused the constraints check.
  /// \param Satisfaction if true is returned, will contain details of the
  /// satisfaction, with enough information to diagnose an unsatisfied
  /// expression.
  /// \returns true if an error occurred and satisfaction could not be checked,
  /// false otherwise.
  bool CheckConstraintSatisfaction(
      const NamedDecl *Template, ArrayRef<const Expr *> ConstraintExprs,
      ArrayRef<TemplateArgument> TemplateArgs,
      SourceRange TemplateIDRange, ConstraintSatisfaction &Satisfaction);

  /// \brief Check whether the given non-dependent constraint expression is
  /// satisfied. Returns false and updates Satisfaction with the satisfaction
  /// verdict if successful, emits a diagnostic and returns true if an error
  /// occured and satisfaction could not be determined.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool CheckConstraintSatisfaction(const Expr *ConstraintExpr,
                                   ConstraintSatisfaction &Satisfaction);

  /// Check whether the given function decl's trailing requires clause is
  /// satisfied, if any. Returns false and updates Satisfaction with the
  /// satisfaction verdict if successful, emits a diagnostic and returns true if
  /// an error occured and satisfaction could not be determined.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool CheckFunctionConstraints(const FunctionDecl *FD,
                                ConstraintSatisfaction &Satisfaction,
                                SourceLocation UsageLoc = SourceLocation());


  /// \brief Ensure that the given template arguments satisfy the constraints
  /// associated with the given template, emitting a diagnostic if they do not.
  ///
  /// \param Template The template to which the template arguments are being
  /// provided.
  ///
  /// \param TemplateArgs The converted, canonicalized template arguments.
  ///
  /// \param TemplateIDRange The source range of the template id that
  /// caused the constraints check.
  ///
  /// \returns true if the constrains are not satisfied or could not be checked
  /// for satisfaction, false if the constraints are satisfied.
  bool EnsureTemplateArgumentListConstraints(TemplateDecl *Template,
                                       ArrayRef<TemplateArgument> TemplateArgs,
                                             SourceRange TemplateIDRange);

  /// \brief Emit diagnostics explaining why a constraint expression was deemed
  /// unsatisfied.
  /// \param First whether this is the first time an unsatisfied constraint is
  /// diagnosed for this error.
  void
  DiagnoseUnsatisfiedConstraint(const ConstraintSatisfaction &Satisfaction,
                                bool First = true);

  /// \brief Emit diagnostics explaining why a constraint expression was deemed
  /// unsatisfied.
  void
  DiagnoseUnsatisfiedConstraint(const ASTConstraintSatisfaction &Satisfaction,
                                bool First = true);

  // ParseObjCStringLiteral - Parse Objective-C string literals.
  ExprResult ParseObjCStringLiteral(SourceLocation *AtLocs,
                                    ArrayRef<Expr *> Strings);

  ExprResult BuildObjCStringLiteral(SourceLocation AtLoc, StringLiteral *S);

  /// BuildObjCNumericLiteral - builds an ObjCBoxedExpr AST node for the
  /// numeric literal expression. Type of the expression will be "NSNumber *"
  /// or "id" if NSNumber is unavailable.
  ExprResult BuildObjCNumericLiteral(SourceLocation AtLoc, Expr *Number);
  ExprResult ActOnObjCBoolLiteral(SourceLocation AtLoc, SourceLocation ValueLoc,
                                  bool Value);
  ExprResult BuildObjCArrayLiteral(SourceRange SR, MultiExprArg Elements);

  /// BuildObjCBoxedExpr - builds an ObjCBoxedExpr AST node for the
  /// '@' prefixed parenthesized expression. The type of the expression will
  /// either be "NSNumber *", "NSString *" or "NSValue *" depending on the type
  /// of ValueType, which is allowed to be a built-in numeric type, "char *",
  /// "const char *" or C structure with attribute 'objc_boxable'.
  ExprResult BuildObjCBoxedExpr(SourceRange SR, Expr *ValueExpr);

  ExprResult BuildObjCSubscriptExpression(SourceLocation RB, Expr *BaseExpr,
                                          Expr *IndexExpr,
                                          ObjCMethodDecl *getterMethod,
                                          ObjCMethodDecl *setterMethod);

  ExprResult BuildObjCDictionaryLiteral(SourceRange SR,
                               MutableArrayRef<ObjCDictionaryElement> Elements);

  ExprResult BuildObjCEncodeExpression(SourceLocation AtLoc,
                                  TypeSourceInfo *EncodedTypeInfo,
                                  SourceLocation RParenLoc);
  ExprResult BuildCXXMemberCallExpr(Expr *Exp, NamedDecl *FoundDecl,
                                    CXXConversionDecl *Method,
                                    bool HadMultipleCandidates);

  ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc,
                                       SourceLocation EncodeLoc,
                                       SourceLocation LParenLoc,
                                       ParsedType Ty,
                                       SourceLocation RParenLoc);

  /// ParseObjCSelectorExpression - Build selector expression for \@selector
  ExprResult ParseObjCSelectorExpression(Selector Sel,
                                         SourceLocation AtLoc,
                                         SourceLocation SelLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation RParenLoc,
                                         bool WarnMultipleSelectors);

  /// ParseObjCProtocolExpression - Build protocol expression for \@protocol
  ExprResult ParseObjCProtocolExpression(IdentifierInfo * ProtocolName,
                                         SourceLocation AtLoc,
                                         SourceLocation ProtoLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation ProtoIdLoc,
                                         SourceLocation RParenLoc);

  //===--------------------------------------------------------------------===//
  // C++ Declarations
  //
  Decl *ActOnStartLinkageSpecification(Scope *S,
                                       SourceLocation ExternLoc,
                                       Expr *LangStr,
                                       SourceLocation LBraceLoc);
  Decl *ActOnFinishLinkageSpecification(Scope *S,
                                        Decl *LinkageSpec,
                                        SourceLocation RBraceLoc);


  //===--------------------------------------------------------------------===//
  // C++ Classes
  //
  CXXRecordDecl *getCurrentClass(Scope *S, const CXXScopeSpec *SS);
  bool isCurrentClassName(const IdentifierInfo &II, Scope *S,
                          const CXXScopeSpec *SS = nullptr);
  bool isCurrentClassNameTypo(IdentifierInfo *&II, const CXXScopeSpec *SS);

  bool ActOnAccessSpecifier(AccessSpecifier Access, SourceLocation ASLoc,
                            SourceLocation ColonLoc,
                            const ParsedAttributesView &Attrs);

  NamedDecl *ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS,
                                 Declarator &D,
                                 MultiTemplateParamsArg TemplateParameterLists,
                                 Expr *BitfieldWidth, const VirtSpecifiers &VS,
                                 InClassInitStyle InitStyle);

  void ActOnStartCXXInClassMemberInitializer();
  void ActOnFinishCXXInClassMemberInitializer(Decl *VarDecl,
                                              SourceLocation EqualLoc,
                                              Expr *Init);

  MemInitResult ActOnMemInitializer(Decl *ConstructorD,
                                    Scope *S,
                                    CXXScopeSpec &SS,
                                    IdentifierInfo *MemberOrBase,
                                    ParsedType TemplateTypeTy,
                                    const DeclSpec &DS,
                                    SourceLocation IdLoc,
                                    SourceLocation LParenLoc,
                                    ArrayRef<Expr *> Args,
                                    SourceLocation RParenLoc,
                                    SourceLocation EllipsisLoc);

  MemInitResult ActOnMemInitializer(Decl *ConstructorD,
                                    Scope *S,
                                    CXXScopeSpec &SS,
                                    IdentifierInfo *MemberOrBase,
                                    ParsedType TemplateTypeTy,
                                    const DeclSpec &DS,
                                    SourceLocation IdLoc,
                                    Expr *InitList,
                                    SourceLocation EllipsisLoc);

  MemInitResult BuildMemInitializer(Decl *ConstructorD,
                                    Scope *S,
                                    CXXScopeSpec &SS,
                                    IdentifierInfo *MemberOrBase,
                                    ParsedType TemplateTypeTy,
                                    const DeclSpec &DS,
                                    SourceLocation IdLoc,
                                    Expr *Init,
                                    SourceLocation EllipsisLoc);

  MemInitResult BuildMemberInitializer(ValueDecl *Member,
                                       Expr *Init,
                                       SourceLocation IdLoc);

  MemInitResult BuildBaseInitializer(QualType BaseType,
                                     TypeSourceInfo *BaseTInfo,
                                     Expr *Init,
                                     CXXRecordDecl *ClassDecl,
                                     SourceLocation EllipsisLoc);

  MemInitResult BuildDelegatingInitializer(TypeSourceInfo *TInfo,
                                           Expr *Init,
                                           CXXRecordDecl *ClassDecl);

  bool SetDelegatingInitializer(CXXConstructorDecl *Constructor,
                                CXXCtorInitializer *Initializer);

  bool SetCtorInitializers(CXXConstructorDecl *Constructor, bool AnyErrors,
                           ArrayRef<CXXCtorInitializer *> Initializers = None);

  void SetIvarInitializers(ObjCImplementationDecl *ObjCImplementation);


  /// MarkBaseAndMemberDestructorsReferenced - Given a record decl,
  /// mark all the non-trivial destructors of its members and bases as
  /// referenced.
  void MarkBaseAndMemberDestructorsReferenced(SourceLocation Loc,
                                              CXXRecordDecl *Record);

  /// Mark destructors of virtual bases of this class referenced. In the Itanium
  /// C++ ABI, this is done when emitting a destructor for any non-abstract
  /// class. In the Microsoft C++ ABI, this is done any time a class's
  /// destructor is referenced.
  void MarkVirtualBaseDestructorsReferenced(
      SourceLocation Location, CXXRecordDecl *ClassDecl,
      llvm::SmallPtrSetImpl<const RecordType *> *DirectVirtualBases = nullptr);

  /// Do semantic checks to allow the complete destructor variant to be emitted
  /// when the destructor is defined in another translation unit. In the Itanium
  /// C++ ABI, destructor variants are emitted together. In the MS C++ ABI, they
  /// can be emitted in separate TUs. To emit the complete variant, run a subset
  /// of the checks performed when emitting a regular destructor.
  void CheckCompleteDestructorVariant(SourceLocation CurrentLocation,
                                      CXXDestructorDecl *Dtor);

  /// The list of classes whose vtables have been used within
  /// this translation unit, and the source locations at which the
  /// first use occurred.
  typedef std::pair<CXXRecordDecl*, SourceLocation> VTableUse;

  /// The list of vtables that are required but have not yet been
  /// materialized.
  SmallVector<VTableUse, 16> VTableUses;

  /// The set of classes whose vtables have been used within
  /// this translation unit, and a bit that will be true if the vtable is
  /// required to be emitted (otherwise, it should be emitted only if needed
  /// by code generation).
  llvm::DenseMap<CXXRecordDecl *, bool> VTablesUsed;

  /// Load any externally-stored vtable uses.
  void LoadExternalVTableUses();

  /// Note that the vtable for the given class was used at the
  /// given location.
  void MarkVTableUsed(SourceLocation Loc, CXXRecordDecl *Class,
                      bool DefinitionRequired = false);

  /// Mark the exception specifications of all virtual member functions
  /// in the given class as needed.
  void MarkVirtualMemberExceptionSpecsNeeded(SourceLocation Loc,
                                             const CXXRecordDecl *RD);

  /// MarkVirtualMembersReferenced - Will mark all members of the given
  /// CXXRecordDecl referenced.
  void MarkVirtualMembersReferenced(SourceLocation Loc, const CXXRecordDecl *RD,
                                    bool ConstexprOnly = false);

  /// Define all of the vtables that have been used in this
  /// translation unit and reference any virtual members used by those
  /// vtables.
  ///
  /// \returns true if any work was done, false otherwise.
  bool DefineUsedVTables();

  void AddImplicitlyDeclaredMembersToClass(CXXRecordDecl *ClassDecl);

  void ActOnMemInitializers(Decl *ConstructorDecl,
                            SourceLocation ColonLoc,
                            ArrayRef<CXXCtorInitializer*> MemInits,
                            bool AnyErrors);

  /// Check class-level dllimport/dllexport attribute. The caller must
  /// ensure that referenceDLLExportedClassMethods is called some point later
  /// when all outer classes of Class are complete.
  void checkClassLevelDLLAttribute(CXXRecordDecl *Class);
  void checkClassLevelCodeSegAttribute(CXXRecordDecl *Class);

  void referenceDLLExportedClassMethods();

  void propagateDLLAttrToBaseClassTemplate(
      CXXRecordDecl *Class, Attr *ClassAttr,
      ClassTemplateSpecializationDecl *BaseTemplateSpec,
      SourceLocation BaseLoc);

  /// Add gsl::Pointer attribute to std::container::iterator
  /// \param ND The declaration that introduces the name
  /// std::container::iterator. \param UnderlyingRecord The record named by ND.
  void inferGslPointerAttribute(NamedDecl *ND, CXXRecordDecl *UnderlyingRecord);

  /// Add [[gsl::Owner]] and [[gsl::Pointer]] attributes for std:: types.
  void inferGslOwnerPointerAttribute(CXXRecordDecl *Record);

  /// Add [[gsl::Pointer]] attributes for std:: types.
  void inferGslPointerAttribute(TypedefNameDecl *TD);

  void CheckCompletedCXXClass(Scope *S, CXXRecordDecl *Record);

  /// Check that the C++ class annoated with "trivial_abi" satisfies all the
  /// conditions that are needed for the attribute to have an effect.
  void checkIllFormedTrivialABIStruct(CXXRecordDecl &RD);

  void ActOnFinishCXXMemberSpecification(Scope *S, SourceLocation RLoc,
                                         Decl *TagDecl, SourceLocation LBrac,
                                         SourceLocation RBrac,
                                         const ParsedAttributesView &AttrList);
  void ActOnFinishCXXMemberDecls();
  void ActOnFinishCXXNonNestedClass();

  void ActOnReenterCXXMethodParameter(Scope *S, ParmVarDecl *Param);
  unsigned ActOnReenterTemplateScope(Decl *Template,
                                     llvm::function_ref<Scope *()> EnterScope);
  void ActOnStartDelayedMemberDeclarations(Scope *S, Decl *Record);
  void ActOnStartDelayedCXXMethodDeclaration(Scope *S, Decl *Method);
  void ActOnDelayedCXXMethodParameter(Scope *S, Decl *Param);
  void ActOnFinishDelayedMemberDeclarations(Scope *S, Decl *Record);
  void ActOnFinishDelayedCXXMethodDeclaration(Scope *S, Decl *Method);
  void ActOnFinishDelayedMemberInitializers(Decl *Record);
  void MarkAsLateParsedTemplate(FunctionDecl *FD, Decl *FnD,
                                CachedTokens &Toks);
  void UnmarkAsLateParsedTemplate(FunctionDecl *FD);
  bool IsInsideALocalClassWithinATemplateFunction();

  Decl *ActOnStaticAssertDeclaration(SourceLocation StaticAssertLoc,
                                     Expr *AssertExpr,
                                     Expr *AssertMessageExpr,
                                     SourceLocation RParenLoc);
  Decl *BuildStaticAssertDeclaration(SourceLocation StaticAssertLoc,
                                     Expr *AssertExpr,
                                     StringLiteral *AssertMessageExpr,
                                     SourceLocation RParenLoc,
                                     bool Failed);

  FriendDecl *CheckFriendTypeDecl(SourceLocation LocStart,
                                  SourceLocation FriendLoc,
                                  TypeSourceInfo *TSInfo);
  Decl *ActOnFriendTypeDecl(Scope *S, const DeclSpec &DS,
                            MultiTemplateParamsArg TemplateParams);
  NamedDecl *ActOnFriendFunctionDecl(Scope *S, Declarator &D,
                                     MultiTemplateParamsArg TemplateParams);

  QualType CheckConstructorDeclarator(Declarator &D, QualType R,
                                      StorageClass& SC);
  void CheckConstructor(CXXConstructorDecl *Constructor);
  QualType CheckDestructorDeclarator(Declarator &D, QualType R,
                                     StorageClass& SC);
  bool CheckDestructor(CXXDestructorDecl *Destructor);
  void CheckConversionDeclarator(Declarator &D, QualType &R,
                                 StorageClass& SC);
  Decl *ActOnConversionDeclarator(CXXConversionDecl *Conversion);
  void CheckDeductionGuideDeclarator(Declarator &D, QualType &R,
                                     StorageClass &SC);
  void CheckDeductionGuideTemplate(FunctionTemplateDecl *TD);

  void CheckExplicitlyDefaultedFunction(Scope *S, FunctionDecl *MD);

  bool CheckExplicitlyDefaultedSpecialMember(CXXMethodDecl *MD,
                                             CXXSpecialMember CSM);
  void CheckDelayedMemberExceptionSpecs();

  bool CheckExplicitlyDefaultedComparison(Scope *S, FunctionDecl *MD,
                                          DefaultedComparisonKind DCK);
  void DeclareImplicitEqualityComparison(CXXRecordDecl *RD,
                                         FunctionDecl *Spaceship);
  void DefineDefaultedComparison(SourceLocation Loc, FunctionDecl *FD,
                                 DefaultedComparisonKind DCK);

  //===--------------------------------------------------------------------===//
  // C++ Derived Classes
  //

  /// ActOnBaseSpecifier - Parsed a base specifier
  CXXBaseSpecifier *CheckBaseSpecifier(CXXRecordDecl *Class,
                                       SourceRange SpecifierRange,
                                       bool Virtual, AccessSpecifier Access,
                                       TypeSourceInfo *TInfo,
                                       SourceLocation EllipsisLoc);

  BaseResult ActOnBaseSpecifier(Decl *classdecl,
                                SourceRange SpecifierRange,
                                ParsedAttributes &Attrs,
                                bool Virtual, AccessSpecifier Access,
                                ParsedType basetype,
                                SourceLocation BaseLoc,
                                SourceLocation EllipsisLoc);

  bool AttachBaseSpecifiers(CXXRecordDecl *Class,
                            MutableArrayRef<CXXBaseSpecifier *> Bases);
  void ActOnBaseSpecifiers(Decl *ClassDecl,
                           MutableArrayRef<CXXBaseSpecifier *> Bases);

  bool IsDerivedFrom(SourceLocation Loc, QualType Derived, QualType Base);
  bool IsDerivedFrom(SourceLocation Loc, QualType Derived, QualType Base,
                     CXXBasePaths &Paths);

  // FIXME: I don't like this name.
  void BuildBasePathArray(const CXXBasePaths &Paths, CXXCastPath &BasePath);

  bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                    SourceLocation Loc, SourceRange Range,
                                    CXXCastPath *BasePath = nullptr,
                                    bool IgnoreAccess = false);
  bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                    unsigned InaccessibleBaseID,
                                    unsigned AmbiguousBaseConvID,
                                    SourceLocation Loc, SourceRange Range,
                                    DeclarationName Name,
                                    CXXCastPath *BasePath,
                                    bool IgnoreAccess = false);

  std::string getAmbiguousPathsDisplayString(CXXBasePaths &Paths);

  bool CheckOverridingFunctionAttributes(const CXXMethodDecl *New,
                                         const CXXMethodDecl *Old);

  /// CheckOverridingFunctionReturnType - Checks whether the return types are
  /// covariant, according to C++ [class.virtual]p5.
  bool CheckOverridingFunctionReturnType(const CXXMethodDecl *New,
                                         const CXXMethodDecl *Old);

  /// CheckOverridingFunctionExceptionSpec - Checks whether the exception
  /// spec is a subset of base spec.
  bool CheckOverridingFunctionExceptionSpec(const CXXMethodDecl *New,
                                            const CXXMethodDecl *Old);

  bool CheckPureMethod(CXXMethodDecl *Method, SourceRange InitRange);

  /// CheckOverrideControl - Check C++11 override control semantics.
  void CheckOverrideControl(NamedDecl *D);

  /// DiagnoseAbsenceOfOverrideControl - Diagnose if 'override' keyword was
  /// not used in the declaration of an overriding method.
  void DiagnoseAbsenceOfOverrideControl(NamedDecl *D, bool Inconsistent);

  /// CheckForFunctionMarkedFinal - Checks whether a virtual member function
  /// overrides a virtual member function marked 'final', according to
  /// C++11 [class.virtual]p4.
  bool CheckIfOverriddenFunctionIsMarkedFinal(const CXXMethodDecl *New,
                                              const CXXMethodDecl *Old);


  //===--------------------------------------------------------------------===//
  // C++ Access Control
  //

  enum AccessResult {
    AR_accessible,
    AR_inaccessible,
    AR_dependent,
    AR_delayed
  };

  bool SetMemberAccessSpecifier(NamedDecl *MemberDecl,
                                NamedDecl *PrevMemberDecl,
                                AccessSpecifier LexicalAS);

  AccessResult CheckUnresolvedMemberAccess(UnresolvedMemberExpr *E,
                                           DeclAccessPair FoundDecl);
  AccessResult CheckUnresolvedLookupAccess(UnresolvedLookupExpr *E,
                                           DeclAccessPair FoundDecl);
  AccessResult CheckAllocationAccess(SourceLocation OperatorLoc,
                                     SourceRange PlacementRange,
                                     CXXRecordDecl *NamingClass,
                                     DeclAccessPair FoundDecl,
                                     bool Diagnose = true);
  AccessResult CheckConstructorAccess(SourceLocation Loc,
                                      CXXConstructorDecl *D,
                                      DeclAccessPair FoundDecl,
                                      const InitializedEntity &Entity,
                                      bool IsCopyBindingRefToTemp = false);
  AccessResult CheckConstructorAccess(SourceLocation Loc,
                                      CXXConstructorDecl *D,
                                      DeclAccessPair FoundDecl,
                                      const InitializedEntity &Entity,
                                      const PartialDiagnostic &PDiag);
  AccessResult CheckDestructorAccess(SourceLocation Loc,
                                     CXXDestructorDecl *Dtor,
                                     const PartialDiagnostic &PDiag,
                                     QualType objectType = QualType());
  AccessResult CheckFriendAccess(NamedDecl *D);
  AccessResult CheckMemberAccess(SourceLocation UseLoc,
                                 CXXRecordDecl *NamingClass,
                                 DeclAccessPair Found);
  AccessResult
  CheckStructuredBindingMemberAccess(SourceLocation UseLoc,
                                     CXXRecordDecl *DecomposedClass,
                                     DeclAccessPair Field);
  AccessResult CheckMemberOperatorAccess(SourceLocation Loc,
                                         Expr *ObjectExpr,
                                         Expr *ArgExpr,
                                         DeclAccessPair FoundDecl);
  AccessResult CheckAddressOfMemberAccess(Expr *OvlExpr,
                                          DeclAccessPair FoundDecl);
  AccessResult CheckBaseClassAccess(SourceLocation AccessLoc,
                                    QualType Base, QualType Derived,
                                    const CXXBasePath &Path,
                                    unsigned DiagID,
                                    bool ForceCheck = false,
                                    bool ForceUnprivileged = false);
  void CheckLookupAccess(const LookupResult &R);
  bool IsSimplyAccessible(NamedDecl *Decl, CXXRecordDecl *NamingClass,
                          QualType BaseType);
  bool isMemberAccessibleForDeletion(CXXRecordDecl *NamingClass,
                                     DeclAccessPair Found, QualType ObjectType,
                                     SourceLocation Loc,
                                     const PartialDiagnostic &Diag);
  bool isMemberAccessibleForDeletion(CXXRecordDecl *NamingClass,
                                     DeclAccessPair Found,
                                     QualType ObjectType) {
    return isMemberAccessibleForDeletion(NamingClass, Found, ObjectType,
                                         SourceLocation(), PDiag());
  }

  void HandleDependentAccessCheck(const DependentDiagnostic &DD,
                         const MultiLevelTemplateArgumentList &TemplateArgs);
  void PerformDependentDiagnostics(const DeclContext *Pattern,
                        const MultiLevelTemplateArgumentList &TemplateArgs);

  void HandleDelayedAccessCheck(sema::DelayedDiagnostic &DD, Decl *Ctx);

  /// When true, access checking violations are treated as SFINAE
  /// failures rather than hard errors.
  bool AccessCheckingSFINAE;

  enum AbstractDiagSelID {
    AbstractNone = -1,
    AbstractReturnType,
    AbstractParamType,
    AbstractVariableType,
    AbstractFieldType,
    AbstractIvarType,
    AbstractSynthesizedIvarType,
    AbstractArrayType
  };

  bool isAbstractType(SourceLocation Loc, QualType T);
  bool RequireNonAbstractType(SourceLocation Loc, QualType T,
                              TypeDiagnoser &Diagnoser);
  template <typename... Ts>
  bool RequireNonAbstractType(SourceLocation Loc, QualType T, unsigned DiagID,
                              const Ts &...Args) {
    BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
    return RequireNonAbstractType(Loc, T, Diagnoser);
  }

  void DiagnoseAbstractType(const CXXRecordDecl *RD);

  //===--------------------------------------------------------------------===//
  // C++ Overloaded Operators [C++ 13.5]
  //

  bool CheckOverloadedOperatorDeclaration(FunctionDecl *FnDecl);

  bool CheckLiteralOperatorDeclaration(FunctionDecl *FnDecl);

  //===--------------------------------------------------------------------===//
  // C++ Templates [C++ 14]
  //
  void FilterAcceptableTemplateNames(LookupResult &R,
                                     bool AllowFunctionTemplates = true,
                                     bool AllowDependent = true);
  bool hasAnyAcceptableTemplateNames(LookupResult &R,
                                     bool AllowFunctionTemplates = true,
                                     bool AllowDependent = true,
                                     bool AllowNonTemplateFunctions = false);
  /// Try to interpret the lookup result D as a template-name.
  ///
  /// \param D A declaration found by name lookup.
  /// \param AllowFunctionTemplates Whether function templates should be
  ///        considered valid results.
  /// \param AllowDependent Whether unresolved using declarations (that might
  ///        name templates) should be considered valid results.
  static NamedDecl *getAsTemplateNameDecl(NamedDecl *D,
                                          bool AllowFunctionTemplates = true,
                                          bool AllowDependent = true);

  enum TemplateNameIsRequiredTag { TemplateNameIsRequired };
  /// Whether and why a template name is required in this lookup.
  class RequiredTemplateKind {
  public:
    /// Template name is required if TemplateKWLoc is valid.
    RequiredTemplateKind(SourceLocation TemplateKWLoc = SourceLocation())
        : TemplateKW(TemplateKWLoc) {}
    /// Template name is unconditionally required.
    RequiredTemplateKind(TemplateNameIsRequiredTag) {}

    SourceLocation getTemplateKeywordLoc() const {
      return TemplateKW.getValueOr(SourceLocation());
    }
    bool hasTemplateKeyword() const { return getTemplateKeywordLoc().isValid(); }
    bool isRequired() const { return TemplateKW != SourceLocation(); }
    explicit operator bool() const { return isRequired(); }

  private:
    llvm::Optional<SourceLocation> TemplateKW;
  };

  enum class AssumedTemplateKind {
    /// This is not assumed to be a template name.
    None,
    /// This is assumed to be a template name because lookup found nothing.
    FoundNothing,
    /// This is assumed to be a template name because lookup found one or more
    /// functions (but no function templates).
    FoundFunctions,
  };
  bool LookupTemplateName(
      LookupResult &R, Scope *S, CXXScopeSpec &SS, QualType ObjectType,
      bool EnteringContext, bool &MemberOfUnknownSpecialization,
      RequiredTemplateKind RequiredTemplate = SourceLocation(),
      AssumedTemplateKind *ATK = nullptr, bool AllowTypoCorrection = true);

  TemplateNameKind isTemplateName(Scope *S,
                                  CXXScopeSpec &SS,
                                  bool hasTemplateKeyword,
                                  const UnqualifiedId &Name,
                                  ParsedType ObjectType,
                                  bool EnteringContext,
                                  TemplateTy &Template,
                                  bool &MemberOfUnknownSpecialization,
                                  bool Disambiguation = false);

  /// Try to resolve an undeclared template name as a type template.
  ///
  /// Sets II to the identifier corresponding to the template name, and updates
  /// Name to a corresponding (typo-corrected) type template name and TNK to
  /// the corresponding kind, if possible.
  void ActOnUndeclaredTypeTemplateName(Scope *S, TemplateTy &Name,
                                       TemplateNameKind &TNK,
                                       SourceLocation NameLoc,
                                       IdentifierInfo *&II);

  bool resolveAssumedTemplateNameAsType(Scope *S, TemplateName &Name,
                                        SourceLocation NameLoc,
                                        bool Diagnose = true);

  /// Determine whether a particular identifier might be the name in a C++1z
  /// deduction-guide declaration.
  bool isDeductionGuideName(Scope *S, const IdentifierInfo &Name,
                            SourceLocation NameLoc,
                            ParsedTemplateTy *Template = nullptr);

  bool DiagnoseUnknownTemplateName(const IdentifierInfo &II,
                                   SourceLocation IILoc,
                                   Scope *S,
                                   const CXXScopeSpec *SS,
                                   TemplateTy &SuggestedTemplate,
                                   TemplateNameKind &SuggestedKind);

  bool DiagnoseUninstantiableTemplate(SourceLocation PointOfInstantiation,
                                      NamedDecl *Instantiation,
                                      bool InstantiatedFromMember,
                                      const NamedDecl *Pattern,
                                      const NamedDecl *PatternDef,
                                      TemplateSpecializationKind TSK,
                                      bool Complain = true);

  void DiagnoseTemplateParameterShadow(SourceLocation Loc, Decl *PrevDecl);
  TemplateDecl *AdjustDeclIfTemplate(Decl *&Decl);

  NamedDecl *ActOnTypeParameter(Scope *S, bool Typename,
                                SourceLocation EllipsisLoc,
                                SourceLocation KeyLoc,
                                IdentifierInfo *ParamName,
                                SourceLocation ParamNameLoc,
                                unsigned Depth, unsigned Position,
                                SourceLocation EqualLoc,
                                ParsedType DefaultArg, bool HasTypeConstraint);

  bool ActOnTypeConstraint(const CXXScopeSpec &SS,
                           TemplateIdAnnotation *TypeConstraint,
                           TemplateTypeParmDecl *ConstrainedParameter,
                           SourceLocation EllipsisLoc);
  bool BuildTypeConstraint(const CXXScopeSpec &SS,
                           TemplateIdAnnotation *TypeConstraint,
                           TemplateTypeParmDecl *ConstrainedParameter,
                           SourceLocation EllipsisLoc,
                           bool AllowUnexpandedPack);

  bool AttachTypeConstraint(NestedNameSpecifierLoc NS,
                            DeclarationNameInfo NameInfo,
                            ConceptDecl *NamedConcept,
                            const TemplateArgumentListInfo *TemplateArgs,
                            TemplateTypeParmDecl *ConstrainedParameter,
                            SourceLocation EllipsisLoc);

  bool AttachTypeConstraint(AutoTypeLoc TL,
                            NonTypeTemplateParmDecl *ConstrainedParameter,
                            SourceLocation EllipsisLoc);

  bool RequireStructuralType(QualType T, SourceLocation Loc);

  QualType CheckNonTypeTemplateParameterType(TypeSourceInfo *&TSI,
                                             SourceLocation Loc);
  QualType CheckNonTypeTemplateParameterType(QualType T, SourceLocation Loc);

  NamedDecl *ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
                                      unsigned Depth,
                                      unsigned Position,
                                      SourceLocation EqualLoc,
                                      Expr *DefaultArg);
  NamedDecl *ActOnTemplateTemplateParameter(Scope *S,
                                       SourceLocation TmpLoc,
                                       TemplateParameterList *Params,
                                       SourceLocation EllipsisLoc,
                                       IdentifierInfo *ParamName,
                                       SourceLocation ParamNameLoc,
                                       unsigned Depth,
                                       unsigned Position,
                                       SourceLocation EqualLoc,
                                       ParsedTemplateArgument DefaultArg);

  TemplateParameterList *
  ActOnTemplateParameterList(unsigned Depth,
                             SourceLocation ExportLoc,
                             SourceLocation TemplateLoc,
                             SourceLocation LAngleLoc,
                             ArrayRef<NamedDecl *> Params,
                             SourceLocation RAngleLoc,
                             Expr *RequiresClause);

  /// The context in which we are checking a template parameter list.
  enum TemplateParamListContext {
    TPC_ClassTemplate,
    TPC_VarTemplate,
    TPC_FunctionTemplate,
    TPC_ClassTemplateMember,
    TPC_FriendClassTemplate,
    TPC_FriendFunctionTemplate,
    TPC_FriendFunctionTemplateDefinition,
    TPC_TypeAliasTemplate
  };

  bool CheckTemplateParameterList(TemplateParameterList *NewParams,
                                  TemplateParameterList *OldParams,
                                  TemplateParamListContext TPC,
                                  SkipBodyInfo *SkipBody = nullptr);
  TemplateParameterList *MatchTemplateParametersToScopeSpecifier(
      SourceLocation DeclStartLoc, SourceLocation DeclLoc,
      const CXXScopeSpec &SS, TemplateIdAnnotation *TemplateId,
      ArrayRef<TemplateParameterList *> ParamLists,
      bool IsFriend, bool &IsMemberSpecialization, bool &Invalid,
      bool SuppressDiagnostic = false);

  DeclResult CheckClassTemplate(
      Scope *S, unsigned TagSpec, TagUseKind TUK, SourceLocation KWLoc,
      CXXScopeSpec &SS, IdentifierInfo *Name, SourceLocation NameLoc,
      const ParsedAttributesView &Attr, TemplateParameterList *TemplateParams,
      AccessSpecifier AS, SourceLocation ModulePrivateLoc,
      SourceLocation FriendLoc, unsigned NumOuterTemplateParamLists,
      TemplateParameterList **OuterTemplateParamLists,
      SkipBodyInfo *SkipBody = nullptr);

  TemplateArgumentLoc getTrivialTemplateArgumentLoc(const TemplateArgument &Arg,
                                                    QualType NTTPType,
                                                    SourceLocation Loc);

  /// Get a template argument mapping the given template parameter to itself,
  /// e.g. for X in \c template<int X>, this would return an expression template
  /// argument referencing X.
  TemplateArgumentLoc getIdentityTemplateArgumentLoc(NamedDecl *Param,
                                                     SourceLocation Location);

  void translateTemplateArguments(const ASTTemplateArgsPtr &In,
                                  TemplateArgumentListInfo &Out);

  ParsedTemplateArgument ActOnTemplateTypeArgument(TypeResult ParsedType);

  void NoteAllFoundTemplates(TemplateName Name);

  QualType CheckTemplateIdType(TemplateName Template,
                               SourceLocation TemplateLoc,
                              TemplateArgumentListInfo &TemplateArgs);

  TypeResult
  ActOnTemplateIdType(Scope *S, CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
                      TemplateTy Template, IdentifierInfo *TemplateII,
                      SourceLocation TemplateIILoc, SourceLocation LAngleLoc,
                      ASTTemplateArgsPtr TemplateArgs, SourceLocation RAngleLoc,
                      bool IsCtorOrDtorName = false, bool IsClassName = false);

  /// Parsed an elaborated-type-specifier that refers to a template-id,
  /// such as \c class T::template apply<U>.
  TypeResult ActOnTagTemplateIdType(TagUseKind TUK,
                                    TypeSpecifierType TagSpec,
                                    SourceLocation TagLoc,
                                    CXXScopeSpec &SS,
                                    SourceLocation TemplateKWLoc,
                                    TemplateTy TemplateD,
                                    SourceLocation TemplateLoc,
                                    SourceLocation LAngleLoc,
                                    ASTTemplateArgsPtr TemplateArgsIn,
                                    SourceLocation RAngleLoc);

  DeclResult ActOnVarTemplateSpecialization(
      Scope *S, Declarator &D, TypeSourceInfo *DI,
      SourceLocation TemplateKWLoc, TemplateParameterList *TemplateParams,
      StorageClass SC, bool IsPartialSpecialization);

  /// Get the specialization of the given variable template corresponding to
  /// the specified argument list, or a null-but-valid result if the arguments
  /// are dependent.
  DeclResult CheckVarTemplateId(VarTemplateDecl *Template,
                                SourceLocation TemplateLoc,
                                SourceLocation TemplateNameLoc,
                                const TemplateArgumentListInfo &TemplateArgs);

  /// Form a reference to the specialization of the given variable template
  /// corresponding to the specified argument list, or a null-but-valid result
  /// if the arguments are dependent.
  ExprResult CheckVarTemplateId(const CXXScopeSpec &SS,
                                const DeclarationNameInfo &NameInfo,
                                VarTemplateDecl *Template,
                                SourceLocation TemplateLoc,
                                const TemplateArgumentListInfo *TemplateArgs);

  ExprResult
  CheckConceptTemplateId(const CXXScopeSpec &SS,
                         SourceLocation TemplateKWLoc,
                         const DeclarationNameInfo &ConceptNameInfo,
                         NamedDecl *FoundDecl, ConceptDecl *NamedConcept,
                         const TemplateArgumentListInfo *TemplateArgs);

  void diagnoseMissingTemplateArguments(TemplateName Name, SourceLocation Loc);

  ExprResult BuildTemplateIdExpr(const CXXScopeSpec &SS,
                                 SourceLocation TemplateKWLoc,
                                 LookupResult &R,
                                 bool RequiresADL,
                               const TemplateArgumentListInfo *TemplateArgs);

  ExprResult BuildQualifiedTemplateIdExpr(CXXScopeSpec &SS,
                                          SourceLocation TemplateKWLoc,
                               const DeclarationNameInfo &NameInfo,
                               const TemplateArgumentListInfo *TemplateArgs);

  TemplateNameKind ActOnTemplateName(
      Scope *S, CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
      const UnqualifiedId &Name, ParsedType ObjectType, bool EnteringContext,
      TemplateTy &Template, bool AllowInjectedClassName = false);

  DeclResult ActOnClassTemplateSpecialization(
      Scope *S, unsigned TagSpec, TagUseKind TUK, SourceLocation KWLoc,
      SourceLocation ModulePrivateLoc, CXXScopeSpec &SS,
      TemplateIdAnnotation &TemplateId, const ParsedAttributesView &Attr,
      MultiTemplateParamsArg TemplateParameterLists,
      SkipBodyInfo *SkipBody = nullptr);

  bool CheckTemplatePartialSpecializationArgs(SourceLocation Loc,
                                              TemplateDecl *PrimaryTemplate,
                                              unsigned NumExplicitArgs,
                                              ArrayRef<TemplateArgument> Args);
  void CheckTemplatePartialSpecialization(
      ClassTemplatePartialSpecializationDecl *Partial);
  void CheckTemplatePartialSpecialization(
      VarTemplatePartialSpecializationDecl *Partial);

  Decl *ActOnTemplateDeclarator(Scope *S,
                                MultiTemplateParamsArg TemplateParameterLists,
                                Declarator &D);

  bool
  CheckSpecializationInstantiationRedecl(SourceLocation NewLoc,
                                         TemplateSpecializationKind NewTSK,
                                         NamedDecl *PrevDecl,
                                         TemplateSpecializationKind PrevTSK,
                                         SourceLocation PrevPtOfInstantiation,
                                         bool &SuppressNew);

  bool CheckDependentFunctionTemplateSpecialization(FunctionDecl *FD,
                    const TemplateArgumentListInfo &ExplicitTemplateArgs,
                                                    LookupResult &Previous);

  bool CheckFunctionTemplateSpecialization(
      FunctionDecl *FD, TemplateArgumentListInfo *ExplicitTemplateArgs,
      LookupResult &Previous, bool QualifiedFriend = false);
  bool CheckMemberSpecialization(NamedDecl *Member, LookupResult &Previous);
  void CompleteMemberSpecialization(NamedDecl *Member, LookupResult &Previous);

  DeclResult ActOnExplicitInstantiation(
      Scope *S, SourceLocation ExternLoc, SourceLocation TemplateLoc,
      unsigned TagSpec, SourceLocation KWLoc, const CXXScopeSpec &SS,
      TemplateTy Template, SourceLocation TemplateNameLoc,
      SourceLocation LAngleLoc, ASTTemplateArgsPtr TemplateArgs,
      SourceLocation RAngleLoc, const ParsedAttributesView &Attr);

  DeclResult ActOnExplicitInstantiation(Scope *S, SourceLocation ExternLoc,
                                        SourceLocation TemplateLoc,
                                        unsigned TagSpec, SourceLocation KWLoc,
                                        CXXScopeSpec &SS, IdentifierInfo *Name,
                                        SourceLocation NameLoc,
                                        const ParsedAttributesView &Attr);

  DeclResult ActOnExplicitInstantiation(Scope *S,
                                        SourceLocation ExternLoc,
                                        SourceLocation TemplateLoc,
                                        Declarator &D);

  TemplateArgumentLoc
  SubstDefaultTemplateArgumentIfAvailable(TemplateDecl *Template,
                                          SourceLocation TemplateLoc,
                                          SourceLocation RAngleLoc,
                                          Decl *Param,
                                          SmallVectorImpl<TemplateArgument>
                                            &Converted,
                                          bool &HasDefaultArg);

  /// Specifies the context in which a particular template
  /// argument is being checked.
  enum CheckTemplateArgumentKind {
    /// The template argument was specified in the code or was
    /// instantiated with some deduced template arguments.
    CTAK_Specified,

    /// The template argument was deduced via template argument
    /// deduction.
    CTAK_Deduced,

    /// The template argument was deduced from an array bound
    /// via template argument deduction.
    CTAK_DeducedFromArrayBound
  };

  bool CheckTemplateArgument(NamedDecl *Param,
                             TemplateArgumentLoc &Arg,
                             NamedDecl *Template,
                             SourceLocation TemplateLoc,
                             SourceLocation RAngleLoc,
                             unsigned ArgumentPackIndex,
                           SmallVectorImpl<TemplateArgument> &Converted,
                             CheckTemplateArgumentKind CTAK = CTAK_Specified);

  /// Check that the given template arguments can be be provided to
  /// the given template, converting the arguments along the way.
  ///
  /// \param Template The template to which the template arguments are being
  /// provided.
  ///
  /// \param TemplateLoc The location of the template name in the source.
  ///
  /// \param TemplateArgs The list of template arguments. If the template is
  /// a template template parameter, this function may extend the set of
  /// template arguments to also include substituted, defaulted template
  /// arguments.
  ///
  /// \param PartialTemplateArgs True if the list of template arguments is
  /// intentionally partial, e.g., because we're checking just the initial
  /// set of template arguments.
  ///
  /// \param Converted Will receive the converted, canonicalized template
  /// arguments.
  ///
  /// \param UpdateArgsWithConversions If \c true, update \p TemplateArgs to
  /// contain the converted forms of the template arguments as written.
  /// Otherwise, \p TemplateArgs will not be modified.
  ///
  /// \param ConstraintsNotSatisfied If provided, and an error occured, will
  /// receive true if the cause for the error is the associated constraints of
  /// the template not being satisfied by the template arguments.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool CheckTemplateArgumentList(TemplateDecl *Template,
                                 SourceLocation TemplateLoc,
                                 TemplateArgumentListInfo &TemplateArgs,
                                 bool PartialTemplateArgs,
                                 SmallVectorImpl<TemplateArgument> &Converted,
                                 bool UpdateArgsWithConversions = true,
                                 bool *ConstraintsNotSatisfied = nullptr);

  bool CheckTemplateTypeArgument(TemplateTypeParmDecl *Param,
                                 TemplateArgumentLoc &Arg,
                           SmallVectorImpl<TemplateArgument> &Converted);

  bool CheckTemplateArgument(TypeSourceInfo *Arg);
  ExprResult CheckTemplateArgument(NonTypeTemplateParmDecl *Param,
                                   QualType InstantiatedParamType, Expr *Arg,
                                   TemplateArgument &Converted,
                               CheckTemplateArgumentKind CTAK = CTAK_Specified);
  bool CheckTemplateTemplateArgument(TemplateTemplateParmDecl *Param,
                                     TemplateParameterList *Params,
                                     TemplateArgumentLoc &Arg);

  ExprResult
  BuildExpressionFromDeclTemplateArgument(const TemplateArgument &Arg,
                                          QualType ParamType,
                                          SourceLocation Loc);
  ExprResult
  BuildExpressionFromIntegralTemplateArgument(const TemplateArgument &Arg,
                                              SourceLocation Loc);

  /// Enumeration describing how template parameter lists are compared
  /// for equality.
  enum TemplateParameterListEqualKind {
    /// We are matching the template parameter lists of two templates
    /// that might be redeclarations.
    ///
    /// \code
    /// template<typename T> struct X;
    /// template<typename T> struct X;
    /// \endcode
    TPL_TemplateMatch,

    /// We are matching the template parameter lists of two template
    /// template parameters as part of matching the template parameter lists
    /// of two templates that might be redeclarations.
    ///
    /// \code
    /// template<template<int I> class TT> struct X;
    /// template<template<int Value> class Other> struct X;
    /// \endcode
    TPL_TemplateTemplateParmMatch,

    /// We are matching the template parameter lists of a template
    /// template argument against the template parameter lists of a template
    /// template parameter.
    ///
    /// \code
    /// template<template<int Value> class Metafun> struct X;
    /// template<int Value> struct integer_c;
    /// X<integer_c> xic;
    /// \endcode
    TPL_TemplateTemplateArgumentMatch
  };

  bool TemplateParameterListsAreEqual(TemplateParameterList *New,
                                      TemplateParameterList *Old,
                                      bool Complain,
                                      TemplateParameterListEqualKind Kind,
                                      SourceLocation TemplateArgLoc
                                        = SourceLocation());

  bool CheckTemplateDeclScope(Scope *S, TemplateParameterList *TemplateParams);

  /// Called when the parser has parsed a C++ typename
  /// specifier, e.g., "typename T::type".
  ///
  /// \param S The scope in which this typename type occurs.
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param II the identifier we're retrieving (e.g., 'type' in the example).
  /// \param IdLoc the location of the identifier.
  TypeResult
  ActOnTypenameType(Scope *S, SourceLocation TypenameLoc,
                    const CXXScopeSpec &SS, const IdentifierInfo &II,
                    SourceLocation IdLoc);

  /// Called when the parser has parsed a C++ typename
  /// specifier that ends in a template-id, e.g.,
  /// "typename MetaFun::template apply<T1, T2>".
  ///
  /// \param S The scope in which this typename type occurs.
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param TemplateLoc the location of the 'template' keyword, if any.
  /// \param TemplateName The template name.
  /// \param TemplateII The identifier used to name the template.
  /// \param TemplateIILoc The location of the template name.
  /// \param LAngleLoc The location of the opening angle bracket  ('<').
  /// \param TemplateArgs The template arguments.
  /// \param RAngleLoc The location of the closing angle bracket  ('>').
  TypeResult
  ActOnTypenameType(Scope *S, SourceLocation TypenameLoc,
                    const CXXScopeSpec &SS,
                    SourceLocation TemplateLoc,
                    TemplateTy TemplateName,
                    IdentifierInfo *TemplateII,
                    SourceLocation TemplateIILoc,
                    SourceLocation LAngleLoc,
                    ASTTemplateArgsPtr TemplateArgs,
                    SourceLocation RAngleLoc);

  QualType CheckTypenameType(ElaboratedTypeKeyword Keyword,
                             SourceLocation KeywordLoc,
                             NestedNameSpecifierLoc QualifierLoc,
                             const IdentifierInfo &II,
                             SourceLocation IILoc,
                             TypeSourceInfo **TSI,
                             bool DeducedTSTContext);

  QualType CheckTypenameType(ElaboratedTypeKeyword Keyword,
                             SourceLocation KeywordLoc,
                             NestedNameSpecifierLoc QualifierLoc,
                             const IdentifierInfo &II,
                             SourceLocation IILoc,
                             bool DeducedTSTContext = true);


  TypeSourceInfo *RebuildTypeInCurrentInstantiation(TypeSourceInfo *T,
                                                    SourceLocation Loc,
                                                    DeclarationName Name);
  bool RebuildNestedNameSpecifierInCurrentInstantiation(CXXScopeSpec &SS);

  ExprResult RebuildExprInCurrentInstantiation(Expr *E);
  bool RebuildTemplateParamsInCurrentInstantiation(
                                                TemplateParameterList *Params);

  std::string
  getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                  const TemplateArgumentList &Args);

  std::string
  getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                  const TemplateArgument *Args,
                                  unsigned NumArgs);

  //===--------------------------------------------------------------------===//
  // C++ Concepts
  //===--------------------------------------------------------------------===//
  Decl *ActOnConceptDefinition(
      Scope *S, MultiTemplateParamsArg TemplateParameterLists,
      IdentifierInfo *Name, SourceLocation NameLoc, Expr *ConstraintExpr);

  RequiresExprBodyDecl *
  ActOnStartRequiresExpr(SourceLocation RequiresKWLoc,
                         ArrayRef<ParmVarDecl *> LocalParameters,
                         Scope *BodyScope);
  void ActOnFinishRequiresExpr();
  concepts::Requirement *ActOnSimpleRequirement(Expr *E);
  concepts::Requirement *ActOnTypeRequirement(
      SourceLocation TypenameKWLoc, CXXScopeSpec &SS, SourceLocation NameLoc,
      IdentifierInfo *TypeName, TemplateIdAnnotation *TemplateId);
  concepts::Requirement *ActOnCompoundRequirement(Expr *E,
                                                  SourceLocation NoexceptLoc);
  concepts::Requirement *
  ActOnCompoundRequirement(
      Expr *E, SourceLocation NoexceptLoc, CXXScopeSpec &SS,
      TemplateIdAnnotation *TypeConstraint, unsigned Depth);
  concepts::Requirement *ActOnNestedRequirement(Expr *Constraint);
  concepts::ExprRequirement *
  BuildExprRequirement(
      Expr *E, bool IsSatisfied, SourceLocation NoexceptLoc,
      concepts::ExprRequirement::ReturnTypeRequirement ReturnTypeRequirement);
  concepts::ExprRequirement *
  BuildExprRequirement(
      concepts::Requirement::SubstitutionDiagnostic *ExprSubstDiag,
      bool IsSatisfied, SourceLocation NoexceptLoc,
      concepts::ExprRequirement::ReturnTypeRequirement ReturnTypeRequirement);
  concepts::TypeRequirement *BuildTypeRequirement(TypeSourceInfo *Type);
  concepts::TypeRequirement *
  BuildTypeRequirement(
      concepts::Requirement::SubstitutionDiagnostic *SubstDiag);
  concepts::NestedRequirement *BuildNestedRequirement(Expr *E);
  concepts::NestedRequirement *
  BuildNestedRequirement(
      concepts::Requirement::SubstitutionDiagnostic *SubstDiag);
  ExprResult ActOnRequiresExpr(SourceLocation RequiresKWLoc,
                               RequiresExprBodyDecl *Body,
                               ArrayRef<ParmVarDecl *> LocalParameters,
                               ArrayRef<concepts::Requirement *> Requirements,
                               SourceLocation ClosingBraceLoc);

  //===--------------------------------------------------------------------===//
  // C++ Variadic Templates (C++0x [temp.variadic])
  //===--------------------------------------------------------------------===//

  /// Determine whether an unexpanded parameter pack might be permitted in this
  /// location. Useful for error recovery.
  bool isUnexpandedParameterPackPermitted();

  /// The context in which an unexpanded parameter pack is
  /// being diagnosed.
  ///
  /// Note that the values of this enumeration line up with the first
  /// argument to the \c err_unexpanded_parameter_pack diagnostic.
  enum UnexpandedParameterPackContext {
    /// An arbitrary expression.
    UPPC_Expression = 0,

    /// The base type of a class type.
    UPPC_BaseType,

    /// The type of an arbitrary declaration.
    UPPC_DeclarationType,

    /// The type of a data member.
    UPPC_DataMemberType,

    /// The size of a bit-field.
    UPPC_BitFieldWidth,

    /// The expression in a static assertion.
    UPPC_StaticAssertExpression,

    /// The fixed underlying type of an enumeration.
    UPPC_FixedUnderlyingType,

    /// The enumerator value.
    UPPC_EnumeratorValue,

    /// A using declaration.
    UPPC_UsingDeclaration,

    /// A friend declaration.
    UPPC_FriendDeclaration,

    /// A declaration qualifier.
    UPPC_DeclarationQualifier,

    /// An initializer.
    UPPC_Initializer,

    /// A default argument.
    UPPC_DefaultArgument,

    /// The type of a non-type template parameter.
    UPPC_NonTypeTemplateParameterType,

    /// The type of an exception.
    UPPC_ExceptionType,

    /// Partial specialization.
    UPPC_PartialSpecialization,

    /// Microsoft __if_exists.
    UPPC_IfExists,

    /// Microsoft __if_not_exists.
    UPPC_IfNotExists,

    /// Lambda expression.
    UPPC_Lambda,

    /// Block expression.
    UPPC_Block,

    /// A type constraint.
    UPPC_TypeConstraint,

    // A requirement in a requires-expression.
    UPPC_Requirement,

    // A requires-clause.
    UPPC_RequiresClause,
  };

  /// Diagnose unexpanded parameter packs.
  ///
  /// \param Loc The location at which we should emit the diagnostic.
  ///
  /// \param UPPC The context in which we are diagnosing unexpanded
  /// parameter packs.
  ///
  /// \param Unexpanded the set of unexpanded parameter packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPacks(SourceLocation Loc,
                                        UnexpandedParameterPackContext UPPC,
                                  ArrayRef<UnexpandedParameterPack> Unexpanded);

  /// If the given type contains an unexpanded parameter pack,
  /// diagnose the error.
  ///
  /// \param Loc The source location where a diagnostc should be emitted.
  ///
  /// \param T The type that is being checked for unexpanded parameter
  /// packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(SourceLocation Loc, TypeSourceInfo *T,
                                       UnexpandedParameterPackContext UPPC);

  /// If the given expression contains an unexpanded parameter
  /// pack, diagnose the error.
  ///
  /// \param E The expression that is being checked for unexpanded
  /// parameter packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(Expr *E,
                       UnexpandedParameterPackContext UPPC = UPPC_Expression);

  /// If the given requirees-expression contains an unexpanded reference to one
  /// of its own parameter packs, diagnose the error.
  ///
  /// \param RE The requiress-expression that is being checked for unexpanded
  /// parameter packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPackInRequiresExpr(RequiresExpr *RE);

  /// If the given nested-name-specifier contains an unexpanded
  /// parameter pack, diagnose the error.
  ///
  /// \param SS The nested-name-specifier that is being checked for
  /// unexpanded parameter packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(const CXXScopeSpec &SS,
                                       UnexpandedParameterPackContext UPPC);

  /// If the given name contains an unexpanded parameter pack,
  /// diagnose the error.
  ///
  /// \param NameInfo The name (with source location information) that
  /// is being checked for unexpanded parameter packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(const DeclarationNameInfo &NameInfo,
                                       UnexpandedParameterPackContext UPPC);

  /// If the given template name contains an unexpanded parameter pack,
  /// diagnose the error.
  ///
  /// \param Loc The location of the template name.
  ///
  /// \param Template The template name that is being checked for unexpanded
  /// parameter packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(SourceLocation Loc,
                                       TemplateName Template,
                                       UnexpandedParameterPackContext UPPC);

  /// If the given template argument contains an unexpanded parameter
  /// pack, diagnose the error.
  ///
  /// \param Arg The template argument that is being checked for unexpanded
  /// parameter packs.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(TemplateArgumentLoc Arg,
                                       UnexpandedParameterPackContext UPPC);

  /// Collect the set of unexpanded parameter packs within the given
  /// template argument.
  ///
  /// \param Arg The template argument that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(TemplateArgument Arg,
                   SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// Collect the set of unexpanded parameter packs within the given
  /// template argument.
  ///
  /// \param Arg The template argument that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(TemplateArgumentLoc Arg,
                    SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// Collect the set of unexpanded parameter packs within the given
  /// type.
  ///
  /// \param T The type that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(QualType T,
                   SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// Collect the set of unexpanded parameter packs within the given
  /// type.
  ///
  /// \param TL The type that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(TypeLoc TL,
                   SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// Collect the set of unexpanded parameter packs within the given
  /// nested-name-specifier.
  ///
  /// \param NNS The nested-name-specifier that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(NestedNameSpecifierLoc NNS,
                         SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// Collect the set of unexpanded parameter packs within the given
  /// name.
  ///
  /// \param NameInfo The name that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(const DeclarationNameInfo &NameInfo,
                         SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// Invoked when parsing a template argument followed by an
  /// ellipsis, which creates a pack expansion.
  ///
  /// \param Arg The template argument preceding the ellipsis, which
  /// may already be invalid.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  ParsedTemplateArgument ActOnPackExpansion(const ParsedTemplateArgument &Arg,
                                            SourceLocation EllipsisLoc);

  /// Invoked when parsing a type followed by an ellipsis, which
  /// creates a pack expansion.
  ///
  /// \param Type The type preceding the ellipsis, which will become
  /// the pattern of the pack expansion.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  TypeResult ActOnPackExpansion(ParsedType Type, SourceLocation EllipsisLoc);

  /// Construct a pack expansion type from the pattern of the pack
  /// expansion.
  TypeSourceInfo *CheckPackExpansion(TypeSourceInfo *Pattern,
                                     SourceLocation EllipsisLoc,
                                     Optional<unsigned> NumExpansions);

  /// Construct a pack expansion type from the pattern of the pack
  /// expansion.
  QualType CheckPackExpansion(QualType Pattern,
                              SourceRange PatternRange,
                              SourceLocation EllipsisLoc,
                              Optional<unsigned> NumExpansions);

  /// Invoked when parsing an expression followed by an ellipsis, which
  /// creates a pack expansion.
  ///
  /// \param Pattern The expression preceding the ellipsis, which will become
  /// the pattern of the pack expansion.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  ExprResult ActOnPackExpansion(Expr *Pattern, SourceLocation EllipsisLoc);

  /// Invoked when parsing an expression followed by an ellipsis, which
  /// creates a pack expansion.
  ///
  /// \param Pattern The expression preceding the ellipsis, which will become
  /// the pattern of the pack expansion.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  ExprResult CheckPackExpansion(Expr *Pattern, SourceLocation EllipsisLoc,
                                Optional<unsigned> NumExpansions);

  /// Determine whether we could expand a pack expansion with the
  /// given set of parameter packs into separate arguments by repeatedly
  /// transforming the pattern.
  ///
  /// \param EllipsisLoc The location of the ellipsis that identifies the
  /// pack expansion.
  ///
  /// \param PatternRange The source range that covers the entire pattern of
  /// the pack expansion.
  ///
  /// \param Unexpanded The set of unexpanded parameter packs within the
  /// pattern.
  ///
  /// \param ShouldExpand Will be set to \c true if the transformer should
  /// expand the corresponding pack expansions into separate arguments. When
  /// set, \c NumExpansions must also be set.
  ///
  /// \param RetainExpansion Whether the caller should add an unexpanded
  /// pack expansion after all of the expanded arguments. This is used
  /// when extending explicitly-specified template argument packs per
  /// C++0x [temp.arg.explicit]p9.
  ///
  /// \param NumExpansions The number of separate arguments that will be in
  /// the expanded form of the corresponding pack expansion. This is both an
  /// input and an output parameter, which can be set by the caller if the
  /// number of expansions is known a priori (e.g., due to a prior substitution)
  /// and will be set by the callee when the number of expansions is known.
  /// The callee must set this value when \c ShouldExpand is \c true; it may
  /// set this value in other cases.
  ///
  /// \returns true if an error occurred (e.g., because the parameter packs
  /// are to be instantiated with arguments of different lengths), false
  /// otherwise. If false, \c ShouldExpand (and possibly \c NumExpansions)
  /// must be set.
  bool CheckParameterPacksForExpansion(SourceLocation EllipsisLoc,
                                       SourceRange PatternRange,
                             ArrayRef<UnexpandedParameterPack> Unexpanded,
                             const MultiLevelTemplateArgumentList &TemplateArgs,
                                       bool &ShouldExpand,
                                       bool &RetainExpansion,
                                       Optional<unsigned> &NumExpansions);

  /// Determine the number of arguments in the given pack expansion
  /// type.
  ///
  /// This routine assumes that the number of arguments in the expansion is
  /// consistent across all of the unexpanded parameter packs in its pattern.
  ///
  /// Returns an empty Optional if the type can't be expanded.
  Optional<unsigned> getNumArgumentsInExpansion(QualType T,
      const MultiLevelTemplateArgumentList &TemplateArgs);

  /// Determine whether the given declarator contains any unexpanded
  /// parameter packs.
  ///
  /// This routine is used by the parser to disambiguate function declarators
  /// with an ellipsis prior to the ')', e.g.,
  ///
  /// \code
  ///   void f(T...);
  /// \endcode
  ///
  /// To determine whether we have an (unnamed) function parameter pack or
  /// a variadic function.
  ///
  /// \returns true if the declarator contains any unexpanded parameter packs,
  /// false otherwise.
  bool containsUnexpandedParameterPacks(Declarator &D);

  /// Returns the pattern of the pack expansion for a template argument.
  ///
  /// \param OrigLoc The template argument to expand.
  ///
  /// \param Ellipsis Will be set to the location of the ellipsis.
  ///
  /// \param NumExpansions Will be set to the number of expansions that will
  /// be generated from this pack expansion, if known a priori.
  TemplateArgumentLoc getTemplateArgumentPackExpansionPattern(
      TemplateArgumentLoc OrigLoc,
      SourceLocation &Ellipsis,
      Optional<unsigned> &NumExpansions) const;

  /// Given a template argument that contains an unexpanded parameter pack, but
  /// which has already been substituted, attempt to determine the number of
  /// elements that will be produced once this argument is fully-expanded.
  ///
  /// This is intended for use when transforming 'sizeof...(Arg)' in order to
  /// avoid actually expanding the pack where possible.
  Optional<unsigned> getFullyPackExpandedSize(TemplateArgument Arg);

  //===--------------------------------------------------------------------===//
  // C++ Template Argument Deduction (C++ [temp.deduct])
  //===--------------------------------------------------------------------===//

  /// Adjust the type \p ArgFunctionType to match the calling convention,
  /// noreturn, and optionally the exception specification of \p FunctionType.
  /// Deduction often wants to ignore these properties when matching function
  /// types.
  QualType adjustCCAndNoReturn(QualType ArgFunctionType, QualType FunctionType,
                               bool AdjustExceptionSpec = false);

  /// Describes the result of template argument deduction.
  ///
  /// The TemplateDeductionResult enumeration describes the result of
  /// template argument deduction, as returned from
  /// DeduceTemplateArguments(). The separate TemplateDeductionInfo
  /// structure provides additional information about the results of
  /// template argument deduction, e.g., the deduced template argument
  /// list (if successful) or the specific template parameters or
  /// deduced arguments that were involved in the failure.
  enum TemplateDeductionResult {
    /// Template argument deduction was successful.
    TDK_Success = 0,
    /// The declaration was invalid; do nothing.
    TDK_Invalid,
    /// Template argument deduction exceeded the maximum template
    /// instantiation depth (which has already been diagnosed).
    TDK_InstantiationDepth,
    /// Template argument deduction did not deduce a value
    /// for every template parameter.
    TDK_Incomplete,
    /// Template argument deduction did not deduce a value for every
    /// expansion of an expanded template parameter pack.
    TDK_IncompletePack,
    /// Template argument deduction produced inconsistent
    /// deduced values for the given template parameter.
    TDK_Inconsistent,
    /// Template argument deduction failed due to inconsistent
    /// cv-qualifiers on a template parameter type that would
    /// otherwise be deduced, e.g., we tried to deduce T in "const T"
    /// but were given a non-const "X".
    TDK_Underqualified,
    /// Substitution of the deduced template argument values
    /// resulted in an error.
    TDK_SubstitutionFailure,
    /// After substituting deduced template arguments, a dependent
    /// parameter type did not match the corresponding argument.
    TDK_DeducedMismatch,
    /// After substituting deduced template arguments, an element of
    /// a dependent parameter type did not match the corresponding element
    /// of the corresponding argument (when deducing from an initializer list).
    TDK_DeducedMismatchNested,
    /// A non-depnedent component of the parameter did not match the
    /// corresponding component of the argument.
    TDK_NonDeducedMismatch,
    /// When performing template argument deduction for a function
    /// template, there were too many call arguments.
    TDK_TooManyArguments,
    /// When performing template argument deduction for a function
    /// template, there were too few call arguments.
    TDK_TooFewArguments,
    /// The explicitly-specified template arguments were not valid
    /// template arguments for the given template.
    TDK_InvalidExplicitArguments,
    /// Checking non-dependent argument conversions failed.
    TDK_NonDependentConversionFailure,
    /// The deduced arguments did not satisfy the constraints associated
    /// with the template.
    TDK_ConstraintsNotSatisfied,
    /// Deduction failed; that's all we know.
    TDK_MiscellaneousDeductionFailure,
    /// CUDA Target attributes do not match.
    TDK_CUDATargetMismatch
  };

  TemplateDeductionResult
  DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                          const TemplateArgumentList &TemplateArgs,
                          sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(VarTemplatePartialSpecializationDecl *Partial,
                          const TemplateArgumentList &TemplateArgs,
                          sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult SubstituteExplicitTemplateArguments(
      FunctionTemplateDecl *FunctionTemplate,
      TemplateArgumentListInfo &ExplicitTemplateArgs,
      SmallVectorImpl<DeducedTemplateArgument> &Deduced,
      SmallVectorImpl<QualType> &ParamTypes, QualType *FunctionType,
      sema::TemplateDeductionInfo &Info);

  /// brief A function argument from which we performed template argument
  // deduction for a call.
  struct OriginalCallArg {
    OriginalCallArg(QualType OriginalParamType, bool DecomposedParam,
                    unsigned ArgIdx, QualType OriginalArgType)
        : OriginalParamType(OriginalParamType),
          DecomposedParam(DecomposedParam), ArgIdx(ArgIdx),
          OriginalArgType(OriginalArgType) {}

    QualType OriginalParamType;
    bool DecomposedParam;
    unsigned ArgIdx;
    QualType OriginalArgType;
  };

  TemplateDeductionResult FinishTemplateArgumentDeduction(
      FunctionTemplateDecl *FunctionTemplate,
      SmallVectorImpl<DeducedTemplateArgument> &Deduced,
      unsigned NumExplicitlySpecified, FunctionDecl *&Specialization,
      sema::TemplateDeductionInfo &Info,
      SmallVectorImpl<OriginalCallArg> const *OriginalCallArgs = nullptr,
      bool PartialOverloading = false,
      llvm::function_ref<bool()> CheckNonDependent = []{ return false; });

  TemplateDeductionResult DeduceTemplateArguments(
      FunctionTemplateDecl *FunctionTemplate,
      TemplateArgumentListInfo *ExplicitTemplateArgs, ArrayRef<Expr *> Args,
      FunctionDecl *&Specialization, sema::TemplateDeductionInfo &Info,
      bool PartialOverloading,
      llvm::function_ref<bool(ArrayRef<QualType>)> CheckNonDependent);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          TemplateArgumentListInfo *ExplicitTemplateArgs,
                          QualType ArgFunctionType,
                          FunctionDecl *&Specialization,
                          sema::TemplateDeductionInfo &Info,
                          bool IsAddressOfFunction = false);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          QualType ToType,
                          CXXConversionDecl *&Specialization,
                          sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          TemplateArgumentListInfo *ExplicitTemplateArgs,
                          FunctionDecl *&Specialization,
                          sema::TemplateDeductionInfo &Info,
                          bool IsAddressOfFunction = false);

  /// Substitute Replacement for \p auto in \p TypeWithAuto
  QualType SubstAutoType(QualType TypeWithAuto, QualType Replacement);
  /// Substitute Replacement for auto in TypeWithAuto
  TypeSourceInfo* SubstAutoTypeSourceInfo(TypeSourceInfo *TypeWithAuto,
                                          QualType Replacement);

  // Substitute auto in TypeWithAuto for a Dependent auto type
  QualType SubstAutoTypeDependent(QualType TypeWithAuto);

  // Substitute auto in TypeWithAuto for a Dependent auto type
  TypeSourceInfo *
  SubstAutoTypeSourceInfoDependent(TypeSourceInfo *TypeWithAuto);

  /// Completely replace the \c auto in \p TypeWithAuto by
  /// \p Replacement. This does not retain any \c auto type sugar.
  QualType ReplaceAutoType(QualType TypeWithAuto, QualType Replacement);
  TypeSourceInfo *ReplaceAutoTypeSourceInfo(TypeSourceInfo *TypeWithAuto,
                                            QualType Replacement);

  /// Result type of DeduceAutoType.
  enum DeduceAutoResult {
    DAR_Succeeded,
    DAR_Failed,
    DAR_FailedAlreadyDiagnosed
  };

  DeduceAutoResult
  DeduceAutoType(TypeSourceInfo *AutoType, Expr *&Initializer, QualType &Result,
                 Optional<unsigned> DependentDeductionDepth = None,
                 bool IgnoreConstraints = false);
  DeduceAutoResult
  DeduceAutoType(TypeLoc AutoTypeLoc, Expr *&Initializer, QualType &Result,
                 Optional<unsigned> DependentDeductionDepth = None,
                 bool IgnoreConstraints = false);
  void DiagnoseAutoDeductionFailure(VarDecl *VDecl, Expr *Init);
  bool DeduceReturnType(FunctionDecl *FD, SourceLocation Loc,
                        bool Diagnose = true);

  /// Declare implicit deduction guides for a class template if we've
  /// not already done so.
  void DeclareImplicitDeductionGuides(TemplateDecl *Template,
                                      SourceLocation Loc);

  QualType DeduceTemplateSpecializationFromInitializer(
      TypeSourceInfo *TInfo, const InitializedEntity &Entity,
      const InitializationKind &Kind, MultiExprArg Init);

  QualType deduceVarTypeFromInitializer(VarDecl *VDecl, DeclarationName Name,
                                        QualType Type, TypeSourceInfo *TSI,
                                        SourceRange Range, bool DirectInit,
                                        Expr *Init);

  TypeLoc getReturnTypeLoc(FunctionDecl *FD) const;

  bool DeduceFunctionTypeFromReturnExpr(FunctionDecl *FD,
                                        SourceLocation ReturnLoc,
                                        Expr *&RetExpr, AutoType *AT);

  FunctionTemplateDecl *getMoreSpecializedTemplate(
      FunctionTemplateDecl *FT1, FunctionTemplateDecl *FT2, SourceLocation Loc,
      TemplatePartialOrderingContext TPOC, unsigned NumCallArguments1,
      unsigned NumCallArguments2, bool Reversed = false);
  UnresolvedSetIterator
  getMostSpecialized(UnresolvedSetIterator SBegin, UnresolvedSetIterator SEnd,
                     TemplateSpecCandidateSet &FailedCandidates,
                     SourceLocation Loc,
                     const PartialDiagnostic &NoneDiag,
                     const PartialDiagnostic &AmbigDiag,
                     const PartialDiagnostic &CandidateDiag,
                     bool Complain = true, QualType TargetType = QualType());

  ClassTemplatePartialSpecializationDecl *
  getMoreSpecializedPartialSpecialization(
                                  ClassTemplatePartialSpecializationDecl *PS1,
                                  ClassTemplatePartialSpecializationDecl *PS2,
                                  SourceLocation Loc);

  bool isMoreSpecializedThanPrimary(ClassTemplatePartialSpecializationDecl *T,
                                    sema::TemplateDeductionInfo &Info);

  VarTemplatePartialSpecializationDecl *getMoreSpecializedPartialSpecialization(
      VarTemplatePartialSpecializationDecl *PS1,
      VarTemplatePartialSpecializationDecl *PS2, SourceLocation Loc);

  bool isMoreSpecializedThanPrimary(VarTemplatePartialSpecializationDecl *T,
                                    sema::TemplateDeductionInfo &Info);

  bool isTemplateTemplateParameterAtLeastAsSpecializedAs(
      TemplateParameterList *PParam, TemplateDecl *AArg, SourceLocation Loc);

  void MarkUsedTemplateParameters(const Expr *E, bool OnlyDeduced,
                                  unsigned Depth, llvm::SmallBitVector &Used);

  void MarkUsedTemplateParameters(const TemplateArgumentList &TemplateArgs,
                                  bool OnlyDeduced,
                                  unsigned Depth,
                                  llvm::SmallBitVector &Used);
  void MarkDeducedTemplateParameters(
                                  const FunctionTemplateDecl *FunctionTemplate,
                                  llvm::SmallBitVector &Deduced) {
    return MarkDeducedTemplateParameters(Context, FunctionTemplate, Deduced);
  }
  static void MarkDeducedTemplateParameters(ASTContext &Ctx,
                                  const FunctionTemplateDecl *FunctionTemplate,
                                  llvm::SmallBitVector &Deduced);

  //===--------------------------------------------------------------------===//
  // C++ Template Instantiation
  //

  MultiLevelTemplateArgumentList
  getTemplateInstantiationArgs(NamedDecl *D,
                               const TemplateArgumentList *Innermost = nullptr,
                               bool RelativeToPrimary = false,
                               const FunctionDecl *Pattern = nullptr);

  /// A context in which code is being synthesized (where a source location
  /// alone is not sufficient to identify the context). This covers template
  /// instantiation and various forms of implicitly-generated functions.
  struct CodeSynthesisContext {
    /// The kind of template instantiation we are performing
    enum SynthesisKind {
      /// We are instantiating a template declaration. The entity is
      /// the declaration we're instantiating (e.g., a CXXRecordDecl).
      TemplateInstantiation,

      /// We are instantiating a default argument for a template
      /// parameter. The Entity is the template parameter whose argument is
      /// being instantiated, the Template is the template, and the
      /// TemplateArgs/NumTemplateArguments provide the template arguments as
      /// specified.
      DefaultTemplateArgumentInstantiation,

      /// We are instantiating a default argument for a function.
      /// The Entity is the ParmVarDecl, and TemplateArgs/NumTemplateArgs
      /// provides the template arguments as specified.
      DefaultFunctionArgumentInstantiation,

      /// We are substituting explicit template arguments provided for
      /// a function template. The entity is a FunctionTemplateDecl.
      ExplicitTemplateArgumentSubstitution,

      /// We are substituting template argument determined as part of
      /// template argument deduction for either a class template
      /// partial specialization or a function template. The
      /// Entity is either a {Class|Var}TemplatePartialSpecializationDecl or
      /// a TemplateDecl.
      DeducedTemplateArgumentSubstitution,

      /// We are substituting prior template arguments into a new
      /// template parameter. The template parameter itself is either a
      /// NonTypeTemplateParmDecl or a TemplateTemplateParmDecl.
      PriorTemplateArgumentSubstitution,

      /// We are checking the validity of a default template argument that
      /// has been used when naming a template-id.
      DefaultTemplateArgumentChecking,

      /// We are computing the exception specification for a defaulted special
      /// member function.
      ExceptionSpecEvaluation,

      /// We are instantiating the exception specification for a function
      /// template which was deferred until it was needed.
      ExceptionSpecInstantiation,

      /// We are instantiating a requirement of a requires expression.
      RequirementInstantiation,

      /// We are checking the satisfaction of a nested requirement of a requires
      /// expression.
      NestedRequirementConstraintsCheck,

      /// We are declaring an implicit special member function.
      DeclaringSpecialMember,

      /// We are declaring an implicit 'operator==' for a defaulted
      /// 'operator<=>'.
      DeclaringImplicitEqualityComparison,

      /// We are defining a synthesized function (such as a defaulted special
      /// member).
      DefiningSynthesizedFunction,

      // We are checking the constraints associated with a constrained entity or
      // the constraint expression of a concept. This includes the checks that
      // atomic constraints have the type 'bool' and that they can be constant
      // evaluated.
      ConstraintsCheck,

      // We are substituting template arguments into a constraint expression.
      ConstraintSubstitution,

      // We are normalizing a constraint expression.
      ConstraintNormalization,

      // We are substituting into the parameter mapping of an atomic constraint
      // during normalization.
      ParameterMappingSubstitution,

      /// We are rewriting a comparison operator in terms of an operator<=>.
      RewritingOperatorAsSpaceship,

      /// We are initializing a structured binding.
      InitializingStructuredBinding,

      /// We are marking a class as __dllexport.
      MarkingClassDllexported,

      /// Added for Template instantiation observation.
      /// Memoization means we are _not_ instantiating a template because
      /// it is already instantiated (but we entered a context where we
      /// would have had to if it was not already instantiated).
      Memoization
    } Kind;

    /// Was the enclosing context a non-instantiation SFINAE context?
    bool SavedInNonInstantiationSFINAEContext;

    /// The point of instantiation or synthesis within the source code.
    SourceLocation PointOfInstantiation;

    /// The entity that is being synthesized.
    Decl *Entity;

    /// The template (or partial specialization) in which we are
    /// performing the instantiation, for substitutions of prior template
    /// arguments.
    NamedDecl *Template;

    /// The list of template arguments we are substituting, if they
    /// are not part of the entity.
    const TemplateArgument *TemplateArgs;

    // FIXME: Wrap this union around more members, or perhaps store the
    // kind-specific members in the RAII object owning the context.
    union {
      /// The number of template arguments in TemplateArgs.
      unsigned NumTemplateArgs;

      /// The special member being declared or defined.
      CXXSpecialMember SpecialMember;
    };

    ArrayRef<TemplateArgument> template_arguments() const {
      assert(Kind != DeclaringSpecialMember);
      return {TemplateArgs, NumTemplateArgs};
    }

    /// The template deduction info object associated with the
    /// substitution or checking of explicit or deduced template arguments.
    sema::TemplateDeductionInfo *DeductionInfo;

    /// The source range that covers the construct that cause
    /// the instantiation, e.g., the template-id that causes a class
    /// template instantiation.
    SourceRange InstantiationRange;

    CodeSynthesisContext()
        : Kind(TemplateInstantiation),
          SavedInNonInstantiationSFINAEContext(false), Entity(nullptr),
          Template(nullptr), TemplateArgs(nullptr), NumTemplateArgs(0),
          DeductionInfo(nullptr) {}

    /// Determines whether this template is an actual instantiation
    /// that should be counted toward the maximum instantiation depth.
    bool isInstantiationRecord() const;
  };

  /// List of active code synthesis contexts.
  ///
  /// This vector is treated as a stack. As synthesis of one entity requires
  /// synthesis of another, additional contexts are pushed onto the stack.
  SmallVector<CodeSynthesisContext, 16> CodeSynthesisContexts;

  /// Specializations whose definitions are currently being instantiated.
  llvm::DenseSet<std::pair<Decl *, unsigned>> InstantiatingSpecializations;

  /// Non-dependent types used in templates that have already been instantiated
  /// by some template instantiation.
  llvm::DenseSet<QualType> InstantiatedNonDependentTypes;

  /// Extra modules inspected when performing a lookup during a template
  /// instantiation. Computed lazily.
  SmallVector<Module*, 16> CodeSynthesisContextLookupModules;

  /// Cache of additional modules that should be used for name lookup
  /// within the current template instantiation. Computed lazily; use
  /// getLookupModules() to get a complete set.
  llvm::DenseSet<Module*> LookupModulesCache;

  /// Get the set of additional modules that should be checked during
  /// name lookup. A module and its imports become visible when instanting a
  /// template defined within it.
  llvm::DenseSet<Module*> &getLookupModules();

  /// Map from the most recent declaration of a namespace to the most
  /// recent visible declaration of that namespace.
  llvm::DenseMap<NamedDecl*, NamedDecl*> VisibleNamespaceCache;

  /// Whether we are in a SFINAE context that is not associated with
  /// template instantiation.
  ///
  /// This is used when setting up a SFINAE trap (\c see SFINAETrap) outside
  /// of a template instantiation or template argument deduction.
  bool InNonInstantiationSFINAEContext;

  /// The number of \p CodeSynthesisContexts that are not template
  /// instantiations and, therefore, should not be counted as part of the
  /// instantiation depth.
  ///
  /// When the instantiation depth reaches the user-configurable limit
  /// \p LangOptions::InstantiationDepth we will abort instantiation.
  // FIXME: Should we have a similar limit for other forms of synthesis?
  unsigned NonInstantiationEntries;

  /// The depth of the context stack at the point when the most recent
  /// error or warning was produced.
  ///
  /// This value is used to suppress printing of redundant context stacks
  /// when there are multiple errors or warnings in the same instantiation.
  // FIXME: Does this belong in Sema? It's tough to implement it anywhere else.
  unsigned LastEmittedCodeSynthesisContextDepth = 0;

  /// The template instantiation callbacks to trace or track
  /// instantiations (objects can be chained).
  ///
  /// This callbacks is used to print, trace or track template
  /// instantiations as they are being constructed.
  std::vector<std::unique_ptr<TemplateInstantiationCallback>>
      TemplateInstCallbacks;

  /// The current index into pack expansion arguments that will be
  /// used for substitution of parameter packs.
  ///
  /// The pack expansion index will be -1 to indicate that parameter packs
  /// should be instantiated as themselves. Otherwise, the index specifies
  /// which argument within the parameter pack will be used for substitution.
  int ArgumentPackSubstitutionIndex;

  /// RAII object used to change the argument pack substitution index
  /// within a \c Sema object.
  ///
  /// See \c ArgumentPackSubstitutionIndex for more information.
  class ArgumentPackSubstitutionIndexRAII {
    Sema &Self;
    int OldSubstitutionIndex;

  public:
    ArgumentPackSubstitutionIndexRAII(Sema &Self, int NewSubstitutionIndex)
      : Self(Self), OldSubstitutionIndex(Self.ArgumentPackSubstitutionIndex) {
      Self.ArgumentPackSubstitutionIndex = NewSubstitutionIndex;
    }

    ~ArgumentPackSubstitutionIndexRAII() {
      Self.ArgumentPackSubstitutionIndex = OldSubstitutionIndex;
    }
  };

  friend class ArgumentPackSubstitutionRAII;

  /// For each declaration that involved template argument deduction, the
  /// set of diagnostics that were suppressed during that template argument
  /// deduction.
  ///
  /// FIXME: Serialize this structure to the AST file.
  typedef llvm::DenseMap<Decl *, SmallVector<PartialDiagnosticAt, 1> >
    SuppressedDiagnosticsMap;
  SuppressedDiagnosticsMap SuppressedDiagnostics;

  /// A stack object to be created when performing template
  /// instantiation.
  ///
  /// Construction of an object of type \c InstantiatingTemplate
  /// pushes the current instantiation onto the stack of active
  /// instantiations. If the size of this stack exceeds the maximum
  /// number of recursive template instantiations, construction
  /// produces an error and evaluates true.
  ///
  /// Destruction of this object will pop the named instantiation off
  /// the stack.
  struct InstantiatingTemplate {
    /// Note that we are instantiating a class template,
    /// function template, variable template, alias template,
    /// or a member thereof.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          Decl *Entity,
                          SourceRange InstantiationRange = SourceRange());

    struct ExceptionSpecification {};
    /// Note that we are instantiating an exception specification
    /// of a function template.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          FunctionDecl *Entity, ExceptionSpecification,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we are instantiating a default argument in a
    /// template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateParameter Param, TemplateDecl *Template,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we are substituting either explicitly-specified or
    /// deduced template arguments during function template argument deduction.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          FunctionTemplateDecl *FunctionTemplate,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          CodeSynthesisContext::SynthesisKind Kind,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we are instantiating as part of template
    /// argument deduction for a class template declaration.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we are instantiating as part of template
    /// argument deduction for a class template partial
    /// specialization.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ClassTemplatePartialSpecializationDecl *PartialSpec,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we are instantiating as part of template
    /// argument deduction for a variable template partial
    /// specialization.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          VarTemplatePartialSpecializationDecl *PartialSpec,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we are instantiating a default argument for a function
    /// parameter.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ParmVarDecl *Param,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we are substituting prior template arguments into a
    /// non-type parameter.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          NamedDecl *Template,
                          NonTypeTemplateParmDecl *Param,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          SourceRange InstantiationRange);

    /// Note that we are substituting prior template arguments into a
    /// template template parameter.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          NamedDecl *Template,
                          TemplateTemplateParmDecl *Param,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          SourceRange InstantiationRange);

    /// Note that we are checking the default template argument
    /// against the template parameter for a given template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          NamedDecl *Param,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          SourceRange InstantiationRange);

    struct ConstraintsCheck {};
    /// \brief Note that we are checking the constraints associated with some
    /// constrained entity (a concept declaration or a template with associated
    /// constraints).
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ConstraintsCheck, NamedDecl *Template,
                          ArrayRef<TemplateArgument> TemplateArgs,
                          SourceRange InstantiationRange);

    struct ConstraintSubstitution {};
    /// \brief Note that we are checking a constraint expression associated
    /// with a template declaration or as part of the satisfaction check of a
    /// concept.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ConstraintSubstitution, NamedDecl *Template,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange);

    struct ConstraintNormalization {};
    /// \brief Note that we are normalizing a constraint expression.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ConstraintNormalization, NamedDecl *Template,
                          SourceRange InstantiationRange);

    struct ParameterMappingSubstitution {};
    /// \brief Note that we are subtituting into the parameter mapping of an
    /// atomic constraint during constraint normalization.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ParameterMappingSubstitution, NamedDecl *Template,
                          SourceRange InstantiationRange);

    /// \brief Note that we are substituting template arguments into a part of
    /// a requirement of a requires expression.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          concepts::Requirement *Req,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are checking the satisfaction of the constraint
    /// expression inside of a nested requirement.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          concepts::NestedRequirement *Req, ConstraintsCheck,
                          SourceRange InstantiationRange = SourceRange());

    /// Note that we have finished instantiating this template.
    void Clear();

    ~InstantiatingTemplate() { Clear(); }

    /// Determines whether we have exceeded the maximum
    /// recursive template instantiations.
    bool isInvalid() const { return Invalid; }

    /// Determine whether we are already instantiating this
    /// specialization in some surrounding active instantiation.
    bool isAlreadyInstantiating() const { return AlreadyInstantiating; }

  private:
    Sema &SemaRef;
    bool Invalid;
    bool AlreadyInstantiating;
    bool CheckInstantiationDepth(SourceLocation PointOfInstantiation,
                                 SourceRange InstantiationRange);

    InstantiatingTemplate(
        Sema &SemaRef, CodeSynthesisContext::SynthesisKind Kind,
        SourceLocation PointOfInstantiation, SourceRange InstantiationRange,
        Decl *Entity, NamedDecl *Template = nullptr,
        ArrayRef<TemplateArgument> TemplateArgs = None,
        sema::TemplateDeductionInfo *DeductionInfo = nullptr);

    InstantiatingTemplate(const InstantiatingTemplate&) = delete;

    InstantiatingTemplate&
    operator=(const InstantiatingTemplate&) = delete;
  };

  void pushCodeSynthesisContext(CodeSynthesisContext Ctx);
  void popCodeSynthesisContext();

  /// Determine whether we are currently performing template instantiation.
  bool inTemplateInstantiation() const {
    return CodeSynthesisContexts.size() > NonInstantiationEntries;
  }

  void PrintContextStack() {
    if (!CodeSynthesisContexts.empty() &&
        CodeSynthesisContexts.size() != LastEmittedCodeSynthesisContextDepth) {
      PrintInstantiationStack();
      LastEmittedCodeSynthesisContextDepth = CodeSynthesisContexts.size();
    }
    if (PragmaAttributeCurrentTargetDecl)
      PrintPragmaAttributeInstantiationPoint();
  }
  void PrintInstantiationStack();

  void PrintPragmaAttributeInstantiationPoint();

  /// Determines whether we are currently in a context where
  /// template argument substitution failures are not considered
  /// errors.
  ///
  /// \returns An empty \c Optional if we're not in a SFINAE context.
  /// Otherwise, contains a pointer that, if non-NULL, contains the nearest
  /// template-deduction context object, which can be used to capture
  /// diagnostics that will be suppressed.
  Optional<sema::TemplateDeductionInfo *> isSFINAEContext() const;

  /// Determines whether we are currently in a context that
  /// is not evaluated as per C++ [expr] p5.
  bool isUnevaluatedContext() const {
    assert(!ExprEvalContexts.empty() &&
           "Must be in an expression evaluation context");
    return ExprEvalContexts.back().isUnevaluated();
  }

  bool isImmediateFunctionContext() const {
    assert(!ExprEvalContexts.empty() &&
           "Must be in an expression evaluation context");
    return ExprEvalContexts.back().isImmediateFunctionContext();
  }

  /// RAII class used to determine whether SFINAE has
  /// trapped any errors that occur during template argument
  /// deduction.
  class SFINAETrap {
    Sema &SemaRef;
    unsigned PrevSFINAEErrors;
    bool PrevInNonInstantiationSFINAEContext;
    bool PrevAccessCheckingSFINAE;
    bool PrevLastDiagnosticIgnored;

  public:
    explicit SFINAETrap(Sema &SemaRef, bool AccessCheckingSFINAE = false)
      : SemaRef(SemaRef), PrevSFINAEErrors(SemaRef.NumSFINAEErrors),
        PrevInNonInstantiationSFINAEContext(
                                      SemaRef.InNonInstantiationSFINAEContext),
        PrevAccessCheckingSFINAE(SemaRef.AccessCheckingSFINAE),
        PrevLastDiagnosticIgnored(
            SemaRef.getDiagnostics().isLastDiagnosticIgnored())
    {
      if (!SemaRef.isSFINAEContext())
        SemaRef.InNonInstantiationSFINAEContext = true;
      SemaRef.AccessCheckingSFINAE = AccessCheckingSFINAE;
    }

    ~SFINAETrap() {
      SemaRef.NumSFINAEErrors = PrevSFINAEErrors;
      SemaRef.InNonInstantiationSFINAEContext
        = PrevInNonInstantiationSFINAEContext;
      SemaRef.AccessCheckingSFINAE = PrevAccessCheckingSFINAE;
      SemaRef.getDiagnostics().setLastDiagnosticIgnored(
          PrevLastDiagnosticIgnored);
    }

    /// Determine whether any SFINAE errors have been trapped.
    bool hasErrorOccurred() const {
      return SemaRef.NumSFINAEErrors > PrevSFINAEErrors;
    }
  };

  /// RAII class used to indicate that we are performing provisional
  /// semantic analysis to determine the validity of a construct, so
  /// typo-correction and diagnostics in the immediate context (not within
  /// implicitly-instantiated templates) should be suppressed.
  class TentativeAnalysisScope {
    Sema &SemaRef;
    // FIXME: Using a SFINAETrap for this is a hack.
    SFINAETrap Trap;
    bool PrevDisableTypoCorrection;
  public:
    explicit TentativeAnalysisScope(Sema &SemaRef)
        : SemaRef(SemaRef), Trap(SemaRef, true),
          PrevDisableTypoCorrection(SemaRef.DisableTypoCorrection) {
      SemaRef.DisableTypoCorrection = true;
    }
    ~TentativeAnalysisScope() {
      SemaRef.DisableTypoCorrection = PrevDisableTypoCorrection;
    }
  };

  /// The current instantiation scope used to store local
  /// variables.
  LocalInstantiationScope *CurrentInstantiationScope;

  /// Tracks whether we are in a context where typo correction is
  /// disabled.
  bool DisableTypoCorrection;

  /// The number of typos corrected by CorrectTypo.
  unsigned TyposCorrected;

  typedef llvm::SmallSet<SourceLocation, 2> SrcLocSet;
  typedef llvm::DenseMap<IdentifierInfo *, SrcLocSet> IdentifierSourceLocations;

  /// A cache containing identifiers for which typo correction failed and
  /// their locations, so that repeated attempts to correct an identifier in a
  /// given location are ignored if typo correction already failed for it.
  IdentifierSourceLocations TypoCorrectionFailures;

  /// Worker object for performing CFG-based warnings.
  sema::AnalysisBasedWarnings AnalysisWarnings;
  threadSafety::BeforeSet *ThreadSafetyDeclCache;

  /// An entity for which implicit template instantiation is required.
  ///
  /// The source location associated with the declaration is the first place in
  /// the source code where the declaration was "used". It is not necessarily
  /// the point of instantiation (which will be either before or after the
  /// namespace-scope declaration that triggered this implicit instantiation),
  /// However, it is the location that diagnostics should generally refer to,
  /// because users will need to know what code triggered the instantiation.
  typedef std::pair<ValueDecl *, SourceLocation> PendingImplicitInstantiation;

  /// The queue of implicit template instantiations that are required
  /// but have not yet been performed.
  std::deque<PendingImplicitInstantiation> PendingInstantiations;

  /// Queue of implicit template instantiations that cannot be performed
  /// eagerly.
  SmallVector<PendingImplicitInstantiation, 1> LateParsedInstantiations;

  class GlobalEagerInstantiationScope {
  public:
    GlobalEagerInstantiationScope(Sema &S, bool Enabled)
        : S(S), Enabled(Enabled) {
      if (!Enabled) return;

      SavedPendingInstantiations.swap(S.PendingInstantiations);
      SavedVTableUses.swap(S.VTableUses);
    }

    void perform() {
      if (Enabled) {
        S.DefineUsedVTables();
        S.PerformPendingInstantiations();
      }
    }

    ~GlobalEagerInstantiationScope() {
      if (!Enabled) return;

      // Restore the set of pending vtables.
      assert(S.VTableUses.empty() &&
             "VTableUses should be empty before it is discarded.");
      S.VTableUses.swap(SavedVTableUses);

      // Restore the set of pending implicit instantiations.
      if (S.TUKind != TU_Prefix || !S.LangOpts.PCHInstantiateTemplates) {
        assert(S.PendingInstantiations.empty() &&
               "PendingInstantiations should be empty before it is discarded.");
        S.PendingInstantiations.swap(SavedPendingInstantiations);
      } else {
        // Template instantiations in the PCH may be delayed until the TU.
        S.PendingInstantiations.swap(SavedPendingInstantiations);
        S.PendingInstantiations.insert(S.PendingInstantiations.end(),
                                       SavedPendingInstantiations.begin(),
                                       SavedPendingInstantiations.end());
      }
    }

  private:
    Sema &S;
    SmallVector<VTableUse, 16> SavedVTableUses;
    std::deque<PendingImplicitInstantiation> SavedPendingInstantiations;
    bool Enabled;
  };

  /// The queue of implicit template instantiations that are required
  /// and must be performed within the current local scope.
  ///
  /// This queue is only used for member functions of local classes in
  /// templates, which must be instantiated in the same scope as their
  /// enclosing function, so that they can reference function-local
  /// types, static variables, enumerators, etc.
  std::deque<PendingImplicitInstantiation> PendingLocalImplicitInstantiations;

  class LocalEagerInstantiationScope {
  public:
    LocalEagerInstantiationScope(Sema &S) : S(S) {
      SavedPendingLocalImplicitInstantiations.swap(
          S.PendingLocalImplicitInstantiations);
    }

    void perform() { S.PerformPendingInstantiations(/*LocalOnly=*/true); }

    ~LocalEagerInstantiationScope() {
      assert(S.PendingLocalImplicitInstantiations.empty() &&
             "there shouldn't be any pending local implicit instantiations");
      SavedPendingLocalImplicitInstantiations.swap(
          S.PendingLocalImplicitInstantiations);
    }

  private:
    Sema &S;
    std::deque<PendingImplicitInstantiation>
        SavedPendingLocalImplicitInstantiations;
  };

  /// A helper class for building up ExtParameterInfos.
  class ExtParameterInfoBuilder {
    SmallVector<FunctionProtoType::ExtParameterInfo, 16> Infos;
    bool HasInteresting = false;

  public:
    /// Set the ExtParameterInfo for the parameter at the given index,
    ///
    void set(unsigned index, FunctionProtoType::ExtParameterInfo info) {
      assert(Infos.size() <= index);
      Infos.resize(index);
      Infos.push_back(info);

      if (!HasInteresting)
        HasInteresting = (info != FunctionProtoType::ExtParameterInfo());
    }

    /// Return a pointer (suitable for setting in an ExtProtoInfo) to the
    /// ExtParameterInfo array we've built up.
    const FunctionProtoType::ExtParameterInfo *
    getPointerOrNull(unsigned numParams) {
      if (!HasInteresting) return nullptr;
      Infos.resize(numParams);
      return Infos.data();
    }
  };

  void PerformPendingInstantiations(bool LocalOnly = false);

  TypeSourceInfo *SubstType(TypeSourceInfo *T,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                            SourceLocation Loc, DeclarationName Entity,
                            bool AllowDeducedTST = false);

  QualType SubstType(QualType T,
                     const MultiLevelTemplateArgumentList &TemplateArgs,
                     SourceLocation Loc, DeclarationName Entity);

  TypeSourceInfo *SubstType(TypeLoc TL,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                            SourceLocation Loc, DeclarationName Entity);

  TypeSourceInfo *SubstFunctionDeclType(TypeSourceInfo *T,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                                        SourceLocation Loc,
                                        DeclarationName Entity,
                                        CXXRecordDecl *ThisContext,
                                        Qualifiers ThisTypeQuals);
  void SubstExceptionSpec(FunctionDecl *New, const FunctionProtoType *Proto,
                          const MultiLevelTemplateArgumentList &Args);
  bool SubstExceptionSpec(SourceLocation Loc,
                          FunctionProtoType::ExceptionSpecInfo &ESI,
                          SmallVectorImpl<QualType> &ExceptionStorage,
                          const MultiLevelTemplateArgumentList &Args);
  ParmVarDecl *SubstParmVarDecl(ParmVarDecl *D,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                                int indexAdjustment,
                                Optional<unsigned> NumExpansions,
                                bool ExpectParameterPack);
  bool SubstParmTypes(SourceLocation Loc, ArrayRef<ParmVarDecl *> Params,
                      const FunctionProtoType::ExtParameterInfo *ExtParamInfos,
                      const MultiLevelTemplateArgumentList &TemplateArgs,
                      SmallVectorImpl<QualType> &ParamTypes,
                      SmallVectorImpl<ParmVarDecl *> *OutParams,
                      ExtParameterInfoBuilder &ParamInfos);
  ExprResult SubstExpr(Expr *E,
                       const MultiLevelTemplateArgumentList &TemplateArgs);

  /// Substitute the given template arguments into a list of
  /// expressions, expanding pack expansions if required.
  ///
  /// \param Exprs The list of expressions to substitute into.
  ///
  /// \param IsCall Whether this is some form of call, in which case
  /// default arguments will be dropped.
  ///
  /// \param TemplateArgs The set of template arguments to substitute.
  ///
  /// \param Outputs Will receive all of the substituted arguments.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool SubstExprs(ArrayRef<Expr *> Exprs, bool IsCall,
                  const MultiLevelTemplateArgumentList &TemplateArgs,
                  SmallVectorImpl<Expr *> &Outputs);

  StmtResult SubstStmt(Stmt *S,
                       const MultiLevelTemplateArgumentList &TemplateArgs);

  TemplateParameterList *
  SubstTemplateParams(TemplateParameterList *Params, DeclContext *Owner,
                      const MultiLevelTemplateArgumentList &TemplateArgs);

  bool
  SubstTemplateArguments(ArrayRef<TemplateArgumentLoc> Args,
                         const MultiLevelTemplateArgumentList &TemplateArgs,
                         TemplateArgumentListInfo &Outputs);


  Decl *SubstDecl(Decl *D, DeclContext *Owner,
                  const MultiLevelTemplateArgumentList &TemplateArgs);

  /// Substitute the name and return type of a defaulted 'operator<=>' to form
  /// an implicit 'operator=='.
  FunctionDecl *SubstSpaceshipAsEqualEqual(CXXRecordDecl *RD,
                                           FunctionDecl *Spaceship);

  ExprResult SubstInitializer(Expr *E,
                       const MultiLevelTemplateArgumentList &TemplateArgs,
                       bool CXXDirectInit);

  bool
  SubstBaseSpecifiers(CXXRecordDecl *Instantiation,
                      CXXRecordDecl *Pattern,
                      const MultiLevelTemplateArgumentList &TemplateArgs);

  bool
  InstantiateClass(SourceLocation PointOfInstantiation,
                   CXXRecordDecl *Instantiation, CXXRecordDecl *Pattern,
                   const MultiLevelTemplateArgumentList &TemplateArgs,
                   TemplateSpecializationKind TSK,
                   bool Complain = true);

  bool InstantiateEnum(SourceLocation PointOfInstantiation,
                       EnumDecl *Instantiation, EnumDecl *Pattern,
                       const MultiLevelTemplateArgumentList &TemplateArgs,
                       TemplateSpecializationKind TSK);

  bool InstantiateInClassInitializer(
      SourceLocation PointOfInstantiation, FieldDecl *Instantiation,
      FieldDecl *Pattern, const MultiLevelTemplateArgumentList &TemplateArgs);

  struct LateInstantiatedAttribute {
    const Attr *TmplAttr;
    LocalInstantiationScope *Scope;
    Decl *NewDecl;

    LateInstantiatedAttribute(const Attr *A, LocalInstantiationScope *S,
                              Decl *D)
      : TmplAttr(A), Scope(S), NewDecl(D)
    { }
  };
  typedef SmallVector<LateInstantiatedAttribute, 16> LateInstantiatedAttrVec;

  void InstantiateAttrs(const MultiLevelTemplateArgumentList &TemplateArgs,
                        const Decl *Pattern, Decl *Inst,
                        LateInstantiatedAttrVec *LateAttrs = nullptr,
                        LocalInstantiationScope *OuterMostScope = nullptr);

  void
  InstantiateAttrsForDecl(const MultiLevelTemplateArgumentList &TemplateArgs,
                          const Decl *Pattern, Decl *Inst,
                          LateInstantiatedAttrVec *LateAttrs = nullptr,
                          LocalInstantiationScope *OuterMostScope = nullptr);

  void InstantiateDefaultCtorDefaultArgs(CXXConstructorDecl *Ctor);

  bool usesPartialOrExplicitSpecialization(
      SourceLocation Loc, ClassTemplateSpecializationDecl *ClassTemplateSpec);

  bool
  InstantiateClassTemplateSpecialization(SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           TemplateSpecializationKind TSK,
                           bool Complain = true);

  void InstantiateClassMembers(SourceLocation PointOfInstantiation,
                               CXXRecordDecl *Instantiation,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                               TemplateSpecializationKind TSK);

  void InstantiateClassTemplateSpecializationMembers(
                                          SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                                                TemplateSpecializationKind TSK);

  NestedNameSpecifierLoc
  SubstNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
                           const MultiLevelTemplateArgumentList &TemplateArgs);

  DeclarationNameInfo
  SubstDeclarationNameInfo(const DeclarationNameInfo &NameInfo,
                           const MultiLevelTemplateArgumentList &TemplateArgs);
  TemplateName
  SubstTemplateName(NestedNameSpecifierLoc QualifierLoc, TemplateName Name,
                    SourceLocation Loc,
                    const MultiLevelTemplateArgumentList &TemplateArgs);

  bool SubstTypeConstraint(TemplateTypeParmDecl *Inst, const TypeConstraint *TC,
                           const MultiLevelTemplateArgumentList &TemplateArgs);

  bool InstantiateDefaultArgument(SourceLocation CallLoc, FunctionDecl *FD,
                                  ParmVarDecl *Param);
  void InstantiateExceptionSpec(SourceLocation PointOfInstantiation,
                                FunctionDecl *Function);
  bool CheckInstantiatedFunctionTemplateConstraints(
      SourceLocation PointOfInstantiation, FunctionDecl *Decl,
      ArrayRef<TemplateArgument> TemplateArgs,
      ConstraintSatisfaction &Satisfaction);
  FunctionDecl *InstantiateFunctionDeclaration(FunctionTemplateDecl *FTD,
                                               const TemplateArgumentList *Args,
                                               SourceLocation Loc);
  void InstantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                     FunctionDecl *Function,
                                     bool Recursive = false,
                                     bool DefinitionRequired = false,
                                     bool AtEndOfTU = false);
  VarTemplateSpecializationDecl *BuildVarTemplateInstantiation(
      VarTemplateDecl *VarTemplate, VarDecl *FromVar,
      const TemplateArgumentList &TemplateArgList,
      const TemplateArgumentListInfo &TemplateArgsInfo,
      SmallVectorImpl<TemplateArgument> &Converted,
      SourceLocation PointOfInstantiation,
      LateInstantiatedAttrVec *LateAttrs = nullptr,
      LocalInstantiationScope *StartingScope = nullptr);
  VarTemplateSpecializationDecl *CompleteVarTemplateSpecializationDecl(
      VarTemplateSpecializationDecl *VarSpec, VarDecl *PatternDecl,
      const MultiLevelTemplateArgumentList &TemplateArgs);
  void
  BuildVariableInstantiation(VarDecl *NewVar, VarDecl *OldVar,
                             const MultiLevelTemplateArgumentList &TemplateArgs,
                             LateInstantiatedAttrVec *LateAttrs,
                             DeclContext *Owner,
                             LocalInstantiationScope *StartingScope,
                             bool InstantiatingVarTemplate = false,
                             VarTemplateSpecializationDecl *PrevVTSD = nullptr);

  void InstantiateVariableInitializer(
      VarDecl *Var, VarDecl *OldVar,
      const MultiLevelTemplateArgumentList &TemplateArgs);
  void InstantiateVariableDefinition(SourceLocation PointOfInstantiation,
                                     VarDecl *Var, bool Recursive = false,
                                     bool DefinitionRequired = false,
                                     bool AtEndOfTU = false);

  void InstantiateMemInitializers(CXXConstructorDecl *New,
                                  const CXXConstructorDecl *Tmpl,
                            const MultiLevelTemplateArgumentList &TemplateArgs);

  NamedDecl *FindInstantiatedDecl(SourceLocation Loc, NamedDecl *D,
                          const MultiLevelTemplateArgumentList &TemplateArgs,
                          bool FindingInstantiatedContext = false);
  DeclContext *FindInstantiatedContext(SourceLocation Loc, DeclContext *DC,
                          const MultiLevelTemplateArgumentList &TemplateArgs);

  // Objective-C declarations.
  enum ObjCContainerKind {
    OCK_None = -1,
    OCK_Interface = 0,
    OCK_Protocol,
    OCK_Category,
    OCK_ClassExtension,
    OCK_Implementation,
    OCK_CategoryImplementation
  };
  ObjCContainerKind getObjCContainerKind() const;

  DeclResult actOnObjCTypeParam(Scope *S,
                                ObjCTypeParamVariance variance,
                                SourceLocation varianceLoc,
                                unsigned index,
                                IdentifierInfo *paramName,
                                SourceLocation paramLoc,
                                SourceLocation colonLoc,
                                ParsedType typeBound);

  ObjCTypeParamList *actOnObjCTypeParamList(Scope *S, SourceLocation lAngleLoc,
                                            ArrayRef<Decl *> typeParams,
                                            SourceLocation rAngleLoc);
  void popObjCTypeParamList(Scope *S, ObjCTypeParamList *typeParamList);

  Decl *ActOnStartClassInterface(
      Scope *S, SourceLocation AtInterfaceLoc, IdentifierInfo *ClassName,
      SourceLocation ClassLoc, ObjCTypeParamList *typeParamList,
      IdentifierInfo *SuperName, SourceLocation SuperLoc,
      ArrayRef<ParsedType> SuperTypeArgs, SourceRange SuperTypeArgsRange,
      Decl *const *ProtoRefs, unsigned NumProtoRefs,
      const SourceLocation *ProtoLocs, SourceLocation EndProtoLoc,
      const ParsedAttributesView &AttrList);

  void ActOnSuperClassOfClassInterface(Scope *S,
                                       SourceLocation AtInterfaceLoc,
                                       ObjCInterfaceDecl *IDecl,
                                       IdentifierInfo *ClassName,
                                       SourceLocation ClassLoc,
                                       IdentifierInfo *SuperName,
                                       SourceLocation SuperLoc,
                                       ArrayRef<ParsedType> SuperTypeArgs,
                                       SourceRange SuperTypeArgsRange);

  void ActOnTypedefedProtocols(SmallVectorImpl<Decl *> &ProtocolRefs,
                               SmallVectorImpl<SourceLocation> &ProtocolLocs,
                               IdentifierInfo *SuperName,
                               SourceLocation SuperLoc);

  Decl *ActOnCompatibilityAlias(
                    SourceLocation AtCompatibilityAliasLoc,
                    IdentifierInfo *AliasName,  SourceLocation AliasLocation,
                    IdentifierInfo *ClassName, SourceLocation ClassLocation);

  bool CheckForwardProtocolDeclarationForCircularDependency(
    IdentifierInfo *PName,
    SourceLocation &PLoc, SourceLocation PrevLoc,
    const ObjCList<ObjCProtocolDecl> &PList);

  Decl *ActOnStartProtocolInterface(
      SourceLocation AtProtoInterfaceLoc, IdentifierInfo *ProtocolName,
      SourceLocation ProtocolLoc, Decl *const *ProtoRefNames,
      unsigned NumProtoRefs, const SourceLocation *ProtoLocs,
      SourceLocation EndProtoLoc, const ParsedAttributesView &AttrList);

  Decl *ActOnStartCategoryInterface(
      SourceLocation AtInterfaceLoc, IdentifierInfo *ClassName,
      SourceLocation ClassLoc, ObjCTypeParamList *typeParamList,
      IdentifierInfo *CategoryName, SourceLocation CategoryLoc,
      Decl *const *ProtoRefs, unsigned NumProtoRefs,
      const SourceLocation *ProtoLocs, SourceLocation EndProtoLoc,
      const ParsedAttributesView &AttrList);

  Decl *ActOnStartClassImplementation(SourceLocation AtClassImplLoc,
                                      IdentifierInfo *ClassName,
                                      SourceLocation ClassLoc,
                                      IdentifierInfo *SuperClassname,
                                      SourceLocation SuperClassLoc,
                                      const ParsedAttributesView &AttrList);

  Decl *ActOnStartCategoryImplementation(SourceLocation AtCatImplLoc,
                                         IdentifierInfo *ClassName,
                                         SourceLocation ClassLoc,
                                         IdentifierInfo *CatName,
                                         SourceLocation CatLoc,
                                         const ParsedAttributesView &AttrList);

  DeclGroupPtrTy ActOnFinishObjCImplementation(Decl *ObjCImpDecl,
                                               ArrayRef<Decl *> Decls);

  DeclGroupPtrTy ActOnForwardClassDeclaration(SourceLocation Loc,
                   IdentifierInfo **IdentList,
                   SourceLocation *IdentLocs,
                   ArrayRef<ObjCTypeParamList *> TypeParamLists,
                   unsigned NumElts);

  DeclGroupPtrTy
  ActOnForwardProtocolDeclaration(SourceLocation AtProtoclLoc,
                                  ArrayRef<IdentifierLocPair> IdentList,
                                  const ParsedAttributesView &attrList);

  void FindProtocolDeclaration(bool WarnOnDeclarations, bool ForObjCContainer,
                               ArrayRef<IdentifierLocPair> ProtocolId,
                               SmallVectorImpl<Decl *> &Protocols);

  void DiagnoseTypeArgsAndProtocols(IdentifierInfo *ProtocolId,
                                    SourceLocation ProtocolLoc,
                                    IdentifierInfo *TypeArgId,
                                    SourceLocation TypeArgLoc,
                                    bool SelectProtocolFirst = false);

  /// Given a list of identifiers (and their locations), resolve the
  /// names to either Objective-C protocol qualifiers or type
  /// arguments, as appropriate.
  void actOnObjCTypeArgsOrProtocolQualifiers(
         Scope *S,
         ParsedType baseType,
         SourceLocation lAngleLoc,
         ArrayRef<IdentifierInfo *> identifiers,
         ArrayRef<SourceLocation> identifierLocs,
         SourceLocation rAngleLoc,
         SourceLocation &typeArgsLAngleLoc,
         SmallVectorImpl<ParsedType> &typeArgs,
         SourceLocation &typeArgsRAngleLoc,
         SourceLocation &protocolLAngleLoc,
         SmallVectorImpl<Decl *> &protocols,
         SourceLocation &protocolRAngleLoc,
         bool warnOnIncompleteProtocols);

  /// Build a an Objective-C protocol-qualified 'id' type where no
  /// base type was specified.
  TypeResult actOnObjCProtocolQualifierType(
               SourceLocation lAngleLoc,
               ArrayRef<Decl *> protocols,
               ArrayRef<SourceLocation> protocolLocs,
               SourceLocation rAngleLoc);

  /// Build a specialized and/or protocol-qualified Objective-C type.
  TypeResult actOnObjCTypeArgsAndProtocolQualifiers(
               Scope *S,
               SourceLocation Loc,
               ParsedType BaseType,
               SourceLocation TypeArgsLAngleLoc,
               ArrayRef<ParsedType> TypeArgs,
               SourceLocation TypeArgsRAngleLoc,
               SourceLocation ProtocolLAngleLoc,
               ArrayRef<Decl *> Protocols,
               ArrayRef<SourceLocation> ProtocolLocs,
               SourceLocation ProtocolRAngleLoc);

  /// Build an Objective-C type parameter type.
  QualType BuildObjCTypeParamType(const ObjCTypeParamDecl *Decl,
                                  SourceLocation ProtocolLAngleLoc,
                                  ArrayRef<ObjCProtocolDecl *> Protocols,
                                  ArrayRef<SourceLocation> ProtocolLocs,
                                  SourceLocation ProtocolRAngleLoc,
                                  bool FailOnError = false);

  /// Build an Objective-C object pointer type.
  QualType BuildObjCObjectType(QualType BaseType,
                               SourceLocation Loc,
                               SourceLocation TypeArgsLAngleLoc,
                               ArrayRef<TypeSourceInfo *> TypeArgs,
                               SourceLocation TypeArgsRAngleLoc,
                               SourceLocation ProtocolLAngleLoc,
                               ArrayRef<ObjCProtocolDecl *> Protocols,
                               ArrayRef<SourceLocation> ProtocolLocs,
                               SourceLocation ProtocolRAngleLoc,
                               bool FailOnError = false);

  /// Ensure attributes are consistent with type.
  /// \param [in, out] Attributes The attributes to check; they will
  /// be modified to be consistent with \p PropertyTy.
  void CheckObjCPropertyAttributes(Decl *PropertyPtrTy,
                                   SourceLocation Loc,
                                   unsigned &Attributes,
                                   bool propertyInPrimaryClass);

  /// Process the specified property declaration and create decls for the
  /// setters and getters as needed.
  /// \param property The property declaration being processed
  void ProcessPropertyDecl(ObjCPropertyDecl *property);


  void DiagnosePropertyMismatch(ObjCPropertyDecl *Property,
                                ObjCPropertyDecl *SuperProperty,
                                const IdentifierInfo *Name,
                                bool OverridingProtocolProperty);

  void DiagnoseClassExtensionDupMethods(ObjCCategoryDecl *CAT,
                                        ObjCInterfaceDecl *ID);

  Decl *ActOnAtEnd(Scope *S, SourceRange AtEnd,
                   ArrayRef<Decl *> allMethods = None,
                   ArrayRef<DeclGroupPtrTy> allTUVars = None);

  Decl *ActOnProperty(Scope *S, SourceLocation AtLoc,
                      SourceLocation LParenLoc,
                      FieldDeclarator &FD, ObjCDeclSpec &ODS,
                      Selector GetterSel, Selector SetterSel,
                      tok::ObjCKeywordKind MethodImplKind,
                      DeclContext *lexicalDC = nullptr);

  Decl *ActOnPropertyImplDecl(Scope *S,
                              SourceLocation AtLoc,
                              SourceLocation PropertyLoc,
                              bool ImplKind,
                              IdentifierInfo *PropertyId,
                              IdentifierInfo *PropertyIvar,
                              SourceLocation PropertyIvarLoc,
                              ObjCPropertyQueryKind QueryKind);

  enum ObjCSpecialMethodKind {
    OSMK_None,
    OSMK_Alloc,
    OSMK_New,
    OSMK_Copy,
    OSMK_RetainingInit,
    OSMK_NonRetainingInit
  };

  struct ObjCArgInfo {
    IdentifierInfo *Name;
    SourceLocation NameLoc;
    // The Type is null if no type was specified, and the DeclSpec is invalid
    // in this case.
    ParsedType Type;
    ObjCDeclSpec DeclSpec;

    /// ArgAttrs - Attribute list for this argument.
    ParsedAttributesView ArgAttrs;
  };

  Decl *ActOnMethodDeclaration(
      Scope *S,
      SourceLocation BeginLoc, // location of the + or -.
      SourceLocation EndLoc,   // location of the ; or {.
      tok::TokenKind MethodType, ObjCDeclSpec &ReturnQT, ParsedType ReturnType,
      ArrayRef<SourceLocation> SelectorLocs, Selector Sel,
      // optional arguments. The number of types/arguments is obtained
      // from the Sel.getNumArgs().
      ObjCArgInfo *ArgInfo, DeclaratorChunk::ParamInfo *CParamInfo,
      unsigned CNumArgs, // c-style args
      const ParsedAttributesView &AttrList, tok::ObjCKeywordKind MethodImplKind,
      bool isVariadic, bool MethodDefinition);

  ObjCMethodDecl *LookupMethodInQualifiedType(Selector Sel,
                                              const ObjCObjectPointerType *OPT,
                                              bool IsInstance);
  ObjCMethodDecl *LookupMethodInObjectType(Selector Sel, QualType Ty,
                                           bool IsInstance);

  bool CheckARCMethodDecl(ObjCMethodDecl *method);
  bool inferObjCARCLifetime(ValueDecl *decl);

  void deduceOpenCLAddressSpace(ValueDecl *decl);

  ExprResult
  HandleExprPropertyRefExpr(const ObjCObjectPointerType *OPT,
                            Expr *BaseExpr,
                            SourceLocation OpLoc,
                            DeclarationName MemberName,
                            SourceLocation MemberLoc,
                            SourceLocation SuperLoc, QualType SuperType,
                            bool Super);

  ExprResult
  ActOnClassPropertyRefExpr(IdentifierInfo &receiverName,
                            IdentifierInfo &propertyName,
                            SourceLocation receiverNameLoc,
                            SourceLocation propertyNameLoc);

  ObjCMethodDecl *tryCaptureObjCSelf(SourceLocation Loc);

  /// Describes the kind of message expression indicated by a message
  /// send that starts with an identifier.
  enum ObjCMessageKind {
    /// The message is sent to 'super'.
    ObjCSuperMessage,
    /// The message is an instance message.
    ObjCInstanceMessage,
    /// The message is a class message, and the identifier is a type
    /// name.
    ObjCClassMessage
  };

  ObjCMessageKind getObjCMessageKind(Scope *S,
                                     IdentifierInfo *Name,
                                     SourceLocation NameLoc,
                                     bool IsSuper,
                                     bool HasTrailingDot,
                                     ParsedType &ReceiverType);

  ExprResult ActOnSuperMessage(Scope *S, SourceLocation SuperLoc,
                               Selector Sel,
                               SourceLocation LBracLoc,
                               ArrayRef<SourceLocation> SelectorLocs,
                               SourceLocation RBracLoc,
                               MultiExprArg Args);

  ExprResult BuildClassMessage(TypeSourceInfo *ReceiverTypeInfo,
                               QualType ReceiverType,
                               SourceLocation SuperLoc,
                               Selector Sel,
                               ObjCMethodDecl *Method,
                               SourceLocation LBracLoc,
                               ArrayRef<SourceLocation> SelectorLocs,
                               SourceLocation RBracLoc,
                               MultiExprArg Args,
                               bool isImplicit = false);

  ExprResult BuildClassMessageImplicit(QualType ReceiverType,
                                       bool isSuperReceiver,
                                       SourceLocation Loc,
                                       Selector Sel,
                                       ObjCMethodDecl *Method,
                                       MultiExprArg Args);

  ExprResult ActOnClassMessage(Scope *S,
                               ParsedType Receiver,
                               Selector Sel,
                               SourceLocation LBracLoc,
                               ArrayRef<SourceLocation> SelectorLocs,
                               SourceLocation RBracLoc,
                               MultiExprArg Args);

  ExprResult BuildInstanceMessage(Expr *Receiver,
                                  QualType ReceiverType,
                                  SourceLocation SuperLoc,
                                  Selector Sel,
                                  ObjCMethodDecl *Method,
                                  SourceLocation LBracLoc,
                                  ArrayRef<SourceLocation> SelectorLocs,
                                  SourceLocation RBracLoc,
                                  MultiExprArg Args,
                                  bool isImplicit = false);

  ExprResult BuildInstanceMessageImplicit(Expr *Receiver,
                                          QualType ReceiverType,
                                          SourceLocation Loc,
                                          Selector Sel,
                                          ObjCMethodDecl *Method,
                                          MultiExprArg Args);

  ExprResult ActOnInstanceMessage(Scope *S,
                                  Expr *Receiver,
                                  Selector Sel,
                                  SourceLocation LBracLoc,
                                  ArrayRef<SourceLocation> SelectorLocs,
                                  SourceLocation RBracLoc,
                                  MultiExprArg Args);

  ExprResult BuildObjCBridgedCast(SourceLocation LParenLoc,
                                  ObjCBridgeCastKind Kind,
                                  SourceLocation BridgeKeywordLoc,
                                  TypeSourceInfo *TSInfo,
                                  Expr *SubExpr);

  ExprResult ActOnObjCBridgedCast(Scope *S,
                                  SourceLocation LParenLoc,
                                  ObjCBridgeCastKind Kind,
                                  SourceLocation BridgeKeywordLoc,
                                  ParsedType Type,
                                  SourceLocation RParenLoc,
                                  Expr *SubExpr);

  void CheckTollFreeBridgeCast(QualType castType, Expr *castExpr);

  void CheckObjCBridgeRelatedCast(QualType castType, Expr *castExpr);

  bool CheckTollFreeBridgeStaticCast(QualType castType, Expr *castExpr,
                                     CastKind &Kind);

  bool checkObjCBridgeRelatedComponents(SourceLocation Loc,
                                        QualType DestType, QualType SrcType,
                                        ObjCInterfaceDecl *&RelatedClass,
                                        ObjCMethodDecl *&ClassMethod,
                                        ObjCMethodDecl *&InstanceMethod,
                                        TypedefNameDecl *&TDNDecl,
                                        bool CfToNs, bool Diagnose = true);

  bool CheckObjCBridgeRelatedConversions(SourceLocation Loc,
                                         QualType DestType, QualType SrcType,
                                         Expr *&SrcExpr, bool Diagnose = true);

  bool CheckConversionToObjCLiteral(QualType DstType, Expr *&SrcExpr,
                                    bool Diagnose = true);

  bool checkInitMethod(ObjCMethodDecl *method, QualType receiverTypeIfCall);

  /// Check whether the given new method is a valid override of the
  /// given overridden method, and set any properties that should be inherited.
  void CheckObjCMethodOverride(ObjCMethodDecl *NewMethod,
                               const ObjCMethodDecl *Overridden);

  /// Describes the compatibility of a result type with its method.
  enum ResultTypeCompatibilityKind {
    RTC_Compatible,
    RTC_Incompatible,
    RTC_Unknown
  };

  void CheckObjCMethodDirectOverrides(ObjCMethodDecl *method,
                                      ObjCMethodDecl *overridden);

  void CheckObjCMethodOverrides(ObjCMethodDecl *ObjCMethod,
                                ObjCInterfaceDecl *CurrentClass,
                                ResultTypeCompatibilityKind RTC);

  enum PragmaOptionsAlignKind {
    POAK_Native,  // #pragma options align=native
    POAK_Natural, // #pragma options align=natural
    POAK_Packed,  // #pragma options align=packed
    POAK_Power,   // #pragma options align=power
    POAK_Mac68k,  // #pragma options align=mac68k
    POAK_Reset    // #pragma options align=reset
  };

  /// ActOnPragmaClangSection - Called on well formed \#pragma clang section
  void ActOnPragmaClangSection(SourceLocation PragmaLoc,
                               PragmaClangSectionAction Action,
                               PragmaClangSectionKind SecKind, StringRef SecName);

  /// ActOnPragmaOptionsAlign - Called on well formed \#pragma options align.
  void ActOnPragmaOptionsAlign(PragmaOptionsAlignKind Kind,
                               SourceLocation PragmaLoc);

  /// ActOnPragmaPack - Called on well formed \#pragma pack(...).
  void ActOnPragmaPack(SourceLocation PragmaLoc, PragmaMsStackAction Action,
                       StringRef SlotLabel, Expr *Alignment);

  enum class PragmaAlignPackDiagnoseKind {
    NonDefaultStateAtInclude,
    ChangedStateAtExit
  };

  void DiagnoseNonDefaultPragmaAlignPack(PragmaAlignPackDiagnoseKind Kind,
                                         SourceLocation IncludeLoc);
  void DiagnoseUnterminatedPragmaAlignPack();

  /// ActOnPragmaMSStruct - Called on well formed \#pragma ms_struct [on|off].
  void ActOnPragmaMSStruct(PragmaMSStructKind Kind);

  /// ActOnPragmaMSComment - Called on well formed
  /// \#pragma comment(kind, "arg").
  void ActOnPragmaMSComment(SourceLocation CommentLoc, PragmaMSCommentKind Kind,
                            StringRef Arg);

  /// ActOnPragmaMSPointersToMembers - called on well formed \#pragma
  /// pointers_to_members(representation method[, general purpose
  /// representation]).
  void ActOnPragmaMSPointersToMembers(
      LangOptions::PragmaMSPointersToMembersKind Kind,
      SourceLocation PragmaLoc);

  /// Called on well formed \#pragma vtordisp().
  void ActOnPragmaMSVtorDisp(PragmaMsStackAction Action,
                             SourceLocation PragmaLoc,
                             MSVtorDispMode Value);

  enum PragmaSectionKind {
    PSK_DataSeg,
    PSK_BSSSeg,
    PSK_ConstSeg,
    PSK_CodeSeg,
  };

  bool UnifySection(StringRef SectionName, int SectionFlags,
                    NamedDecl *TheDecl);
  bool UnifySection(StringRef SectionName,
                    int SectionFlags,
                    SourceLocation PragmaSectionLocation);

  /// Called on well formed \#pragma bss_seg/data_seg/const_seg/code_seg.
  void ActOnPragmaMSSeg(SourceLocation PragmaLocation,
                        PragmaMsStackAction Action,
                        llvm::StringRef StackSlotLabel,
                        StringLiteral *SegmentName,
                        llvm::StringRef PragmaName);

  /// Called on well formed \#pragma section().
  void ActOnPragmaMSSection(SourceLocation PragmaLocation,
                            int SectionFlags, StringLiteral *SegmentName);

  /// Called on well-formed \#pragma init_seg().
  void ActOnPragmaMSInitSeg(SourceLocation PragmaLocation,
                            StringLiteral *SegmentName);

  /// Called on #pragma clang __debug dump II
  void ActOnPragmaDump(Scope *S, SourceLocation Loc, IdentifierInfo *II);

  /// ActOnPragmaDetectMismatch - Call on well-formed \#pragma detect_mismatch
  void ActOnPragmaDetectMismatch(SourceLocation Loc, StringRef Name,
                                 StringRef Value);

  /// Are precise floating point semantics currently enabled?
  bool isPreciseFPEnabled() {
    return !CurFPFeatures.getAllowFPReassociate() &&
           !CurFPFeatures.getNoSignedZero() &&
           !CurFPFeatures.getAllowReciprocal() &&
           !CurFPFeatures.getAllowApproxFunc();
  }

  /// ActOnPragmaFloatControl - Call on well-formed \#pragma float_control
  void ActOnPragmaFloatControl(SourceLocation Loc, PragmaMsStackAction Action,
                               PragmaFloatControlKind Value);

  /// ActOnPragmaUnused - Called on well-formed '\#pragma unused'.
  void ActOnPragmaUnused(const Token &Identifier,
                         Scope *curScope,
                         SourceLocation PragmaLoc);

  /// ActOnPragmaVisibility - Called on well formed \#pragma GCC visibility... .
  void ActOnPragmaVisibility(const IdentifierInfo* VisType,
                             SourceLocation PragmaLoc);

  NamedDecl *DeclClonePragmaWeak(NamedDecl *ND, IdentifierInfo *II,
                                 SourceLocation Loc);
  void DeclApplyPragmaWeak(Scope *S, NamedDecl *ND, WeakInfo &W);

  /// ActOnPragmaWeakID - Called on well formed \#pragma weak ident.
  void ActOnPragmaWeakID(IdentifierInfo* WeakName,
                         SourceLocation PragmaLoc,
                         SourceLocation WeakNameLoc);

  /// ActOnPragmaRedefineExtname - Called on well formed
  /// \#pragma redefine_extname oldname newname.
  void ActOnPragmaRedefineExtname(IdentifierInfo* WeakName,
                                  IdentifierInfo* AliasName,
                                  SourceLocation PragmaLoc,
                                  SourceLocation WeakNameLoc,
                                  SourceLocation AliasNameLoc);

  /// ActOnPragmaWeakAlias - Called on well formed \#pragma weak ident = ident.
  void ActOnPragmaWeakAlias(IdentifierInfo* WeakName,
                            IdentifierInfo* AliasName,
                            SourceLocation PragmaLoc,
                            SourceLocation WeakNameLoc,
                            SourceLocation AliasNameLoc);

  /// ActOnPragmaFPContract - Called on well formed
  /// \#pragma {STDC,OPENCL} FP_CONTRACT and
  /// \#pragma clang fp contract
  void ActOnPragmaFPContract(SourceLocation Loc, LangOptions::FPModeKind FPC);

  /// Called on well formed
  /// \#pragma clang fp reassociate
  void ActOnPragmaFPReassociate(SourceLocation Loc, bool IsEnabled);

  /// ActOnPragmaFenvAccess - Called on well formed
  /// \#pragma STDC FENV_ACCESS
  void ActOnPragmaFEnvAccess(SourceLocation Loc, bool IsEnabled);

  /// Called on well formed '\#pragma clang fp' that has option 'exceptions'.
  void ActOnPragmaFPExceptions(SourceLocation Loc,
                               LangOptions::FPExceptionModeKind);

  /// Called to set constant rounding mode for floating point operations.
  void setRoundingMode(SourceLocation Loc, llvm::RoundingMode);

  /// Called to set exception behavior for floating point operations.
  void setExceptionMode(SourceLocation Loc, LangOptions::FPExceptionModeKind);

  /// AddAlignmentAttributesForRecord - Adds any needed alignment attributes to
  /// a the record decl, to handle '\#pragma pack' and '\#pragma options align'.
  void AddAlignmentAttributesForRecord(RecordDecl *RD);

  /// AddMsStructLayoutForRecord - Adds ms_struct layout attribute to record.
  void AddMsStructLayoutForRecord(RecordDecl *RD);

  /// PushNamespaceVisibilityAttr - Note that we've entered a
  /// namespace with a visibility attribute.
  void PushNamespaceVisibilityAttr(const VisibilityAttr *Attr,
                                   SourceLocation Loc);

  /// AddPushedVisibilityAttribute - If '\#pragma GCC visibility' was used,
  /// add an appropriate visibility attribute.
  void AddPushedVisibilityAttribute(Decl *RD);

  /// PopPragmaVisibility - Pop the top element of the visibility stack; used
  /// for '\#pragma GCC visibility' and visibility attributes on namespaces.
  void PopPragmaVisibility(bool IsNamespaceEnd, SourceLocation EndLoc);

  /// FreeVisContext - Deallocate and null out VisContext.
  void FreeVisContext();

  /// AddCFAuditedAttribute - Check whether we're currently within
  /// '\#pragma clang arc_cf_code_audited' and, if so, consider adding
  /// the appropriate attribute.
  void AddCFAuditedAttribute(Decl *D);

  void ActOnPragmaAttributeAttribute(ParsedAttr &Attribute,
                                     SourceLocation PragmaLoc,
                                     attr::ParsedSubjectMatchRuleSet Rules);
  void ActOnPragmaAttributeEmptyPush(SourceLocation PragmaLoc,
                                     const IdentifierInfo *Namespace);

  /// Called on well-formed '\#pragma clang attribute pop'.
  void ActOnPragmaAttributePop(SourceLocation PragmaLoc,
                               const IdentifierInfo *Namespace);

  /// Adds the attributes that have been specified using the
  /// '\#pragma clang attribute push' directives to the given declaration.
  void AddPragmaAttributes(Scope *S, Decl *D);

  void DiagnoseUnterminatedPragmaAttribute();

  /// Called on well formed \#pragma clang optimize.
  void ActOnPragmaOptimize(bool On, SourceLocation PragmaLoc);

  /// Get the location for the currently active "\#pragma clang optimize
  /// off". If this location is invalid, then the state of the pragma is "on".
  SourceLocation getOptimizeOffPragmaLocation() const {
    return OptimizeOffPragmaLocation;
  }

  /// Only called on function definitions; if there is a pragma in scope
  /// with the effect of a range-based optnone, consider marking the function
  /// with attribute optnone.
  void AddRangeBasedOptnone(FunctionDecl *FD);

  /// Adds the 'optnone' attribute to the function declaration if there
  /// are no conflicts; Loc represents the location causing the 'optnone'
  /// attribute to be added (usually because of a pragma).
  void AddOptnoneAttributeIfNoConflicts(FunctionDecl *FD, SourceLocation Loc);

  /// AddAlignedAttr - Adds an aligned attribute to a particular declaration.
  void AddAlignedAttr(Decl *D, const AttributeCommonInfo &CI, Expr *E,
                      bool IsPackExpansion);
  void AddAlignedAttr(Decl *D, const AttributeCommonInfo &CI, TypeSourceInfo *T,
                      bool IsPackExpansion);

  /// AddAssumeAlignedAttr - Adds an assume_aligned attribute to a particular
  /// declaration.
  void AddAssumeAlignedAttr(Decl *D, const AttributeCommonInfo &CI, Expr *E,
                            Expr *OE);

  /// AddAllocAlignAttr - Adds an alloc_align attribute to a particular
  /// declaration.
  void AddAllocAlignAttr(Decl *D, const AttributeCommonInfo &CI,
                         Expr *ParamExpr);

  /// AddAlignValueAttr - Adds an align_value attribute to a particular
  /// declaration.
  void AddAlignValueAttr(Decl *D, const AttributeCommonInfo &CI, Expr *E);

  /// AddAnnotationAttr - Adds an annotation Annot with Args arguments to D.
  void AddAnnotationAttr(Decl *D, const AttributeCommonInfo &CI,
                         StringRef Annot, MutableArrayRef<Expr *> Args);

  /// AddLaunchBoundsAttr - Adds a launch_bounds attribute to a particular
  /// declaration.
  void AddLaunchBoundsAttr(Decl *D, const AttributeCommonInfo &CI,
                           Expr *MaxThreads, Expr *MinBlocks);

  /// AddModeAttr - Adds a mode attribute to a particular declaration.
  void AddModeAttr(Decl *D, const AttributeCommonInfo &CI, IdentifierInfo *Name,
                   bool InInstantiation = false);

  void AddParameterABIAttr(Decl *D, const AttributeCommonInfo &CI,
                           ParameterABI ABI);

  enum class RetainOwnershipKind {NS, CF, OS};
  void AddXConsumedAttr(Decl *D, const AttributeCommonInfo &CI,
                        RetainOwnershipKind K, bool IsTemplateInstantiation);

  /// addAMDGPUFlatWorkGroupSizeAttr - Adds an amdgpu_flat_work_group_size
  /// attribute to a particular declaration.
  void addAMDGPUFlatWorkGroupSizeAttr(Decl *D, const AttributeCommonInfo &CI,
                                      Expr *Min, Expr *Max);

  /// addAMDGPUWavePersEUAttr - Adds an amdgpu_waves_per_eu attribute to a
  /// particular declaration.
  void addAMDGPUWavesPerEUAttr(Decl *D, const AttributeCommonInfo &CI,
                               Expr *Min, Expr *Max);

  bool checkNSReturnsRetainedReturnType(SourceLocation loc, QualType type);

  //===--------------------------------------------------------------------===//
  // C++ Coroutines TS
  //
  bool ActOnCoroutineBodyStart(Scope *S, SourceLocation KwLoc,
                               StringRef Keyword);
  ExprResult ActOnCoawaitExpr(Scope *S, SourceLocation KwLoc, Expr *E);
  ExprResult ActOnCoyieldExpr(Scope *S, SourceLocation KwLoc, Expr *E);
  StmtResult ActOnCoreturnStmt(Scope *S, SourceLocation KwLoc, Expr *E);

  ExprResult BuildResolvedCoawaitExpr(SourceLocation KwLoc, Expr *E,
                                      bool IsImplicit = false);
  ExprResult BuildUnresolvedCoawaitExpr(SourceLocation KwLoc, Expr *E,
                                        UnresolvedLookupExpr* Lookup);
  ExprResult BuildCoyieldExpr(SourceLocation KwLoc, Expr *E);
  StmtResult BuildCoreturnStmt(SourceLocation KwLoc, Expr *E,
                               bool IsImplicit = false);
  StmtResult BuildCoroutineBodyStmt(CoroutineBodyStmt::CtorArgs);
  bool buildCoroutineParameterMoves(SourceLocation Loc);
  VarDecl *buildCoroutinePromise(SourceLocation Loc);
  void CheckCompletedCoroutineBody(FunctionDecl *FD, Stmt *&Body);
  /// Lookup 'coroutine_traits' in std namespace and std::experimental
  /// namespace. The namespace found is recorded in Namespace.
  ClassTemplateDecl *lookupCoroutineTraits(SourceLocation KwLoc,
                                           SourceLocation FuncLoc,
                                           NamespaceDecl *&Namespace);
  /// Check that the expression co_await promise.final_suspend() shall not be
  /// potentially-throwing.
  bool checkFinalSuspendNoThrow(const Stmt *FinalSuspend);

  //===--------------------------------------------------------------------===//
  // OpenMP directives and clauses.
  //
private:
  void *VarDataSharingAttributesStack;

  struct DeclareTargetContextInfo {
    struct MapInfo {
      OMPDeclareTargetDeclAttr::MapTypeTy MT;
      SourceLocation Loc;
    };
    /// Explicitly listed variables and functions in a 'to' or 'link' clause.
    llvm::DenseMap<NamedDecl *, MapInfo> ExplicitlyMapped;

    /// The 'device_type' as parsed from the clause.
    OMPDeclareTargetDeclAttr::DevTypeTy DT = OMPDeclareTargetDeclAttr::DT_Any;

    /// The directive kind, `begin declare target` or `declare target`.
    OpenMPDirectiveKind Kind;

    /// The directive with indirect clause.
    Optional<Expr *> Indirect;

    /// The directive location.
    SourceLocation Loc;

    DeclareTargetContextInfo(OpenMPDirectiveKind Kind, SourceLocation Loc)
        : Kind(Kind), Loc(Loc) {}
  };

  /// Number of nested '#pragma omp declare target' directives.
  SmallVector<DeclareTargetContextInfo, 4> DeclareTargetNesting;

  /// Initialization of data-sharing attributes stack.
  void InitDataSharingAttributesStack();
  void DestroyDataSharingAttributesStack();
  ExprResult
  VerifyPositiveIntegerConstantInClause(Expr *Op, OpenMPClauseKind CKind,
                                        bool StrictlyPositive = true,
                                        bool SuppressExprDiags = false);
  /// Returns OpenMP nesting level for current directive.
  unsigned getOpenMPNestingLevel() const;

  /// Adjusts the function scopes index for the target-based regions.
  void adjustOpenMPTargetScopeIndex(unsigned &FunctionScopesIndex,
                                    unsigned Level) const;

  /// Returns the number of scopes associated with the construct on the given
  /// OpenMP level.
  int getNumberOfConstructScopes(unsigned Level) const;

  /// Push new OpenMP function region for non-capturing function.
  void pushOpenMPFunctionRegion();

  /// Pop OpenMP function region for non-capturing function.
  void popOpenMPFunctionRegion(const sema::FunctionScopeInfo *OldFSI);

  /// Analyzes and checks a loop nest for use by a loop transformation.
  ///
  /// \param Kind          The loop transformation directive kind.
  /// \param NumLoops      How many nested loops the directive is expecting.
  /// \param AStmt         Associated statement of the transformation directive.
  /// \param LoopHelpers   [out] The loop analysis result.
  /// \param Body          [out] The body code nested in \p NumLoops loop.
  /// \param OriginalInits [out] Collection of statements and declarations that
  ///                      must have been executed/declared before entering the
  ///                      loop.
  ///
  /// \return Whether there was any error.
  bool checkTransformableLoopNest(
      OpenMPDirectiveKind Kind, Stmt *AStmt, int NumLoops,
      SmallVectorImpl<OMPLoopBasedDirective::HelperExprs> &LoopHelpers,
      Stmt *&Body,
      SmallVectorImpl<SmallVector<llvm::PointerUnion<Stmt *, Decl *>, 0>>
          &OriginalInits);

  /// Helper to keep information about the current `omp begin/end declare
  /// variant` nesting.
  struct OMPDeclareVariantScope {
    /// The associated OpenMP context selector.
    OMPTraitInfo *TI;

    /// The associated OpenMP context selector mangling.
    std::string NameSuffix;

    OMPDeclareVariantScope(OMPTraitInfo &TI);
  };

  /// Return the OMPTraitInfo for the surrounding scope, if any.
  OMPTraitInfo *getOMPTraitInfoForSurroundingScope() {
    return OMPDeclareVariantScopes.empty() ? nullptr
                                           : OMPDeclareVariantScopes.back().TI;
  }

  /// The current `omp begin/end declare variant` scopes.
  SmallVector<OMPDeclareVariantScope, 4> OMPDeclareVariantScopes;

  /// The current `omp begin/end assumes` scopes.
  SmallVector<AssumptionAttr *, 4> OMPAssumeScoped;

  /// All `omp assumes` we encountered so far.
  SmallVector<AssumptionAttr *, 4> OMPAssumeGlobal;

public:
  /// The declarator \p D defines a function in the scope \p S which is nested
  /// in an `omp begin/end declare variant` scope. In this method we create a
  /// declaration for \p D and rename \p D according to the OpenMP context
  /// selector of the surrounding scope. Return all base functions in \p Bases.
  void ActOnStartOfFunctionDefinitionInOpenMPDeclareVariantScope(
      Scope *S, Declarator &D, MultiTemplateParamsArg TemplateParameterLists,
      SmallVectorImpl<FunctionDecl *> &Bases);

  /// Register \p D as specialization of all base functions in \p Bases in the
  /// current `omp begin/end declare variant` scope.
  void ActOnFinishedFunctionDefinitionInOpenMPDeclareVariantScope(
      Decl *D, SmallVectorImpl<FunctionDecl *> &Bases);

  /// Act on \p D, a function definition inside of an `omp [begin/end] assumes`.
  void ActOnFinishedFunctionDefinitionInOpenMPAssumeScope(Decl *D);

  /// Can we exit an OpenMP declare variant scope at the moment.
  bool isInOpenMPDeclareVariantScope() const {
    return !OMPDeclareVariantScopes.empty();
  }

  /// Given the potential call expression \p Call, determine if there is a
  /// specialization via the OpenMP declare variant mechanism available. If
  /// there is, return the specialized call expression, otherwise return the
  /// original \p Call.
  ExprResult ActOnOpenMPCall(ExprResult Call, Scope *Scope,
                             SourceLocation LParenLoc, MultiExprArg ArgExprs,
                             SourceLocation RParenLoc, Expr *ExecConfig);

  /// Handle a `omp begin declare variant`.
  void ActOnOpenMPBeginDeclareVariant(SourceLocation Loc, OMPTraitInfo &TI);

  /// Handle a `omp end declare variant`.
  void ActOnOpenMPEndDeclareVariant();

  /// Checks if the variant/multiversion functions are compatible.
  bool areMultiversionVariantFunctionsCompatible(
      const FunctionDecl *OldFD, const FunctionDecl *NewFD,
      const PartialDiagnostic &NoProtoDiagID,
      const PartialDiagnosticAt &NoteCausedDiagIDAt,
      const PartialDiagnosticAt &NoSupportDiagIDAt,
      const PartialDiagnosticAt &DiffDiagIDAt, bool TemplatesSupported,
      bool ConstexprSupported, bool CLinkageMayDiffer);

  /// Function tries to capture lambda's captured variables in the OpenMP region
  /// before the original lambda is captured.
  void tryCaptureOpenMPLambdas(ValueDecl *V);

  /// Return true if the provided declaration \a VD should be captured by
  /// reference.
  /// \param Level Relative level of nested OpenMP construct for that the check
  /// is performed.
  /// \param OpenMPCaptureLevel Capture level within an OpenMP construct.
  bool isOpenMPCapturedByRef(const ValueDecl *D, unsigned Level,
                             unsigned OpenMPCaptureLevel) const;

  /// Check if the specified variable is used in one of the private
  /// clauses (private, firstprivate, lastprivate, reduction etc.) in OpenMP
  /// constructs.
  VarDecl *isOpenMPCapturedDecl(ValueDecl *D, bool CheckScopeInfo = false,
                                unsigned StopAt = 0);
  ExprResult getOpenMPCapturedExpr(VarDecl *Capture, ExprValueKind VK,
                                   ExprObjectKind OK, SourceLocation Loc);

  /// If the current region is a loop-based region, mark the start of the loop
  /// construct.
  void startOpenMPLoop();

  /// If the current region is a range loop-based region, mark the start of the
  /// loop construct.
  void startOpenMPCXXRangeFor();

  /// Check if the specified variable is used in 'private' clause.
  /// \param Level Relative level of nested OpenMP construct for that the check
  /// is performed.
  OpenMPClauseKind isOpenMPPrivateDecl(ValueDecl *D, unsigned Level,
                                       unsigned CapLevel) const;

  /// Sets OpenMP capture kind (OMPC_private, OMPC_firstprivate, OMPC_map etc.)
  /// for \p FD based on DSA for the provided corresponding captured declaration
  /// \p D.
  void setOpenMPCaptureKind(FieldDecl *FD, const ValueDecl *D, unsigned Level);

  /// Check if the specified variable is captured  by 'target' directive.
  /// \param Level Relative level of nested OpenMP construct for that the check
  /// is performed.
  bool isOpenMPTargetCapturedDecl(const ValueDecl *D, unsigned Level,
                                  unsigned CaptureLevel) const;

  /// Check if the specified global variable must be captured  by outer capture
  /// regions.
  /// \param Level Relative level of nested OpenMP construct for that
  /// the check is performed.
  bool isOpenMPGlobalCapturedDecl(ValueDecl *D, unsigned Level,
                                  unsigned CaptureLevel) const;

  ExprResult PerformOpenMPImplicitIntegerConversion(SourceLocation OpLoc,
                                                    Expr *Op);
  /// Called on start of new data sharing attribute block.
  void StartOpenMPDSABlock(OpenMPDirectiveKind K,
                           const DeclarationNameInfo &DirName, Scope *CurScope,
                           SourceLocation Loc);
  /// Start analysis of clauses.
  void StartOpenMPClause(OpenMPClauseKind K);
  /// End analysis of clauses.
  void EndOpenMPClause();
  /// Called on end of data sharing attribute block.
  void EndOpenMPDSABlock(Stmt *CurDirective);

  /// Check if the current region is an OpenMP loop region and if it is,
  /// mark loop control variable, used in \p Init for loop initialization, as
  /// private by default.
  /// \param Init First part of the for loop.
  void ActOnOpenMPLoopInitialization(SourceLocation ForLoc, Stmt *Init);

  /// Called on well-formed '\#pragma omp metadirective' after parsing
  /// of the  associated statement.
  StmtResult ActOnOpenMPMetaDirective(ArrayRef<OMPClause *> Clauses,
                                      Stmt *AStmt, SourceLocation StartLoc,
                                      SourceLocation EndLoc);

  // OpenMP directives and clauses.
  /// Called on correct id-expression from the '#pragma omp
  /// threadprivate'.
  ExprResult ActOnOpenMPIdExpression(Scope *CurScope, CXXScopeSpec &ScopeSpec,
                                     const DeclarationNameInfo &Id,
                                     OpenMPDirectiveKind Kind);
  /// Called on well-formed '#pragma omp threadprivate'.
  DeclGroupPtrTy ActOnOpenMPThreadprivateDirective(
                                     SourceLocation Loc,
                                     ArrayRef<Expr *> VarList);
  /// Builds a new OpenMPThreadPrivateDecl and checks its correctness.
  OMPThreadPrivateDecl *CheckOMPThreadPrivateDecl(SourceLocation Loc,
                                                  ArrayRef<Expr *> VarList);
  /// Called on well-formed '#pragma omp allocate'.
  DeclGroupPtrTy ActOnOpenMPAllocateDirective(SourceLocation Loc,
                                              ArrayRef<Expr *> VarList,
                                              ArrayRef<OMPClause *> Clauses,
                                              DeclContext *Owner = nullptr);

  /// Called on well-formed '#pragma omp [begin] assume[s]'.
  void ActOnOpenMPAssumesDirective(SourceLocation Loc,
                                   OpenMPDirectiveKind DKind,
                                   ArrayRef<std::string> Assumptions,
                                   bool SkippedClauses);

  /// Check if there is an active global `omp begin assumes` directive.
  bool isInOpenMPAssumeScope() const { return !OMPAssumeScoped.empty(); }

  /// Check if there is an active global `omp assumes` directive.
  bool hasGlobalOpenMPAssumes() const { return !OMPAssumeGlobal.empty(); }

  /// Called on well-formed '#pragma omp end assumes'.
  void ActOnOpenMPEndAssumesDirective();

  /// Called on well-formed '#pragma omp requires'.
  DeclGroupPtrTy ActOnOpenMPRequiresDirective(SourceLocation Loc,
                                              ArrayRef<OMPClause *> ClauseList);
  /// Check restrictions on Requires directive
  OMPRequiresDecl *CheckOMPRequiresDecl(SourceLocation Loc,
                                        ArrayRef<OMPClause *> Clauses);
  /// Check if the specified type is allowed to be used in 'omp declare
  /// reduction' construct.
  QualType ActOnOpenMPDeclareReductionType(SourceLocation TyLoc,
                                           TypeResult ParsedType);
  /// Called on start of '#pragma omp declare reduction'.
  DeclGroupPtrTy ActOnOpenMPDeclareReductionDirectiveStart(
      Scope *S, DeclContext *DC, DeclarationName Name,
      ArrayRef<std::pair<QualType, SourceLocation>> ReductionTypes,
      AccessSpecifier AS, Decl *PrevDeclInScope = nullptr);
  /// Initialize declare reduction construct initializer.
  void ActOnOpenMPDeclareReductionCombinerStart(Scope *S, Decl *D);
  /// Finish current declare reduction construct initializer.
  void ActOnOpenMPDeclareReductionCombinerEnd(Decl *D, Expr *Combiner);
  /// Initialize declare reduction construct initializer.
  /// \return omp_priv variable.
  VarDecl *ActOnOpenMPDeclareReductionInitializerStart(Scope *S, Decl *D);
  /// Finish current declare reduction construct initializer.
  void ActOnOpenMPDeclareReductionInitializerEnd(Decl *D, Expr *Initializer,
                                                 VarDecl *OmpPrivParm);
  /// Called at the end of '#pragma omp declare reduction'.
  DeclGroupPtrTy ActOnOpenMPDeclareReductionDirectiveEnd(
      Scope *S, DeclGroupPtrTy DeclReductions, bool IsValid);

  /// Check variable declaration in 'omp declare mapper' construct.
  TypeResult ActOnOpenMPDeclareMapperVarDecl(Scope *S, Declarator &D);
  /// Check if the specified type is allowed to be used in 'omp declare
  /// mapper' construct.
  QualType ActOnOpenMPDeclareMapperType(SourceLocation TyLoc,
                                        TypeResult ParsedType);
  /// Called on start of '#pragma omp declare mapper'.
  DeclGroupPtrTy ActOnOpenMPDeclareMapperDirective(
      Scope *S, DeclContext *DC, DeclarationName Name, QualType MapperType,
      SourceLocation StartLoc, DeclarationName VN, AccessSpecifier AS,
      Expr *MapperVarRef, ArrayRef<OMPClause *> Clauses,
      Decl *PrevDeclInScope = nullptr);
  /// Build the mapper variable of '#pragma omp declare mapper'.
  ExprResult ActOnOpenMPDeclareMapperDirectiveVarDecl(Scope *S,
                                                      QualType MapperType,
                                                      SourceLocation StartLoc,
                                                      DeclarationName VN);
  bool isOpenMPDeclareMapperVarDeclAllowed(const VarDecl *VD) const;
  const ValueDecl *getOpenMPDeclareMapperVarName() const;

  /// Called on the start of target region i.e. '#pragma omp declare target'.
  bool ActOnStartOpenMPDeclareTargetContext(DeclareTargetContextInfo &DTCI);

  /// Called at the end of target region i.e. '#pragma omp end declare target'.
  const DeclareTargetContextInfo ActOnOpenMPEndDeclareTargetDirective();

  /// Called once a target context is completed, that can be when a
  /// '#pragma omp end declare target' was encountered or when a
  /// '#pragma omp declare target' without declaration-definition-seq was
  /// encountered.
  void ActOnFinishedOpenMPDeclareTargetContext(DeclareTargetContextInfo &DTCI);

  /// Searches for the provided declaration name for OpenMP declare target
  /// directive.
  NamedDecl *lookupOpenMPDeclareTargetName(Scope *CurScope,
                                           CXXScopeSpec &ScopeSpec,
                                           const DeclarationNameInfo &Id);

  /// Called on correct id-expression from the '#pragma omp declare target'.
  void ActOnOpenMPDeclareTargetName(NamedDecl *ND, SourceLocation Loc,
                                    OMPDeclareTargetDeclAttr::MapTypeTy MT,
                                    DeclareTargetContextInfo &DTCI);

  /// Check declaration inside target region.
  void
  checkDeclIsAllowedInOpenMPTarget(Expr *E, Decl *D,
                                   SourceLocation IdLoc = SourceLocation());
  /// Finishes analysis of the deferred functions calls that may be declared as
  /// host/nohost during device/host compilation.
  void finalizeOpenMPDelayedAnalysis(const FunctionDecl *Caller,
                                     const FunctionDecl *Callee,
                                     SourceLocation Loc);
  /// Return true inside OpenMP declare target region.
  bool isInOpenMPDeclareTargetContext() const {
    return !DeclareTargetNesting.empty();
  }
  /// Return true inside OpenMP target region.
  bool isInOpenMPTargetExecutionDirective() const;

  /// Return the number of captured regions created for an OpenMP directive.
  static int getOpenMPCaptureLevels(OpenMPDirectiveKind Kind);

  /// Initialization of captured region for OpenMP region.
  void ActOnOpenMPRegionStart(OpenMPDirectiveKind DKind, Scope *CurScope);

  /// Called for syntactical loops (ForStmt or CXXForRangeStmt) associated to
  /// an OpenMP loop directive.
  StmtResult ActOnOpenMPCanonicalLoop(Stmt *AStmt);

  /// Process a canonical OpenMP loop nest that can either be a canonical
  /// literal loop (ForStmt or CXXForRangeStmt), or the generated loop of an
  /// OpenMP loop transformation construct.
  StmtResult ActOnOpenMPLoopnest(Stmt *AStmt);

  /// End of OpenMP region.
  ///
  /// \param S Statement associated with the current OpenMP region.
  /// \param Clauses List of clauses for the current OpenMP region.
  ///
  /// \returns Statement for finished OpenMP region.
  StmtResult ActOnOpenMPRegionEnd(StmtResult S, ArrayRef<OMPClause *> Clauses);
  StmtResult ActOnOpenMPExecutableDirective(
      OpenMPDirectiveKind Kind, const DeclarationNameInfo &DirName,
      OpenMPDirectiveKind CancelRegion, ArrayRef<OMPClause *> Clauses,
      Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp parallel' after parsing
  /// of the  associated statement.
  StmtResult ActOnOpenMPParallelDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc);
  using VarsWithInheritedDSAType =
      llvm::SmallDenseMap<const ValueDecl *, const Expr *, 4>;
  /// Called on well-formed '\#pragma omp simd' after parsing
  /// of the associated statement.
  StmtResult
  ActOnOpenMPSimdDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                           SourceLocation StartLoc, SourceLocation EndLoc,
                           VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '#pragma omp tile' after parsing of its clauses and
  /// the associated statement.
  StmtResult ActOnOpenMPTileDirective(ArrayRef<OMPClause *> Clauses,
                                      Stmt *AStmt, SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed '#pragma omp unroll' after parsing of its clauses
  /// and the associated statement.
  StmtResult ActOnOpenMPUnrollDirective(ArrayRef<OMPClause *> Clauses,
                                        Stmt *AStmt, SourceLocation StartLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp for' after parsing
  /// of the associated statement.
  StmtResult
  ActOnOpenMPForDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                          SourceLocation StartLoc, SourceLocation EndLoc,
                          VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp for simd' after parsing
  /// of the associated statement.
  StmtResult
  ActOnOpenMPForSimdDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                              SourceLocation StartLoc, SourceLocation EndLoc,
                              VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp sections' after parsing
  /// of the associated statement.
  StmtResult ActOnOpenMPSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp section' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPSectionDirective(Stmt *AStmt, SourceLocation StartLoc,
                                         SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp single' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPSingleDirective(ArrayRef<OMPClause *> Clauses,
                                        Stmt *AStmt, SourceLocation StartLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp master' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPMasterDirective(Stmt *AStmt, SourceLocation StartLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp critical' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPCriticalDirective(const DeclarationNameInfo &DirName,
                                          ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp parallel for' after parsing
  /// of the  associated statement.
  StmtResult ActOnOpenMPParallelForDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp parallel for simd' after
  /// parsing of the  associated statement.
  StmtResult ActOnOpenMPParallelForSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp parallel master' after
  /// parsing of the  associated statement.
  StmtResult ActOnOpenMPParallelMasterDirective(ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp parallel sections' after
  /// parsing of the  associated statement.
  StmtResult ActOnOpenMPParallelSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                                  Stmt *AStmt,
                                                  SourceLocation StartLoc,
                                                  SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp task' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPTaskDirective(ArrayRef<OMPClause *> Clauses,
                                      Stmt *AStmt, SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp taskyield'.
  StmtResult ActOnOpenMPTaskyieldDirective(SourceLocation StartLoc,
                                           SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp barrier'.
  StmtResult ActOnOpenMPBarrierDirective(SourceLocation StartLoc,
                                         SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp taskwait'.
  StmtResult ActOnOpenMPTaskwaitDirective(ArrayRef<OMPClause *> Clauses,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp taskgroup'.
  StmtResult ActOnOpenMPTaskgroupDirective(ArrayRef<OMPClause *> Clauses,
                                           Stmt *AStmt, SourceLocation StartLoc,
                                           SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp flush'.
  StmtResult ActOnOpenMPFlushDirective(ArrayRef<OMPClause *> Clauses,
                                       SourceLocation StartLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp depobj'.
  StmtResult ActOnOpenMPDepobjDirective(ArrayRef<OMPClause *> Clauses,
                                        SourceLocation StartLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp scan'.
  StmtResult ActOnOpenMPScanDirective(ArrayRef<OMPClause *> Clauses,
                                      SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp ordered' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPOrderedDirective(ArrayRef<OMPClause *> Clauses,
                                         Stmt *AStmt, SourceLocation StartLoc,
                                         SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp atomic' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPAtomicDirective(ArrayRef<OMPClause *> Clauses,
                                        Stmt *AStmt, SourceLocation StartLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp target' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPTargetDirective(ArrayRef<OMPClause *> Clauses,
                                        Stmt *AStmt, SourceLocation StartLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp target data' after parsing of
  /// the associated statement.
  StmtResult ActOnOpenMPTargetDataDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt, SourceLocation StartLoc,
                                            SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp target enter data' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPTargetEnterDataDirective(ArrayRef<OMPClause *> Clauses,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 Stmt *AStmt);
  /// Called on well-formed '\#pragma omp target exit data' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPTargetExitDataDirective(ArrayRef<OMPClause *> Clauses,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc,
                                                Stmt *AStmt);
  /// Called on well-formed '\#pragma omp target parallel' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPTargetParallelDirective(ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp target parallel for' after
  /// parsing of the  associated statement.
  StmtResult ActOnOpenMPTargetParallelForDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp teams' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPTeamsDirective(ArrayRef<OMPClause *> Clauses,
                                       Stmt *AStmt, SourceLocation StartLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp cancellation point'.
  StmtResult
  ActOnOpenMPCancellationPointDirective(SourceLocation StartLoc,
                                        SourceLocation EndLoc,
                                        OpenMPDirectiveKind CancelRegion);
  /// Called on well-formed '\#pragma omp cancel'.
  StmtResult ActOnOpenMPCancelDirective(ArrayRef<OMPClause *> Clauses,
                                        SourceLocation StartLoc,
                                        SourceLocation EndLoc,
                                        OpenMPDirectiveKind CancelRegion);
  /// Called on well-formed '\#pragma omp taskloop' after parsing of the
  /// associated statement.
  StmtResult
  ActOnOpenMPTaskLoopDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                               SourceLocation StartLoc, SourceLocation EndLoc,
                               VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp taskloop simd' after parsing of
  /// the associated statement.
  StmtResult ActOnOpenMPTaskLoopSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp master taskloop' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPMasterTaskLoopDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp master taskloop simd' after parsing of
  /// the associated statement.
  StmtResult ActOnOpenMPMasterTaskLoopSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp parallel master taskloop' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPParallelMasterTaskLoopDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp parallel master taskloop simd' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPParallelMasterTaskLoopSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp distribute' after parsing
  /// of the associated statement.
  StmtResult
  ActOnOpenMPDistributeDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                                 SourceLocation StartLoc, SourceLocation EndLoc,
                                 VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp target update'.
  StmtResult ActOnOpenMPTargetUpdateDirective(ArrayRef<OMPClause *> Clauses,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc,
                                              Stmt *AStmt);
  /// Called on well-formed '\#pragma omp distribute parallel for' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPDistributeParallelForDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp distribute parallel for simd'
  /// after parsing of the associated statement.
  StmtResult ActOnOpenMPDistributeParallelForSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp distribute simd' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPDistributeSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp target parallel for simd' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPTargetParallelForSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp target simd' after parsing of
  /// the associated statement.
  StmtResult
  ActOnOpenMPTargetSimdDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                                 SourceLocation StartLoc, SourceLocation EndLoc,
                                 VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp teams distribute' after parsing of
  /// the associated statement.
  StmtResult ActOnOpenMPTeamsDistributeDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp teams distribute simd' after parsing
  /// of the associated statement.
  StmtResult ActOnOpenMPTeamsDistributeSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp teams distribute parallel for simd'
  /// after parsing of the associated statement.
  StmtResult ActOnOpenMPTeamsDistributeParallelForSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp teams distribute parallel for'
  /// after parsing of the associated statement.
  StmtResult ActOnOpenMPTeamsDistributeParallelForDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp target teams' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPTargetTeamsDirective(ArrayRef<OMPClause *> Clauses,
                                             Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp target teams distribute' after parsing
  /// of the associated statement.
  StmtResult ActOnOpenMPTargetTeamsDistributeDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp target teams distribute parallel for'
  /// after parsing of the associated statement.
  StmtResult ActOnOpenMPTargetTeamsDistributeParallelForDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp target teams distribute parallel for
  /// simd' after parsing of the associated statement.
  StmtResult ActOnOpenMPTargetTeamsDistributeParallelForSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp target teams distribute simd' after
  /// parsing of the associated statement.
  StmtResult ActOnOpenMPTargetTeamsDistributeSimdDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
  /// Called on well-formed '\#pragma omp interop'.
  StmtResult ActOnOpenMPInteropDirective(ArrayRef<OMPClause *> Clauses,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp dispatch' after parsing of the
  // /associated statement.
  StmtResult ActOnOpenMPDispatchDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc);
  /// Called on well-formed '\#pragma omp masked' after parsing of the
  // /associated statement.
  StmtResult ActOnOpenMPMaskedDirective(ArrayRef<OMPClause *> Clauses,
                                        Stmt *AStmt, SourceLocation StartLoc,
                                        SourceLocation EndLoc);

  /// Called on well-formed '\#pragma omp loop' after parsing of the
  /// associated statement.
  StmtResult ActOnOpenMPGenericLoopDirective(
      ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
      SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);

  /// Checks correctness of linear modifiers.
  bool CheckOpenMPLinearModifier(OpenMPLinearClauseKind LinKind,
                                 SourceLocation LinLoc);
  /// Checks that the specified declaration matches requirements for the linear
  /// decls.
  bool CheckOpenMPLinearDecl(const ValueDecl *D, SourceLocation ELoc,
                             OpenMPLinearClauseKind LinKind, QualType Type,
                             bool IsDeclareSimd = false);

  /// Called on well-formed '\#pragma omp declare simd' after parsing of
  /// the associated method/function.
  DeclGroupPtrTy ActOnOpenMPDeclareSimdDirective(
      DeclGroupPtrTy DG, OMPDeclareSimdDeclAttr::BranchStateTy BS,
      Expr *Simdlen, ArrayRef<Expr *> Uniforms, ArrayRef<Expr *> Aligneds,
      ArrayRef<Expr *> Alignments, ArrayRef<Expr *> Linears,
      ArrayRef<unsigned> LinModifiers, ArrayRef<Expr *> Steps, SourceRange SR);

  /// Checks '\#pragma omp declare variant' variant function and original
  /// functions after parsing of the associated method/function.
  /// \param DG Function declaration to which declare variant directive is
  /// applied to.
  /// \param VariantRef Expression that references the variant function, which
  /// must be used instead of the original one, specified in \p DG.
  /// \param TI The trait info object representing the match clause.
  /// \param NumAppendArgs The number of omp_interop_t arguments to account for
  /// in checking.
  /// \returns None, if the function/variant function are not compatible with
  /// the pragma, pair of original function/variant ref expression otherwise.
  Optional<std::pair<FunctionDecl *, Expr *>>
  checkOpenMPDeclareVariantFunction(DeclGroupPtrTy DG, Expr *VariantRef,
                                    OMPTraitInfo &TI, unsigned NumAppendArgs,
                                    SourceRange SR);

  /// Called on well-formed '\#pragma omp declare variant' after parsing of
  /// the associated method/function.
  /// \param FD Function declaration to which declare variant directive is
  /// applied to.
  /// \param VariantRef Expression that references the variant function, which
  /// must be used instead of the original one, specified in \p DG.
  /// \param TI The context traits associated with the function variant.
  /// \param AdjustArgsNothing The list of 'nothing' arguments.
  /// \param AdjustArgsNeedDevicePtr The list of 'need_device_ptr' arguments.
  /// \param AppendArgs The list of 'append_args' arguments.
  /// \param AdjustArgsLoc The Location of an 'adjust_args' clause.
  /// \param AppendArgsLoc The Location of an 'append_args' clause.
  /// \param SR The SourceRange of the 'declare variant' directive.
  void ActOnOpenMPDeclareVariantDirective(
      FunctionDecl *FD, Expr *VariantRef, OMPTraitInfo &TI,
      ArrayRef<Expr *> AdjustArgsNothing,
      ArrayRef<Expr *> AdjustArgsNeedDevicePtr,
      ArrayRef<OMPDeclareVariantAttr::InteropType> AppendArgs,
      SourceLocation AdjustArgsLoc, SourceLocation AppendArgsLoc,
      SourceRange SR);

  OMPClause *ActOnOpenMPSingleExprClause(OpenMPClauseKind Kind,
                                         Expr *Expr,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc);
  /// Called on well-formed 'allocator' clause.
  OMPClause *ActOnOpenMPAllocatorClause(Expr *Allocator,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed 'if' clause.
  OMPClause *ActOnOpenMPIfClause(OpenMPDirectiveKind NameModifier,
                                 Expr *Condition, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation NameModifierLoc,
                                 SourceLocation ColonLoc,
                                 SourceLocation EndLoc);
  /// Called on well-formed 'final' clause.
  OMPClause *ActOnOpenMPFinalClause(Expr *Condition, SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'num_threads' clause.
  OMPClause *ActOnOpenMPNumThreadsClause(Expr *NumThreads,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc);
  /// Called on well-formed 'align' clause.
  OMPClause *ActOnOpenMPAlignClause(Expr *Alignment, SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'safelen' clause.
  OMPClause *ActOnOpenMPSafelenClause(Expr *Length,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'simdlen' clause.
  OMPClause *ActOnOpenMPSimdlenClause(Expr *Length, SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);
  /// Called on well-form 'sizes' clause.
  OMPClause *ActOnOpenMPSizesClause(ArrayRef<Expr *> SizeExprs,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc);
  /// Called on well-form 'full' clauses.
  OMPClause *ActOnOpenMPFullClause(SourceLocation StartLoc,
                                   SourceLocation EndLoc);
  /// Called on well-form 'partial' clauses.
  OMPClause *ActOnOpenMPPartialClause(Expr *FactorExpr, SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'collapse' clause.
  OMPClause *ActOnOpenMPCollapseClause(Expr *NumForLoops,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed 'ordered' clause.
  OMPClause *
  ActOnOpenMPOrderedClause(SourceLocation StartLoc, SourceLocation EndLoc,
                           SourceLocation LParenLoc = SourceLocation(),
                           Expr *NumForLoops = nullptr);
  /// Called on well-formed 'grainsize' clause.
  OMPClause *ActOnOpenMPGrainsizeClause(Expr *Size, SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed 'num_tasks' clause.
  OMPClause *ActOnOpenMPNumTasksClause(Expr *NumTasks, SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed 'hint' clause.
  OMPClause *ActOnOpenMPHintClause(Expr *Hint, SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'detach' clause.
  OMPClause *ActOnOpenMPDetachClause(Expr *Evt, SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);

  OMPClause *ActOnOpenMPSimpleClause(OpenMPClauseKind Kind,
                                     unsigned Argument,
                                     SourceLocation ArgumentLoc,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'when' clause.
  OMPClause *ActOnOpenMPWhenClause(OMPTraitInfo &TI, SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'default' clause.
  OMPClause *ActOnOpenMPDefaultClause(llvm::omp::DefaultKind Kind,
                                      SourceLocation KindLoc,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'proc_bind' clause.
  OMPClause *ActOnOpenMPProcBindClause(llvm::omp::ProcBindKind Kind,
                                       SourceLocation KindLoc,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed 'order' clause.
  OMPClause *ActOnOpenMPOrderClause(OpenMPOrderClauseKind Kind,
                                    SourceLocation KindLoc,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'update' clause.
  OMPClause *ActOnOpenMPUpdateClause(OpenMPDependClauseKind Kind,
                                     SourceLocation KindLoc,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);

  OMPClause *ActOnOpenMPSingleExprWithArgClause(
      OpenMPClauseKind Kind, ArrayRef<unsigned> Arguments, Expr *Expr,
      SourceLocation StartLoc, SourceLocation LParenLoc,
      ArrayRef<SourceLocation> ArgumentsLoc, SourceLocation DelimLoc,
      SourceLocation EndLoc);
  /// Called on well-formed 'schedule' clause.
  OMPClause *ActOnOpenMPScheduleClause(
      OpenMPScheduleClauseModifier M1, OpenMPScheduleClauseModifier M2,
      OpenMPScheduleClauseKind Kind, Expr *ChunkSize, SourceLocation StartLoc,
      SourceLocation LParenLoc, SourceLocation M1Loc, SourceLocation M2Loc,
      SourceLocation KindLoc, SourceLocation CommaLoc, SourceLocation EndLoc);

  OMPClause *ActOnOpenMPClause(OpenMPClauseKind Kind, SourceLocation StartLoc,
                               SourceLocation EndLoc);
  /// Called on well-formed 'nowait' clause.
  OMPClause *ActOnOpenMPNowaitClause(SourceLocation StartLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'untied' clause.
  OMPClause *ActOnOpenMPUntiedClause(SourceLocation StartLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'mergeable' clause.
  OMPClause *ActOnOpenMPMergeableClause(SourceLocation StartLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed 'read' clause.
  OMPClause *ActOnOpenMPReadClause(SourceLocation StartLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'write' clause.
  OMPClause *ActOnOpenMPWriteClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'update' clause.
  OMPClause *ActOnOpenMPUpdateClause(SourceLocation StartLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'capture' clause.
  OMPClause *ActOnOpenMPCaptureClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'compare' clause.
  OMPClause *ActOnOpenMPCompareClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'seq_cst' clause.
  OMPClause *ActOnOpenMPSeqCstClause(SourceLocation StartLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'acq_rel' clause.
  OMPClause *ActOnOpenMPAcqRelClause(SourceLocation StartLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'acquire' clause.
  OMPClause *ActOnOpenMPAcquireClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'release' clause.
  OMPClause *ActOnOpenMPReleaseClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'relaxed' clause.
  OMPClause *ActOnOpenMPRelaxedClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc);

  /// Called on well-formed 'init' clause.
  OMPClause *ActOnOpenMPInitClause(Expr *InteropVar, ArrayRef<Expr *> PrefExprs,
                                   bool IsTarget, bool IsTargetSync,
                                   SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation VarLoc,
                                   SourceLocation EndLoc);

  /// Called on well-formed 'use' clause.
  OMPClause *ActOnOpenMPUseClause(Expr *InteropVar, SourceLocation StartLoc,
                                  SourceLocation LParenLoc,
                                  SourceLocation VarLoc, SourceLocation EndLoc);

  /// Called on well-formed 'destroy' clause.
  OMPClause *ActOnOpenMPDestroyClause(Expr *InteropVar, SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation VarLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'novariants' clause.
  OMPClause *ActOnOpenMPNovariantsClause(Expr *Condition,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc);
  /// Called on well-formed 'nocontext' clause.
  OMPClause *ActOnOpenMPNocontextClause(Expr *Condition,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed 'filter' clause.
  OMPClause *ActOnOpenMPFilterClause(Expr *ThreadID, SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'threads' clause.
  OMPClause *ActOnOpenMPThreadsClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'simd' clause.
  OMPClause *ActOnOpenMPSIMDClause(SourceLocation StartLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'nogroup' clause.
  OMPClause *ActOnOpenMPNogroupClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'unified_address' clause.
  OMPClause *ActOnOpenMPUnifiedAddressClause(SourceLocation StartLoc,
                                             SourceLocation EndLoc);

  /// Called on well-formed 'unified_address' clause.
  OMPClause *ActOnOpenMPUnifiedSharedMemoryClause(SourceLocation StartLoc,
                                                  SourceLocation EndLoc);

  /// Called on well-formed 'reverse_offload' clause.
  OMPClause *ActOnOpenMPReverseOffloadClause(SourceLocation StartLoc,
                                             SourceLocation EndLoc);

  /// Called on well-formed 'dynamic_allocators' clause.
  OMPClause *ActOnOpenMPDynamicAllocatorsClause(SourceLocation StartLoc,
                                                SourceLocation EndLoc);

  /// Called on well-formed 'atomic_default_mem_order' clause.
  OMPClause *ActOnOpenMPAtomicDefaultMemOrderClause(
      OpenMPAtomicDefaultMemOrderClauseKind Kind, SourceLocation KindLoc,
      SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc);

  OMPClause *ActOnOpenMPVarListClause(
      OpenMPClauseKind Kind, ArrayRef<Expr *> Vars, Expr *DepModOrTailExpr,
      const OMPVarListLocTy &Locs, SourceLocation ColonLoc,
      CXXScopeSpec &ReductionOrMapperIdScopeSpec,
      DeclarationNameInfo &ReductionOrMapperId, int ExtraModifier,
      ArrayRef<OpenMPMapModifierKind> MapTypeModifiers,
      ArrayRef<SourceLocation> MapTypeModifiersLoc, bool IsMapTypeImplicit,
      SourceLocation ExtraModifierLoc,
      ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
      ArrayRef<SourceLocation> MotionModifiersLoc);
  /// Called on well-formed 'inclusive' clause.
  OMPClause *ActOnOpenMPInclusiveClause(ArrayRef<Expr *> VarList,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed 'exclusive' clause.
  OMPClause *ActOnOpenMPExclusiveClause(ArrayRef<Expr *> VarList,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc);
  /// Called on well-formed 'allocate' clause.
  OMPClause *
  ActOnOpenMPAllocateClause(Expr *Allocator, ArrayRef<Expr *> VarList,
                            SourceLocation StartLoc, SourceLocation ColonLoc,
                            SourceLocation LParenLoc, SourceLocation EndLoc);
  /// Called on well-formed 'private' clause.
  OMPClause *ActOnOpenMPPrivateClause(ArrayRef<Expr *> VarList,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'firstprivate' clause.
  OMPClause *ActOnOpenMPFirstprivateClause(ArrayRef<Expr *> VarList,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc);
  /// Called on well-formed 'lastprivate' clause.
  OMPClause *ActOnOpenMPLastprivateClause(
      ArrayRef<Expr *> VarList, OpenMPLastprivateModifier LPKind,
      SourceLocation LPKindLoc, SourceLocation ColonLoc,
      SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc);
  /// Called on well-formed 'shared' clause.
  OMPClause *ActOnOpenMPSharedClause(ArrayRef<Expr *> VarList,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'reduction' clause.
  OMPClause *ActOnOpenMPReductionClause(
      ArrayRef<Expr *> VarList, OpenMPReductionClauseModifier Modifier,
      SourceLocation StartLoc, SourceLocation LParenLoc,
      SourceLocation ModifierLoc, SourceLocation ColonLoc,
      SourceLocation EndLoc, CXXScopeSpec &ReductionIdScopeSpec,
      const DeclarationNameInfo &ReductionId,
      ArrayRef<Expr *> UnresolvedReductions = llvm::None);
  /// Called on well-formed 'task_reduction' clause.
  OMPClause *ActOnOpenMPTaskReductionClause(
      ArrayRef<Expr *> VarList, SourceLocation StartLoc,
      SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc,
      CXXScopeSpec &ReductionIdScopeSpec,
      const DeclarationNameInfo &ReductionId,
      ArrayRef<Expr *> UnresolvedReductions = llvm::None);
  /// Called on well-formed 'in_reduction' clause.
  OMPClause *ActOnOpenMPInReductionClause(
      ArrayRef<Expr *> VarList, SourceLocation StartLoc,
      SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc,
      CXXScopeSpec &ReductionIdScopeSpec,
      const DeclarationNameInfo &ReductionId,
      ArrayRef<Expr *> UnresolvedReductions = llvm::None);
  /// Called on well-formed 'linear' clause.
  OMPClause *
  ActOnOpenMPLinearClause(ArrayRef<Expr *> VarList, Expr *Step,
                          SourceLocation StartLoc, SourceLocation LParenLoc,
                          OpenMPLinearClauseKind LinKind, SourceLocation LinLoc,
                          SourceLocation ColonLoc, SourceLocation EndLoc);
  /// Called on well-formed 'aligned' clause.
  OMPClause *ActOnOpenMPAlignedClause(ArrayRef<Expr *> VarList,
                                      Expr *Alignment,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation ColonLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'copyin' clause.
  OMPClause *ActOnOpenMPCopyinClause(ArrayRef<Expr *> VarList,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'copyprivate' clause.
  OMPClause *ActOnOpenMPCopyprivateClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc);
  /// Called on well-formed 'flush' pseudo clause.
  OMPClause *ActOnOpenMPFlushClause(ArrayRef<Expr *> VarList,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'depobj' pseudo clause.
  OMPClause *ActOnOpenMPDepobjClause(Expr *Depobj, SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'depend' clause.
  OMPClause *
  ActOnOpenMPDependClause(Expr *DepModifier, OpenMPDependClauseKind DepKind,
                          SourceLocation DepLoc, SourceLocation ColonLoc,
                          ArrayRef<Expr *> VarList, SourceLocation StartLoc,
                          SourceLocation LParenLoc, SourceLocation EndLoc);
  /// Called on well-formed 'device' clause.
  OMPClause *ActOnOpenMPDeviceClause(OpenMPDeviceClauseModifier Modifier,
                                     Expr *Device, SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation ModifierLoc,
                                     SourceLocation EndLoc);
  /// Called on well-formed 'map' clause.
  OMPClause *ActOnOpenMPMapClause(
      ArrayRef<OpenMPMapModifierKind> MapTypeModifiers,
      ArrayRef<SourceLocation> MapTypeModifiersLoc,
      CXXScopeSpec &MapperIdScopeSpec, DeclarationNameInfo &MapperId,
      OpenMPMapClauseKind MapType, bool IsMapTypeImplicit,
      SourceLocation MapLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
      const OMPVarListLocTy &Locs, bool NoDiagnose = false,
      ArrayRef<Expr *> UnresolvedMappers = llvm::None);
  /// Called on well-formed 'num_teams' clause.
  OMPClause *ActOnOpenMPNumTeamsClause(Expr *NumTeams, SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed 'thread_limit' clause.
  OMPClause *ActOnOpenMPThreadLimitClause(Expr *ThreadLimit,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc);
  /// Called on well-formed 'priority' clause.
  OMPClause *ActOnOpenMPPriorityClause(Expr *Priority, SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed 'dist_schedule' clause.
  OMPClause *ActOnOpenMPDistScheduleClause(
      OpenMPDistScheduleClauseKind Kind, Expr *ChunkSize,
      SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation KindLoc,
      SourceLocation CommaLoc, SourceLocation EndLoc);
  /// Called on well-formed 'defaultmap' clause.
  OMPClause *ActOnOpenMPDefaultmapClause(
      OpenMPDefaultmapClauseModifier M, OpenMPDefaultmapClauseKind Kind,
      SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation MLoc,
      SourceLocation KindLoc, SourceLocation EndLoc);
  /// Called on well-formed 'to' clause.
  OMPClause *
  ActOnOpenMPToClause(ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
                      ArrayRef<SourceLocation> MotionModifiersLoc,
                      CXXScopeSpec &MapperIdScopeSpec,
                      DeclarationNameInfo &MapperId, SourceLocation ColonLoc,
                      ArrayRef<Expr *> VarList, const OMPVarListLocTy &Locs,
                      ArrayRef<Expr *> UnresolvedMappers = llvm::None);
  /// Called on well-formed 'from' clause.
  OMPClause *
  ActOnOpenMPFromClause(ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
                        ArrayRef<SourceLocation> MotionModifiersLoc,
                        CXXScopeSpec &MapperIdScopeSpec,
                        DeclarationNameInfo &MapperId, SourceLocation ColonLoc,
                        ArrayRef<Expr *> VarList, const OMPVarListLocTy &Locs,
                        ArrayRef<Expr *> UnresolvedMappers = llvm::None);
  /// Called on well-formed 'use_device_ptr' clause.
  OMPClause *ActOnOpenMPUseDevicePtrClause(ArrayRef<Expr *> VarList,
                                           const OMPVarListLocTy &Locs);
  /// Called on well-formed 'use_device_addr' clause.
  OMPClause *ActOnOpenMPUseDeviceAddrClause(ArrayRef<Expr *> VarList,
                                            const OMPVarListLocTy &Locs);
  /// Called on well-formed 'is_device_ptr' clause.
  OMPClause *ActOnOpenMPIsDevicePtrClause(ArrayRef<Expr *> VarList,
                                          const OMPVarListLocTy &Locs);
  /// Called on well-formed 'nontemporal' clause.
  OMPClause *ActOnOpenMPNontemporalClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc);

  /// Data for list of allocators.
  struct UsesAllocatorsData {
    /// Allocator.
    Expr *Allocator = nullptr;
    /// Allocator traits.
    Expr *AllocatorTraits = nullptr;
    /// Locations of '(' and ')' symbols.
    SourceLocation LParenLoc, RParenLoc;
  };
  /// Called on well-formed 'uses_allocators' clause.
  OMPClause *ActOnOpenMPUsesAllocatorClause(SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc,
                                            ArrayRef<UsesAllocatorsData> Data);
  /// Called on well-formed 'affinity' clause.
  OMPClause *ActOnOpenMPAffinityClause(SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation ColonLoc,
                                       SourceLocation EndLoc, Expr *Modifier,
                                       ArrayRef<Expr *> Locators);
  /// Called on a well-formed 'bind' clause.
  OMPClause *ActOnOpenMPBindClause(OpenMPBindClauseKind Kind,
                                   SourceLocation KindLoc,
                                   SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc);

  /// The kind of conversion being performed.
  enum CheckedConversionKind {
    /// An implicit conversion.
    CCK_ImplicitConversion,
    /// A C-style cast.
    CCK_CStyleCast,
    /// A functional-style cast.
    CCK_FunctionalCast,
    /// A cast other than a C-style cast.
    CCK_OtherCast,
    /// A conversion for an operand of a builtin overloaded operator.
    CCK_ForBuiltinOverloadedOp
  };

  static bool isCast(CheckedConversionKind CCK) {
    return CCK == CCK_CStyleCast || CCK == CCK_FunctionalCast ||
           CCK == CCK_OtherCast;
  }

  /// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit
  /// cast.  If there is already an implicit cast, merge into the existing one.
  /// If isLvalue, the result of the cast is an lvalue.
  ExprResult
  ImpCastExprToType(Expr *E, QualType Type, CastKind CK,
                    ExprValueKind VK = VK_PRValue,
                    const CXXCastPath *BasePath = nullptr,
                    CheckedConversionKind CCK = CCK_ImplicitConversion);

  /// ScalarTypeToBooleanCastKind - Returns the cast kind corresponding
  /// to the conversion from scalar type ScalarTy to the Boolean type.
  static CastKind ScalarTypeToBooleanCastKind(QualType ScalarTy);

  /// IgnoredValueConversions - Given that an expression's result is
  /// syntactically ignored, perform any conversions that are
  /// required.
  ExprResult IgnoredValueConversions(Expr *E);

  // UsualUnaryConversions - promotes integers (C99 6.3.1.1p2) and converts
  // functions and arrays to their respective pointers (C99 6.3.2.1).
  ExprResult UsualUnaryConversions(Expr *E);

  /// CallExprUnaryConversions - a special case of an unary conversion
  /// performed on a function designator of a call expression.
  ExprResult CallExprUnaryConversions(Expr *E);

  // DefaultFunctionArrayConversion - converts functions and arrays
  // to their respective pointers (C99 6.3.2.1).
  ExprResult DefaultFunctionArrayConversion(Expr *E, bool Diagnose = true);

  // DefaultFunctionArrayLvalueConversion - converts functions and
  // arrays to their respective pointers and performs the
  // lvalue-to-rvalue conversion.
  ExprResult DefaultFunctionArrayLvalueConversion(Expr *E,
                                                  bool Diagnose = true);

  // DefaultLvalueConversion - performs lvalue-to-rvalue conversion on
  // the operand. This function is a no-op if the operand has a function type
  // or an array type.
  ExprResult DefaultLvalueConversion(Expr *E);

  // DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
  // do not have a prototype. Integer promotions are performed on each
  // argument, and arguments that have type float are promoted to double.
  ExprResult DefaultArgumentPromotion(Expr *E);

  /// If \p E is a prvalue denoting an unmaterialized temporary, materialize
  /// it as an xvalue. In C++98, the result will still be a prvalue, because
  /// we don't have xvalues there.
  ExprResult TemporaryMaterializationConversion(Expr *E);

  // Used for emitting the right warning by DefaultVariadicArgumentPromotion
  enum VariadicCallType {
    VariadicFunction,
    VariadicBlock,
    VariadicMethod,
    VariadicConstructor,
    VariadicDoesNotApply
  };

  VariadicCallType getVariadicCallType(FunctionDecl *FDecl,
                                       const FunctionProtoType *Proto,
                                       Expr *Fn);

  // Used for determining in which context a type is allowed to be passed to a
  // vararg function.
  enum VarArgKind {
    VAK_Valid,
    VAK_ValidInCXX11,
    VAK_Undefined,
    VAK_MSVCUndefined,
    VAK_Invalid
  };

  // Determines which VarArgKind fits an expression.
  VarArgKind isValidVarArgType(const QualType &Ty);

  /// Check to see if the given expression is a valid argument to a variadic
  /// function, issuing a diagnostic if not.
  void checkVariadicArgument(const Expr *E, VariadicCallType CT);

  /// Check whether the given statement can have musttail applied to it,
  /// issuing a diagnostic and returning false if not. In the success case,
  /// the statement is rewritten to remove implicit nodes from the return
  /// value.
  bool checkAndRewriteMustTailAttr(Stmt *St, const Attr &MTA);

private:
  /// Check whether the given statement can have musttail applied to it,
  /// issuing a diagnostic and returning false if not.
  bool checkMustTailAttr(const Stmt *St, const Attr &MTA);

public:
  /// Check to see if a given expression could have '.c_str()' called on it.
  bool hasCStrMethod(const Expr *E);

  /// GatherArgumentsForCall - Collector argument expressions for various
  /// form of call prototypes.
  bool GatherArgumentsForCall(SourceLocation CallLoc, FunctionDecl *FDecl,
                              const FunctionProtoType *Proto,
                              unsigned FirstParam, ArrayRef<Expr *> Args,
                              SmallVectorImpl<Expr *> &AllArgs,
                              VariadicCallType CallType = VariadicDoesNotApply,
                              bool AllowExplicit = false,
                              bool IsListInitialization = false);

  // DefaultVariadicArgumentPromotion - Like DefaultArgumentPromotion, but
  // will create a runtime trap if the resulting type is not a POD type.
  ExprResult DefaultVariadicArgumentPromotion(Expr *E, VariadicCallType CT,
                                              FunctionDecl *FDecl);

  /// Context in which we're performing a usual arithmetic conversion.
  enum ArithConvKind {
    /// An arithmetic operation.
    ACK_Arithmetic,
    /// A bitwise operation.
    ACK_BitwiseOp,
    /// A comparison.
    ACK_Comparison,
    /// A conditional (?:) operator.
    ACK_Conditional,
    /// A compound assignment expression.
    ACK_CompAssign,
  };

  // UsualArithmeticConversions - performs the UsualUnaryConversions on it's
  // operands and then handles various conversions that are common to binary
  // operators (C99 6.3.1.8). If both operands aren't arithmetic, this
  // routine returns the first non-arithmetic type found. The client is
  // responsible for emitting appropriate error diagnostics.
  QualType UsualArithmeticConversions(ExprResult &LHS, ExprResult &RHS,
                                      SourceLocation Loc, ArithConvKind ACK);

  /// AssignConvertType - All of the 'assignment' semantic checks return this
  /// enum to indicate whether the assignment was allowed.  These checks are
  /// done for simple assignments, as well as initialization, return from
  /// function, argument passing, etc.  The query is phrased in terms of a
  /// source and destination type.
  enum AssignConvertType {
    /// Compatible - the types are compatible according to the standard.
    Compatible,

    /// PointerToInt - The assignment converts a pointer to an int, which we
    /// accept as an extension.
    PointerToInt,

    /// IntToPointer - The assignment converts an int to a pointer, which we
    /// accept as an extension.
    IntToPointer,

    /// FunctionVoidPointer - The assignment is between a function pointer and
    /// void*, which the standard doesn't allow, but we accept as an extension.
    FunctionVoidPointer,

    /// IncompatiblePointer - The assignment is between two pointers types that
    /// are not compatible, but we accept them as an extension.
    IncompatiblePointer,

    /// IncompatibleFunctionPointer - The assignment is between two function
    /// pointers types that are not compatible, but we accept them as an
    /// extension.
    IncompatibleFunctionPointer,

    /// IncompatiblePointerSign - The assignment is between two pointers types
    /// which point to integers which have a different sign, but are otherwise
    /// identical. This is a subset of the above, but broken out because it's by
    /// far the most common case of incompatible pointers.
    IncompatiblePointerSign,

    /// CompatiblePointerDiscardsQualifiers - The assignment discards
    /// c/v/r qualifiers, which we accept as an extension.
    CompatiblePointerDiscardsQualifiers,

    /// IncompatiblePointerDiscardsQualifiers - The assignment
    /// discards qualifiers that we don't permit to be discarded,
    /// like address spaces.
    IncompatiblePointerDiscardsQualifiers,

    /// IncompatibleNestedPointerAddressSpaceMismatch - The assignment
    /// changes address spaces in nested pointer types which is not allowed.
    /// For instance, converting __private int ** to __generic int ** is
    /// illegal even though __private could be converted to __generic.
    IncompatibleNestedPointerAddressSpaceMismatch,

    /// IncompatibleNestedPointerQualifiers - The assignment is between two
    /// nested pointer types, and the qualifiers other than the first two
    /// levels differ e.g. char ** -> const char **, but we accept them as an
    /// extension.
    IncompatibleNestedPointerQualifiers,

    /// IncompatibleVectors - The assignment is between two vector types that
    /// have the same size, which we accept as an extension.
    IncompatibleVectors,

    /// IntToBlockPointer - The assignment converts an int to a block
    /// pointer. We disallow this.
    IntToBlockPointer,

    /// IncompatibleBlockPointer - The assignment is between two block
    /// pointers types that are not compatible.
    IncompatibleBlockPointer,

    /// IncompatibleObjCQualifiedId - The assignment is between a qualified
    /// id type and something else (that is incompatible with it). For example,
    /// "id <XXX>" = "Foo *", where "Foo *" doesn't implement the XXX protocol.
    IncompatibleObjCQualifiedId,

    /// IncompatibleObjCWeakRef - Assigning a weak-unavailable object to an
    /// object with __weak qualifier.
    IncompatibleObjCWeakRef,

    /// Incompatible - We reject this conversion outright, it is invalid to
    /// represent it in the AST.
    Incompatible
  };

  /// DiagnoseAssignmentResult - Emit a diagnostic, if required, for the
  /// assignment conversion type specified by ConvTy.  This returns true if the
  /// conversion was invalid or false if the conversion was accepted.
  bool DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                SourceLocation Loc,
                                QualType DstType, QualType SrcType,
                                Expr *SrcExpr, AssignmentAction Action,
                                bool *Complained = nullptr);

  /// IsValueInFlagEnum - Determine if a value is allowed as part of a flag
  /// enum. If AllowMask is true, then we also allow the complement of a valid
  /// value, to be used as a mask.
  bool IsValueInFlagEnum(const EnumDecl *ED, const llvm::APInt &Val,
                         bool AllowMask) const;

  /// DiagnoseAssignmentEnum - Warn if assignment to enum is a constant
  /// integer not in the range of enum values.
  void DiagnoseAssignmentEnum(QualType DstType, QualType SrcType,
                              Expr *SrcExpr);

  /// CheckAssignmentConstraints - Perform type checking for assignment,
  /// argument passing, variable initialization, and function return values.
  /// C99 6.5.16.
  AssignConvertType CheckAssignmentConstraints(SourceLocation Loc,
                                               QualType LHSType,
                                               QualType RHSType);

  /// Check assignment constraints and optionally prepare for a conversion of
  /// the RHS to the LHS type. The conversion is prepared for if ConvertRHS
  /// is true.
  AssignConvertType CheckAssignmentConstraints(QualType LHSType,
                                               ExprResult &RHS,
                                               CastKind &Kind,
                                               bool ConvertRHS = true);

  /// Check assignment constraints for an assignment of RHS to LHSType.
  ///
  /// \param LHSType The destination type for the assignment.
  /// \param RHS The source expression for the assignment.
  /// \param Diagnose If \c true, diagnostics may be produced when checking
  ///        for assignability. If a diagnostic is produced, \p RHS will be
  ///        set to ExprError(). Note that this function may still return
  ///        without producing a diagnostic, even for an invalid assignment.
  /// \param DiagnoseCFAudited If \c true, the target is a function parameter
  ///        in an audited Core Foundation API and does not need to be checked
  ///        for ARC retain issues.
  /// \param ConvertRHS If \c true, \p RHS will be updated to model the
  ///        conversions necessary to perform the assignment. If \c false,
  ///        \p Diagnose must also be \c false.
  AssignConvertType CheckSingleAssignmentConstraints(
      QualType LHSType, ExprResult &RHS, bool Diagnose = true,
      bool DiagnoseCFAudited = false, bool ConvertRHS = true);

  // If the lhs type is a transparent union, check whether we
  // can initialize the transparent union with the given expression.
  AssignConvertType CheckTransparentUnionArgumentConstraints(QualType ArgType,
                                                             ExprResult &RHS);

  bool IsStringLiteralToNonConstPointerConversion(Expr *From, QualType ToType);

  bool CheckExceptionSpecCompatibility(Expr *From, QualType ToType);

  ExprResult PerformImplicitConversion(Expr *From, QualType ToType,
                                       AssignmentAction Action,
                                       bool AllowExplicit = false);
  ExprResult PerformImplicitConversion(Expr *From, QualType ToType,
                                       const ImplicitConversionSequence& ICS,
                                       AssignmentAction Action,
                                       CheckedConversionKind CCK
                                          = CCK_ImplicitConversion);
  ExprResult PerformImplicitConversion(Expr *From, QualType ToType,
                                       const StandardConversionSequence& SCS,
                                       AssignmentAction Action,
                                       CheckedConversionKind CCK);

  ExprResult PerformQualificationConversion(
      Expr *E, QualType Ty, ExprValueKind VK = VK_PRValue,
      CheckedConversionKind CCK = CCK_ImplicitConversion);

  /// the following "Check" methods will return a valid/converted QualType
  /// or a null QualType (indicating an error diagnostic was issued).

  /// type checking binary operators (subroutines of CreateBuiltinBinOp).
  QualType InvalidOperands(SourceLocation Loc, ExprResult &LHS,
                           ExprResult &RHS);
  QualType InvalidLogicalVectorOperands(SourceLocation Loc, ExprResult &LHS,
                                 ExprResult &RHS);
  QualType CheckPointerToMemberOperands( // C++ 5.5
    ExprResult &LHS, ExprResult &RHS, ExprValueKind &VK,
    SourceLocation OpLoc, bool isIndirect);
  QualType CheckMultiplyDivideOperands( // C99 6.5.5
    ExprResult &LHS, ExprResult &RHS, SourceLocation Loc, bool IsCompAssign,
    bool IsDivide);
  QualType CheckRemainderOperands( // C99 6.5.5
    ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
    bool IsCompAssign = false);
  QualType CheckAdditionOperands( // C99 6.5.6
    ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
    BinaryOperatorKind Opc, QualType* CompLHSTy = nullptr);
  QualType CheckSubtractionOperands( // C99 6.5.6
    ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
    QualType* CompLHSTy = nullptr);
  QualType CheckShiftOperands( // C99 6.5.7
    ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
    BinaryOperatorKind Opc, bool IsCompAssign = false);
  void CheckPtrComparisonWithNullChar(ExprResult &E, ExprResult &NullE);
  QualType CheckCompareOperands( // C99 6.5.8/9
      ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
      BinaryOperatorKind Opc);
  QualType CheckBitwiseOperands( // C99 6.5.[10...12]
      ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
      BinaryOperatorKind Opc);
  QualType CheckLogicalOperands( // C99 6.5.[13,14]
    ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
    BinaryOperatorKind Opc);
  // CheckAssignmentOperands is used for both simple and compound assignment.
  // For simple assignment, pass both expressions and a null converted type.
  // For compound assignment, pass both expressions and the converted type.
  QualType CheckAssignmentOperands( // C99 6.5.16.[1,2]
    Expr *LHSExpr, ExprResult &RHS, SourceLocation Loc, QualType CompoundType);

  ExprResult checkPseudoObjectIncDec(Scope *S, SourceLocation OpLoc,
                                     UnaryOperatorKind Opcode, Expr *Op);
  ExprResult checkPseudoObjectAssignment(Scope *S, SourceLocation OpLoc,
                                         BinaryOperatorKind Opcode,
                                         Expr *LHS, Expr *RHS);
  ExprResult checkPseudoObjectRValue(Expr *E);
  Expr *recreateSyntacticForm(PseudoObjectExpr *E);

  QualType CheckConditionalOperands( // C99 6.5.15
    ExprResult &Cond, ExprResult &LHS, ExprResult &RHS,
    ExprValueKind &VK, ExprObjectKind &OK, SourceLocation QuestionLoc);
  QualType CXXCheckConditionalOperands( // C++ 5.16
    ExprResult &cond, ExprResult &lhs, ExprResult &rhs,
    ExprValueKind &VK, ExprObjectKind &OK, SourceLocation questionLoc);
  QualType CheckVectorConditionalTypes(ExprResult &Cond, ExprResult &LHS,
                                       ExprResult &RHS,
                                       SourceLocation QuestionLoc);
  QualType FindCompositePointerType(SourceLocation Loc, Expr *&E1, Expr *&E2,
                                    bool ConvertArgs = true);
  QualType FindCompositePointerType(SourceLocation Loc,
                                    ExprResult &E1, ExprResult &E2,
                                    bool ConvertArgs = true) {
    Expr *E1Tmp = E1.get(), *E2Tmp = E2.get();
    QualType Composite =
        FindCompositePointerType(Loc, E1Tmp, E2Tmp, ConvertArgs);
    E1 = E1Tmp;
    E2 = E2Tmp;
    return Composite;
  }

  QualType FindCompositeObjCPointerType(ExprResult &LHS, ExprResult &RHS,
                                        SourceLocation QuestionLoc);

  bool DiagnoseConditionalForNull(Expr *LHSExpr, Expr *RHSExpr,
                                  SourceLocation QuestionLoc);

  void DiagnoseAlwaysNonNullPointer(Expr *E,
                                    Expr::NullPointerConstantKind NullType,
                                    bool IsEqual, SourceRange Range);

  /// type checking for vector binary operators.
  QualType CheckVectorOperands(ExprResult &LHS, ExprResult &RHS,
                               SourceLocation Loc, bool IsCompAssign,
                               bool AllowBothBool, bool AllowBoolConversion);
  QualType GetSignedVectorType(QualType V);
  QualType CheckVectorCompareOperands(ExprResult &LHS, ExprResult &RHS,
                                      SourceLocation Loc,
                                      BinaryOperatorKind Opc);
  QualType CheckVectorLogicalOperands(ExprResult &LHS, ExprResult &RHS,
                                      SourceLocation Loc);

  /// Type checking for matrix binary operators.
  QualType CheckMatrixElementwiseOperands(ExprResult &LHS, ExprResult &RHS,
                                          SourceLocation Loc,
                                          bool IsCompAssign);
  QualType CheckMatrixMultiplyOperands(ExprResult &LHS, ExprResult &RHS,
                                       SourceLocation Loc, bool IsCompAssign);

  bool isValidSveBitcast(QualType srcType, QualType destType);

  bool areMatrixTypesOfTheSameDimension(QualType srcTy, QualType destTy);

  bool areVectorTypesSameSize(QualType srcType, QualType destType);
  bool areLaxCompatibleVectorTypes(QualType srcType, QualType destType);
  bool isLaxVectorConversion(QualType srcType, QualType destType);

  /// type checking declaration initializers (C99 6.7.8)
  bool CheckForConstantInitializer(Expr *e, QualType t);

  // type checking C++ declaration initializers (C++ [dcl.init]).

  /// ReferenceCompareResult - Expresses the result of comparing two
  /// types (cv1 T1 and cv2 T2) to determine their compatibility for the
  /// purposes of initialization by reference (C++ [dcl.init.ref]p4).
  enum ReferenceCompareResult {
    /// Ref_Incompatible - The two types are incompatible, so direct
    /// reference binding is not possible.
    Ref_Incompatible = 0,
    /// Ref_Related - The two types are reference-related, which means
    /// that their unqualified forms (T1 and T2) are either the same
    /// or T1 is a base class of T2.
    Ref_Related,
    /// Ref_Compatible - The two types are reference-compatible.
    Ref_Compatible
  };

  // Fake up a scoped enumeration that still contextually converts to bool.
  struct ReferenceConversionsScope {
    /// The conversions that would be performed on an lvalue of type T2 when
    /// binding a reference of type T1 to it, as determined when evaluating
    /// whether T1 is reference-compatible with T2.
    enum ReferenceConversions {
      Qualification = 0x1,
      NestedQualification = 0x2,
      Function = 0x4,
      DerivedToBase = 0x8,
      ObjC = 0x10,
      ObjCLifetime = 0x20,

      LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/ObjCLifetime)
    };
  };
  using ReferenceConversions = ReferenceConversionsScope::ReferenceConversions;

  ReferenceCompareResult
  CompareReferenceRelationship(SourceLocation Loc, QualType T1, QualType T2,
                               ReferenceConversions *Conv = nullptr);

  ExprResult checkUnknownAnyCast(SourceRange TypeRange, QualType CastType,
                                 Expr *CastExpr, CastKind &CastKind,
                                 ExprValueKind &VK, CXXCastPath &Path);

  /// Force an expression with unknown-type to an expression of the
  /// given type.
  ExprResult forceUnknownAnyToType(Expr *E, QualType ToType);

  /// Type-check an expression that's being passed to an
  /// __unknown_anytype parameter.
  ExprResult checkUnknownAnyArg(SourceLocation callLoc,
                                Expr *result, QualType &paramType);

  // CheckMatrixCast - Check type constraints for matrix casts.
  // We allow casting between matrixes of the same dimensions i.e. when they
  // have the same number of rows and column. Returns true if the cast is
  // invalid.
  bool CheckMatrixCast(SourceRange R, QualType DestTy, QualType SrcTy,
                       CastKind &Kind);

  // CheckVectorCast - check type constraints for vectors.
  // Since vectors are an extension, there are no C standard reference for this.
  // We allow casting between vectors and integer datatypes of the same size.
  // returns true if the cast is invalid
  bool CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty,
                       CastKind &Kind);

  /// Prepare `SplattedExpr` for a vector splat operation, adding
  /// implicit casts if necessary.
  ExprResult prepareVectorSplat(QualType VectorTy, Expr *SplattedExpr);

  // CheckExtVectorCast - check type constraints for extended vectors.
  // Since vectors are an extension, there are no C standard reference for this.
  // We allow casting between vectors and integer datatypes of the same size,
  // or vectors and the element type of that vector.
  // returns the cast expr
  ExprResult CheckExtVectorCast(SourceRange R, QualType DestTy, Expr *CastExpr,
                                CastKind &Kind);

  ExprResult BuildCXXFunctionalCastExpr(TypeSourceInfo *TInfo, QualType Type,
                                        SourceLocation LParenLoc,
                                        Expr *CastExpr,
                                        SourceLocation RParenLoc);

  enum ARCConversionResult { ACR_okay, ACR_unbridged, ACR_error };

  /// Checks for invalid conversions and casts between
  /// retainable pointers and other pointer kinds for ARC and Weak.
  ARCConversionResult CheckObjCConversion(SourceRange castRange,
                                          QualType castType, Expr *&op,
                                          CheckedConversionKind CCK,
                                          bool Diagnose = true,
                                          bool DiagnoseCFAudited = false,
                                          BinaryOperatorKind Opc = BO_PtrMemD
                                          );

  Expr *stripARCUnbridgedCast(Expr *e);
  void diagnoseARCUnbridgedCast(Expr *e);

  bool CheckObjCARCUnavailableWeakConversion(QualType castType,
                                             QualType ExprType);

  /// checkRetainCycles - Check whether an Objective-C message send
  /// might create an obvious retain cycle.
  void checkRetainCycles(ObjCMessageExpr *msg);
  void checkRetainCycles(Expr *receiver, Expr *argument);
  void checkRetainCycles(VarDecl *Var, Expr *Init);

  /// checkUnsafeAssigns - Check whether +1 expr is being assigned
  /// to weak/__unsafe_unretained type.
  bool checkUnsafeAssigns(SourceLocation Loc, QualType LHS, Expr *RHS);

  /// checkUnsafeExprAssigns - Check whether +1 expr is being assigned
  /// to weak/__unsafe_unretained expression.
  void checkUnsafeExprAssigns(SourceLocation Loc, Expr *LHS, Expr *RHS);

  /// CheckMessageArgumentTypes - Check types in an Obj-C message send.
  /// \param Method - May be null.
  /// \param [out] ReturnType - The return type of the send.
  /// \return true iff there were any incompatible types.
  bool CheckMessageArgumentTypes(const Expr *Receiver, QualType ReceiverType,
                                 MultiExprArg Args, Selector Sel,
                                 ArrayRef<SourceLocation> SelectorLocs,
                                 ObjCMethodDecl *Method, bool isClassMessage,
                                 bool isSuperMessage, SourceLocation lbrac,
                                 SourceLocation rbrac, SourceRange RecRange,
                                 QualType &ReturnType, ExprValueKind &VK);

  /// Determine the result of a message send expression based on
  /// the type of the receiver, the method expected to receive the message,
  /// and the form of the message send.
  QualType getMessageSendResultType(const Expr *Receiver, QualType ReceiverType,
                                    ObjCMethodDecl *Method, bool isClassMessage,
                                    bool isSuperMessage);

  /// If the given expression involves a message send to a method
  /// with a related result type, emit a note describing what happened.
  void EmitRelatedResultTypeNote(const Expr *E);

  /// Given that we had incompatible pointer types in a return
  /// statement, check whether we're in a method with a related result
  /// type, and if so, emit a note describing what happened.
  void EmitRelatedResultTypeNoteForReturn(QualType destType);

  class ConditionResult {
    Decl *ConditionVar;
    FullExprArg Condition;
    bool Invalid;
    bool HasKnownValue;
    bool KnownValue;

    friend class Sema;
    ConditionResult(Sema &S, Decl *ConditionVar, FullExprArg Condition,
                    bool IsConstexpr)
        : ConditionVar(ConditionVar), Condition(Condition), Invalid(false),
          HasKnownValue(IsConstexpr && Condition.get() &&
                        !Condition.get()->isValueDependent()),
          KnownValue(HasKnownValue &&
                     !!Condition.get()->EvaluateKnownConstInt(S.Context)) {}
    explicit ConditionResult(bool Invalid)
        : ConditionVar(nullptr), Condition(nullptr), Invalid(Invalid),
          HasKnownValue(false), KnownValue(false) {}

  public:
    ConditionResult() : ConditionResult(false) {}
    bool isInvalid() const { return Invalid; }
    std::pair<VarDecl *, Expr *> get() const {
      return std::make_pair(cast_or_null<VarDecl>(ConditionVar),
                            Condition.get());
    }
    llvm::Optional<bool> getKnownValue() const {
      if (!HasKnownValue)
        return None;
      return KnownValue;
    }
  };
  static ConditionResult ConditionError() { return ConditionResult(true); }

  enum class ConditionKind {
    Boolean,     ///< A boolean condition, from 'if', 'while', 'for', or 'do'.
    ConstexprIf, ///< A constant boolean condition from 'if constexpr'.
    Switch       ///< An integral condition for a 'switch' statement.
  };
  QualType PreferredConditionType(ConditionKind K) const {
    return K == ConditionKind::Switch ? Context.IntTy : Context.BoolTy;
  }

  ConditionResult ActOnCondition(Scope *S, SourceLocation Loc, Expr *SubExpr,
                                 ConditionKind CK, bool MissingOK = false);

  ConditionResult ActOnConditionVariable(Decl *ConditionVar,
                                         SourceLocation StmtLoc,
                                         ConditionKind CK);

  DeclResult ActOnCXXConditionDeclaration(Scope *S, Declarator &D);

  ExprResult CheckConditionVariable(VarDecl *ConditionVar,
                                    SourceLocation StmtLoc,
                                    ConditionKind CK);
  ExprResult CheckSwitchCondition(SourceLocation SwitchLoc, Expr *Cond);

  /// CheckBooleanCondition - Diagnose problems involving the use of
  /// the given expression as a boolean condition (e.g. in an if
  /// statement).  Also performs the standard function and array
  /// decays, possibly changing the input variable.
  ///
  /// \param Loc - A location associated with the condition, e.g. the
  /// 'if' keyword.
  /// \return true iff there were any errors
  ExprResult CheckBooleanCondition(SourceLocation Loc, Expr *E,
                                   bool IsConstexpr = false);

  /// ActOnExplicitBoolSpecifier - Build an ExplicitSpecifier from an expression
  /// found in an explicit(bool) specifier.
  ExplicitSpecifier ActOnExplicitBoolSpecifier(Expr *E);

  /// tryResolveExplicitSpecifier - Attempt to resolve the explict specifier.
  /// Returns true if the explicit specifier is now resolved.
  bool tryResolveExplicitSpecifier(ExplicitSpecifier &ExplicitSpec);

  /// DiagnoseAssignmentAsCondition - Given that an expression is
  /// being used as a boolean condition, warn if it's an assignment.
  void DiagnoseAssignmentAsCondition(Expr *E);

  /// Redundant parentheses over an equality comparison can indicate
  /// that the user intended an assignment used as condition.
  void DiagnoseEqualityWithExtraParens(ParenExpr *ParenE);

  /// CheckCXXBooleanCondition - Returns true if conversion to bool is invalid.
  ExprResult CheckCXXBooleanCondition(Expr *CondExpr, bool IsConstexpr = false);

  /// ConvertIntegerToTypeWarnOnOverflow - Convert the specified APInt to have
  /// the specified width and sign.  If an overflow occurs, detect it and emit
  /// the specified diagnostic.
  void ConvertIntegerToTypeWarnOnOverflow(llvm::APSInt &OldVal,
                                          unsigned NewWidth, bool NewSign,
                                          SourceLocation Loc, unsigned DiagID);

  /// Checks that the Objective-C declaration is declared in the global scope.
  /// Emits an error and marks the declaration as invalid if it's not declared
  /// in the global scope.
  bool CheckObjCDeclScope(Decl *D);

  /// Abstract base class used for diagnosing integer constant
  /// expression violations.
  class VerifyICEDiagnoser {
  public:
    bool Suppress;

    VerifyICEDiagnoser(bool Suppress = false) : Suppress(Suppress) { }

    virtual SemaDiagnosticBuilder
    diagnoseNotICEType(Sema &S, SourceLocation Loc, QualType T);
    virtual SemaDiagnosticBuilder diagnoseNotICE(Sema &S,
                                                 SourceLocation Loc) = 0;
    virtual SemaDiagnosticBuilder diagnoseFold(Sema &S, SourceLocation Loc);
    virtual ~VerifyICEDiagnoser() {}
  };

  enum AllowFoldKind {
    NoFold,
    AllowFold,
  };

  /// VerifyIntegerConstantExpression - Verifies that an expression is an ICE,
  /// and reports the appropriate diagnostics. Returns false on success.
  /// Can optionally return the value of the expression.
  ExprResult VerifyIntegerConstantExpression(Expr *E, llvm::APSInt *Result,
                                             VerifyICEDiagnoser &Diagnoser,
                                             AllowFoldKind CanFold = NoFold);
  ExprResult VerifyIntegerConstantExpression(Expr *E, llvm::APSInt *Result,
                                             unsigned DiagID,
                                             AllowFoldKind CanFold = NoFold);
  ExprResult VerifyIntegerConstantExpression(Expr *E,
                                             llvm::APSInt *Result = nullptr,
                                             AllowFoldKind CanFold = NoFold);
  ExprResult VerifyIntegerConstantExpression(Expr *E,
                                             AllowFoldKind CanFold = NoFold) {
    return VerifyIntegerConstantExpression(E, nullptr, CanFold);
  }

  /// VerifyBitField - verifies that a bit field expression is an ICE and has
  /// the correct width, and that the field type is valid.
  /// Returns false on success.
  /// Can optionally return whether the bit-field is of width 0
  ExprResult VerifyBitField(SourceLocation FieldLoc, IdentifierInfo *FieldName,
                            QualType FieldTy, bool IsMsStruct,
                            Expr *BitWidth, bool *ZeroWidth = nullptr);

private:
  unsigned ForceCUDAHostDeviceDepth = 0;

public:
  /// Increments our count of the number of times we've seen a pragma forcing
  /// functions to be __host__ __device__.  So long as this count is greater
  /// than zero, all functions encountered will be __host__ __device__.
  void PushForceCUDAHostDevice();

  /// Decrements our count of the number of times we've seen a pragma forcing
  /// functions to be __host__ __device__.  Returns false if the count is 0
  /// before incrementing, so you can emit an error.
  bool PopForceCUDAHostDevice();

  /// Diagnostics that are emitted only if we discover that the given function
  /// must be codegen'ed.  Because handling these correctly adds overhead to
  /// compilation, this is currently only enabled for CUDA compilations.
  llvm::DenseMap<CanonicalDeclPtr<FunctionDecl>,
                 std::vector<PartialDiagnosticAt>>
      DeviceDeferredDiags;

  /// A pair of a canonical FunctionDecl and a SourceLocation.  When used as the
  /// key in a hashtable, both the FD and location are hashed.
  struct FunctionDeclAndLoc {
    CanonicalDeclPtr<FunctionDecl> FD;
    SourceLocation Loc;
  };

  /// FunctionDecls and SourceLocations for which CheckCUDACall has emitted a
  /// (maybe deferred) "bad call" diagnostic.  We use this to avoid emitting the
  /// same deferred diag twice.
  llvm::DenseSet<FunctionDeclAndLoc> LocsWithCUDACallDiags;

  /// An inverse call graph, mapping known-emitted functions to one of their
  /// known-emitted callers (plus the location of the call).
  ///
  /// Functions that we can tell a priori must be emitted aren't added to this
  /// map.
  llvm::DenseMap</* Callee = */ CanonicalDeclPtr<FunctionDecl>,
                 /* Caller = */ FunctionDeclAndLoc>
      DeviceKnownEmittedFns;

  /// Creates a SemaDiagnosticBuilder that emits the diagnostic if the current
  /// context is "used as device code".
  ///
  /// - If CurContext is a __host__ function, does not emit any diagnostics
  ///   unless \p EmitOnBothSides is true.
  /// - If CurContext is a __device__ or __global__ function, emits the
  ///   diagnostics immediately.
  /// - If CurContext is a __host__ __device__ function and we are compiling for
  ///   the device, creates a diagnostic which is emitted if and when we realize
  ///   that the function will be codegen'ed.
  ///
  /// Example usage:
  ///
  ///  // Variable-length arrays are not allowed in CUDA device code.
  ///  if (CUDADiagIfDeviceCode(Loc, diag::err_cuda_vla) << CurrentCUDATarget())
  ///    return ExprError();
  ///  // Otherwise, continue parsing as normal.
  SemaDiagnosticBuilder CUDADiagIfDeviceCode(SourceLocation Loc,
                                             unsigned DiagID);

  /// Creates a SemaDiagnosticBuilder that emits the diagnostic if the current
  /// context is "used as host code".
  ///
  /// Same as CUDADiagIfDeviceCode, with "host" and "device" switched.
  SemaDiagnosticBuilder CUDADiagIfHostCode(SourceLocation Loc, unsigned DiagID);

  /// Creates a SemaDiagnosticBuilder that emits the diagnostic if the current
  /// context is "used as device code".
  ///
  /// - If CurContext is a `declare target` function or it is known that the
  /// function is emitted for the device, emits the diagnostics immediately.
  /// - If CurContext is a non-`declare target` function and we are compiling
  ///   for the device, creates a diagnostic which is emitted if and when we
  ///   realize that the function will be codegen'ed.
  ///
  /// Example usage:
  ///
  ///  // Variable-length arrays are not allowed in NVPTX device code.
  ///  if (diagIfOpenMPDeviceCode(Loc, diag::err_vla_unsupported))
  ///    return ExprError();
  ///  // Otherwise, continue parsing as normal.
  SemaDiagnosticBuilder
  diagIfOpenMPDeviceCode(SourceLocation Loc, unsigned DiagID, FunctionDecl *FD);

  /// Creates a SemaDiagnosticBuilder that emits the diagnostic if the current
  /// context is "used as host code".
  ///
  /// - If CurContext is a `declare target` function or it is known that the
  /// function is emitted for the host, emits the diagnostics immediately.
  /// - If CurContext is a non-host function, just ignore it.
  ///
  /// Example usage:
  ///
  ///  // Variable-length arrays are not allowed in NVPTX device code.
  ///  if (diagIfOpenMPHostode(Loc, diag::err_vla_unsupported))
  ///    return ExprError();
  ///  // Otherwise, continue parsing as normal.
  SemaDiagnosticBuilder diagIfOpenMPHostCode(SourceLocation Loc,
                                             unsigned DiagID, FunctionDecl *FD);

  SemaDiagnosticBuilder targetDiag(SourceLocation Loc, unsigned DiagID,
                                   FunctionDecl *FD = nullptr);
  SemaDiagnosticBuilder targetDiag(SourceLocation Loc,
                                   const PartialDiagnostic &PD,
                                   FunctionDecl *FD = nullptr) {
    return targetDiag(Loc, PD.getDiagID(), FD) << PD;
  }

  /// Check if the type is allowed to be used for the current target.
  void checkTypeSupport(QualType Ty, SourceLocation Loc,
                        ValueDecl *D = nullptr);

  enum CUDAFunctionTarget {
    CFT_Device,
    CFT_Global,
    CFT_Host,
    CFT_HostDevice,
    CFT_InvalidTarget
  };

  /// Determines whether the given function is a CUDA device/host/kernel/etc.
  /// function.
  ///
  /// Use this rather than examining the function's attributes yourself -- you
  /// will get it wrong.  Returns CFT_Host if D is null.
  CUDAFunctionTarget IdentifyCUDATarget(const FunctionDecl *D,
                                        bool IgnoreImplicitHDAttr = false);
  CUDAFunctionTarget IdentifyCUDATarget(const ParsedAttributesView &Attrs);

  enum CUDAVariableTarget {
    CVT_Device,  /// Emitted on device side with a shadow variable on host side
    CVT_Host,    /// Emitted on host side only
    CVT_Both,    /// Emitted on both sides with different addresses
    CVT_Unified, /// Emitted as a unified address, e.g. managed variables
  };
  /// Determines whether the given variable is emitted on host or device side.
  CUDAVariableTarget IdentifyCUDATarget(const VarDecl *D);

  /// Gets the CUDA target for the current context.
  CUDAFunctionTarget CurrentCUDATarget() {
    return IdentifyCUDATarget(dyn_cast<FunctionDecl>(CurContext));
  }

  static bool isCUDAImplicitHostDeviceFunction(const FunctionDecl *D);

  // CUDA function call preference. Must be ordered numerically from
  // worst to best.
  enum CUDAFunctionPreference {
    CFP_Never,      // Invalid caller/callee combination.
    CFP_WrongSide,  // Calls from host-device to host or device
                    // function that do not match current compilation
                    // mode.
    CFP_HostDevice, // Any calls to host/device functions.
    CFP_SameSide,   // Calls from host-device to host or device
                    // function matching current compilation mode.
    CFP_Native,     // host-to-host or device-to-device calls.
  };

  /// Identifies relative preference of a given Caller/Callee
  /// combination, based on their host/device attributes.
  /// \param Caller function which needs address of \p Callee.
  ///               nullptr in case of global context.
  /// \param Callee target function
  ///
  /// \returns preference value for particular Caller/Callee combination.
  CUDAFunctionPreference IdentifyCUDAPreference(const FunctionDecl *Caller,
                                                const FunctionDecl *Callee);

  /// Determines whether Caller may invoke Callee, based on their CUDA
  /// host/device attributes.  Returns false if the call is not allowed.
  ///
  /// Note: Will return true for CFP_WrongSide calls.  These may appear in
  /// semantically correct CUDA programs, but only if they're never codegen'ed.
  bool IsAllowedCUDACall(const FunctionDecl *Caller,
                         const FunctionDecl *Callee) {
    return IdentifyCUDAPreference(Caller, Callee) != CFP_Never;
  }

  /// May add implicit CUDAHostAttr and CUDADeviceAttr attributes to FD,
  /// depending on FD and the current compilation settings.
  void maybeAddCUDAHostDeviceAttrs(FunctionDecl *FD,
                                   const LookupResult &Previous);

  /// May add implicit CUDAConstantAttr attribute to VD, depending on VD
  /// and current compilation settings.
  void MaybeAddCUDAConstantAttr(VarDecl *VD);

public:
  /// Check whether we're allowed to call Callee from the current context.
  ///
  /// - If the call is never allowed in a semantically-correct program
  ///   (CFP_Never), emits an error and returns false.
  ///
  /// - If the call is allowed in semantically-correct programs, but only if
  ///   it's never codegen'ed (CFP_WrongSide), creates a deferred diagnostic to
  ///   be emitted if and when the caller is codegen'ed, and returns true.
  ///
  ///   Will only create deferred diagnostics for a given SourceLocation once,
  ///   so you can safely call this multiple times without generating duplicate
  ///   deferred errors.
  ///
  /// - Otherwise, returns true without emitting any diagnostics.
  bool CheckCUDACall(SourceLocation Loc, FunctionDecl *Callee);

  void CUDACheckLambdaCapture(CXXMethodDecl *D, const sema::Capture &Capture);

  /// Set __device__ or __host__ __device__ attributes on the given lambda
  /// operator() method.
  ///
  /// CUDA lambdas by default is host device function unless it has explicit
  /// host or device attribute.
  void CUDASetLambdaAttrs(CXXMethodDecl *Method);

  /// Finds a function in \p Matches with highest calling priority
  /// from \p Caller context and erases all functions with lower
  /// calling priority.
  void EraseUnwantedCUDAMatches(
      const FunctionDecl *Caller,
      SmallVectorImpl<std::pair<DeclAccessPair, FunctionDecl *>> &Matches);

  /// Given a implicit special member, infer its CUDA target from the
  /// calls it needs to make to underlying base/field special members.
  /// \param ClassDecl the class for which the member is being created.
  /// \param CSM the kind of special member.
  /// \param MemberDecl the special member itself.
  /// \param ConstRHS true if this is a copy operation with a const object on
  ///        its RHS.
  /// \param Diagnose true if this call should emit diagnostics.
  /// \return true if there was an error inferring.
  /// The result of this call is implicit CUDA target attribute(s) attached to
  /// the member declaration.
  bool inferCUDATargetForImplicitSpecialMember(CXXRecordDecl *ClassDecl,
                                               CXXSpecialMember CSM,
                                               CXXMethodDecl *MemberDecl,
                                               bool ConstRHS,
                                               bool Diagnose);

  /// \return true if \p CD can be considered empty according to CUDA
  /// (E.2.3.1 in CUDA 7.5 Programming guide).
  bool isEmptyCudaConstructor(SourceLocation Loc, CXXConstructorDecl *CD);
  bool isEmptyCudaDestructor(SourceLocation Loc, CXXDestructorDecl *CD);

  // \brief Checks that initializers of \p Var satisfy CUDA restrictions. In
  // case of error emits appropriate diagnostic and invalidates \p Var.
  //
  // \details CUDA allows only empty constructors as initializers for global
  // variables (see E.2.3.1, CUDA 7.5). The same restriction also applies to all
  // __shared__ variables whether they are local or not (they all are implicitly
  // static in CUDA). One exception is that CUDA allows constant initializers
  // for __constant__ and __device__ variables.
  void checkAllowedCUDAInitializer(VarDecl *VD);

  /// Check whether NewFD is a valid overload for CUDA. Emits
  /// diagnostics and invalidates NewFD if not.
  void checkCUDATargetOverload(FunctionDecl *NewFD,
                               const LookupResult &Previous);
  /// Copies target attributes from the template TD to the function FD.
  void inheritCUDATargetAttrs(FunctionDecl *FD, const FunctionTemplateDecl &TD);

  /// Returns the name of the launch configuration function.  This is the name
  /// of the function that will be called to configure kernel call, with the
  /// parameters specified via <<<>>>.
  std::string getCudaConfigureFuncName() const;

  /// \name Code completion
  //@{
  /// Describes the context in which code completion occurs.
  enum ParserCompletionContext {
    /// Code completion occurs at top-level or namespace context.
    PCC_Namespace,
    /// Code completion occurs within a class, struct, or union.
    PCC_Class,
    /// Code completion occurs within an Objective-C interface, protocol,
    /// or category.
    PCC_ObjCInterface,
    /// Code completion occurs within an Objective-C implementation or
    /// category implementation
    PCC_ObjCImplementation,
    /// Code completion occurs within the list of instance variables
    /// in an Objective-C interface, protocol, category, or implementation.
    PCC_ObjCInstanceVariableList,
    /// Code completion occurs following one or more template
    /// headers.
    PCC_Template,
    /// Code completion occurs following one or more template
    /// headers within a class.
    PCC_MemberTemplate,
    /// Code completion occurs within an expression.
    PCC_Expression,
    /// Code completion occurs within a statement, which may
    /// also be an expression or a declaration.
    PCC_Statement,
    /// Code completion occurs at the beginning of the
    /// initialization statement (or expression) in a for loop.
    PCC_ForInit,
    /// Code completion occurs within the condition of an if,
    /// while, switch, or for statement.
    PCC_Condition,
    /// Code completion occurs within the body of a function on a
    /// recovery path, where we do not have a specific handle on our position
    /// in the grammar.
    PCC_RecoveryInFunction,
    /// Code completion occurs where only a type is permitted.
    PCC_Type,
    /// Code completion occurs in a parenthesized expression, which
    /// might also be a type cast.
    PCC_ParenthesizedExpression,
    /// Code completion occurs within a sequence of declaration
    /// specifiers within a function, method, or block.
    PCC_LocalDeclarationSpecifiers
  };

  void CodeCompleteModuleImport(SourceLocation ImportLoc, ModuleIdPath Path);
  void CodeCompleteOrdinaryName(Scope *S,
                                ParserCompletionContext CompletionContext);
  void CodeCompleteDeclSpec(Scope *S, DeclSpec &DS,
                            bool AllowNonIdentifiers,
                            bool AllowNestedNameSpecifiers);

  struct CodeCompleteExpressionData;
  void CodeCompleteExpression(Scope *S,
                              const CodeCompleteExpressionData &Data);
  void CodeCompleteExpression(Scope *S, QualType PreferredType,
                              bool IsParenthesized = false);
  void CodeCompleteMemberReferenceExpr(Scope *S, Expr *Base, Expr *OtherOpBase,
                                       SourceLocation OpLoc, bool IsArrow,
                                       bool IsBaseExprStatement,
                                       QualType PreferredType);
  void CodeCompletePostfixExpression(Scope *S, ExprResult LHS,
                                     QualType PreferredType);
  void CodeCompleteTag(Scope *S, unsigned TagSpec);
  void CodeCompleteTypeQualifiers(DeclSpec &DS);
  void CodeCompleteFunctionQualifiers(DeclSpec &DS, Declarator &D,
                                      const VirtSpecifiers *VS = nullptr);
  void CodeCompleteBracketDeclarator(Scope *S);
  void CodeCompleteCase(Scope *S);
  enum class AttributeCompletion {
    Attribute,
    Scope,
    None,
  };
  void CodeCompleteAttribute(
      AttributeCommonInfo::Syntax Syntax,
      AttributeCompletion Completion = AttributeCompletion::Attribute,
      const IdentifierInfo *Scope = nullptr);
  /// Determines the preferred type of the current function argument, by
  /// examining the signatures of all possible overloads.
  /// Returns null if unknown or ambiguous, or if code completion is off.
  ///
  /// If the code completion point has been reached, also reports the function
  /// signatures that were considered.
  ///
  /// FIXME: rename to GuessCallArgumentType to reduce confusion.
  QualType ProduceCallSignatureHelp(Expr *Fn, ArrayRef<Expr *> Args,
                                    SourceLocation OpenParLoc);
  QualType ProduceConstructorSignatureHelp(QualType Type, SourceLocation Loc,
                                           ArrayRef<Expr *> Args,
                                           SourceLocation OpenParLoc,
                                           bool Braced);
  QualType ProduceCtorInitMemberSignatureHelp(
      Decl *ConstructorDecl, CXXScopeSpec SS, ParsedType TemplateTypeTy,
      ArrayRef<Expr *> ArgExprs, IdentifierInfo *II, SourceLocation OpenParLoc,
      bool Braced);
  QualType ProduceTemplateArgumentSignatureHelp(
      TemplateTy, ArrayRef<ParsedTemplateArgument>, SourceLocation LAngleLoc);
  void CodeCompleteInitializer(Scope *S, Decl *D);
  /// Trigger code completion for a record of \p BaseType. \p InitExprs are
  /// expressions in the initializer list seen so far and \p D is the current
  /// Designation being parsed.
  void CodeCompleteDesignator(const QualType BaseType,
                              llvm::ArrayRef<Expr *> InitExprs,
                              const Designation &D);
  void CodeCompleteAfterIf(Scope *S, bool IsBracedThen);

  void CodeCompleteQualifiedId(Scope *S, CXXScopeSpec &SS, bool EnteringContext,
                               bool IsUsingDeclaration, QualType BaseType,
                               QualType PreferredType);
  void CodeCompleteUsing(Scope *S);
  void CodeCompleteUsingDirective(Scope *S);
  void CodeCompleteNamespaceDecl(Scope *S);
  void CodeCompleteNamespaceAliasDecl(Scope *S);
  void CodeCompleteOperatorName(Scope *S);
  void CodeCompleteConstructorInitializer(
                                Decl *Constructor,
                                ArrayRef<CXXCtorInitializer *> Initializers);

  void CodeCompleteLambdaIntroducer(Scope *S, LambdaIntroducer &Intro,
                                    bool AfterAmpersand);
  void CodeCompleteAfterFunctionEquals(Declarator &D);

  void CodeCompleteObjCAtDirective(Scope *S);
  void CodeCompleteObjCAtVisibility(Scope *S);
  void CodeCompleteObjCAtStatement(Scope *S);
  void CodeCompleteObjCAtExpression(Scope *S);
  void CodeCompleteObjCPropertyFlags(Scope *S, ObjCDeclSpec &ODS);
  void CodeCompleteObjCPropertyGetter(Scope *S);
  void CodeCompleteObjCPropertySetter(Scope *S);
  void CodeCompleteObjCPassingType(Scope *S, ObjCDeclSpec &DS,
                                   bool IsParameter);
  void CodeCompleteObjCMessageReceiver(Scope *S);
  void CodeCompleteObjCSuperMessage(Scope *S, SourceLocation SuperLoc,
                                    ArrayRef<IdentifierInfo *> SelIdents,
                                    bool AtArgumentExpression);
  void CodeCompleteObjCClassMessage(Scope *S, ParsedType Receiver,
                                    ArrayRef<IdentifierInfo *> SelIdents,
                                    bool AtArgumentExpression,
                                    bool IsSuper = false);
  void CodeCompleteObjCInstanceMessage(Scope *S, Expr *Receiver,
                                       ArrayRef<IdentifierInfo *> SelIdents,
                                       bool AtArgumentExpression,
                                       ObjCInterfaceDecl *Super = nullptr);
  void CodeCompleteObjCForCollection(Scope *S,
                                     DeclGroupPtrTy IterationVar);
  void CodeCompleteObjCSelector(Scope *S,
                                ArrayRef<IdentifierInfo *> SelIdents);
  void CodeCompleteObjCProtocolReferences(
                                         ArrayRef<IdentifierLocPair> Protocols);
  void CodeCompleteObjCProtocolDecl(Scope *S);
  void CodeCompleteObjCInterfaceDecl(Scope *S);
  void CodeCompleteObjCSuperclass(Scope *S,
                                  IdentifierInfo *ClassName,
                                  SourceLocation ClassNameLoc);
  void CodeCompleteObjCImplementationDecl(Scope *S);
  void CodeCompleteObjCInterfaceCategory(Scope *S,
                                         IdentifierInfo *ClassName,
                                         SourceLocation ClassNameLoc);
  void CodeCompleteObjCImplementationCategory(Scope *S,
                                              IdentifierInfo *ClassName,
                                              SourceLocation ClassNameLoc);
  void CodeCompleteObjCPropertyDefinition(Scope *S);
  void CodeCompleteObjCPropertySynthesizeIvar(Scope *S,
                                              IdentifierInfo *PropertyName);
  void CodeCompleteObjCMethodDecl(Scope *S, Optional<bool> IsInstanceMethod,
                                  ParsedType ReturnType);
  void CodeCompleteObjCMethodDeclSelector(Scope *S,
                                          bool IsInstanceMethod,
                                          bool AtParameterName,
                                          ParsedType ReturnType,
                                          ArrayRef<IdentifierInfo *> SelIdents);
  void CodeCompleteObjCClassPropertyRefExpr(Scope *S, IdentifierInfo &ClassName,
                                            SourceLocation ClassNameLoc,
                                            bool IsBaseExprStatement);
  void CodeCompletePreprocessorDirective(bool InConditional);
  void CodeCompleteInPreprocessorConditionalExclusion(Scope *S);
  void CodeCompletePreprocessorMacroName(bool IsDefinition);
  void CodeCompletePreprocessorExpression();
  void CodeCompletePreprocessorMacroArgument(Scope *S,
                                             IdentifierInfo *Macro,
                                             MacroInfo *MacroInfo,
                                             unsigned Argument);
  void CodeCompleteIncludedFile(llvm::StringRef Dir, bool IsAngled);
  void CodeCompleteNaturalLanguage();
  void CodeCompleteAvailabilityPlatformName();
  void GatherGlobalCodeCompletions(CodeCompletionAllocator &Allocator,
                                   CodeCompletionTUInfo &CCTUInfo,
                  SmallVectorImpl<CodeCompletionResult> &Results);
  //@}

  //===--------------------------------------------------------------------===//
  // Extra semantic analysis beyond the C type system

public:
  SourceLocation getLocationOfStringLiteralByte(const StringLiteral *SL,
                                                unsigned ByteNo) const;

private:
  void CheckArrayAccess(const Expr *BaseExpr, const Expr *IndexExpr,
                        const ArraySubscriptExpr *ASE=nullptr,
                        bool AllowOnePastEnd=true, bool IndexNegated=false);
  void CheckArrayAccess(const Expr *E);
  // Used to grab the relevant information from a FormatAttr and a
  // FunctionDeclaration.
  struct FormatStringInfo {
    unsigned FormatIdx;
    unsigned FirstDataArg;
    bool HasVAListArg;
  };

  static bool getFormatStringInfo(const FormatAttr *Format, bool IsCXXMember,
                                  FormatStringInfo *FSI);
  bool CheckFunctionCall(FunctionDecl *FDecl, CallExpr *TheCall,
                         const FunctionProtoType *Proto);
  bool CheckObjCMethodCall(ObjCMethodDecl *Method, SourceLocation loc,
                           ArrayRef<const Expr *> Args);
  bool CheckPointerCall(NamedDecl *NDecl, CallExpr *TheCall,
                        const FunctionProtoType *Proto);
  bool CheckOtherCall(CallExpr *TheCall, const FunctionProtoType *Proto);
  void CheckConstructorCall(FunctionDecl *FDecl, QualType ThisType,
                            ArrayRef<const Expr *> Args,
                            const FunctionProtoType *Proto, SourceLocation Loc);

  void CheckArgAlignment(SourceLocation Loc, NamedDecl *FDecl,
                         StringRef ParamName, QualType ArgTy, QualType ParamTy);

  void checkCall(NamedDecl *FDecl, const FunctionProtoType *Proto,
                 const Expr *ThisArg, ArrayRef<const Expr *> Args,
                 bool IsMemberFunction, SourceLocation Loc, SourceRange Range,
                 VariadicCallType CallType);

  bool CheckObjCString(Expr *Arg);
  ExprResult CheckOSLogFormatStringArg(Expr *Arg);

  ExprResult CheckBuiltinFunctionCall(FunctionDecl *FDecl,
                                      unsigned BuiltinID, CallExpr *TheCall);

  bool CheckTSBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                  CallExpr *TheCall);

  void checkFortifiedBuiltinMemoryFunction(FunctionDecl *FD, CallExpr *TheCall);

  bool CheckARMBuiltinExclusiveCall(unsigned BuiltinID, CallExpr *TheCall,
                                    unsigned MaxWidth);
  bool CheckNeonBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                    CallExpr *TheCall);
  bool CheckMVEBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckSVEBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckCDEBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                   CallExpr *TheCall);
  bool CheckARMCoprocessorImmediate(const TargetInfo &TI, const Expr *CoprocArg,
                                    bool WantCDE);
  bool CheckARMBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                   CallExpr *TheCall);

  bool CheckAArch64BuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                       CallExpr *TheCall);
  bool CheckBPFBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckHexagonBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckHexagonBuiltinArgument(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckMipsBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                    CallExpr *TheCall);
  bool CheckMipsBuiltinCpu(const TargetInfo &TI, unsigned BuiltinID,
                           CallExpr *TheCall);
  bool CheckMipsBuiltinArgument(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckSystemZBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckX86BuiltinRoundingOrSAE(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckX86BuiltinGatherScatterScale(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckX86BuiltinTileArguments(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckX86BuiltinTileArgumentsRange(CallExpr *TheCall,
                                         ArrayRef<int> ArgNums);
  bool CheckX86BuiltinTileDuplicate(CallExpr *TheCall, ArrayRef<int> ArgNums);
  bool CheckX86BuiltinTileRangeAndDuplicate(CallExpr *TheCall,
                                            ArrayRef<int> ArgNums);
  bool CheckX86BuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                   CallExpr *TheCall);
  bool CheckPPCBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                   CallExpr *TheCall);
  bool CheckAMDGCNBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckRISCVLMUL(CallExpr *TheCall, unsigned ArgNum);
  bool CheckRISCVBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                     CallExpr *TheCall);

  bool SemaBuiltinVAStart(unsigned BuiltinID, CallExpr *TheCall);
  bool SemaBuiltinVAStartARMMicrosoft(CallExpr *Call);
  bool SemaBuiltinUnorderedCompare(CallExpr *TheCall);
  bool SemaBuiltinFPClassification(CallExpr *TheCall, unsigned NumArgs);
  bool SemaBuiltinComplex(CallExpr *TheCall);
  bool SemaBuiltinVSX(CallExpr *TheCall);
  bool SemaBuiltinOSLogFormat(CallExpr *TheCall);
  bool SemaValueIsRunOfOnes(CallExpr *TheCall, unsigned ArgNum);

public:
  // Used by C++ template instantiation.
  ExprResult SemaBuiltinShuffleVector(CallExpr *TheCall);
  ExprResult SemaConvertVectorExpr(Expr *E, TypeSourceInfo *TInfo,
                                   SourceLocation BuiltinLoc,
                                   SourceLocation RParenLoc);

private:
  bool SemaBuiltinPrefetch(CallExpr *TheCall);
  bool SemaBuiltinAllocaWithAlign(CallExpr *TheCall);
  bool SemaBuiltinArithmeticFence(CallExpr *TheCall);
  bool SemaBuiltinAssume(CallExpr *TheCall);
  bool SemaBuiltinAssumeAligned(CallExpr *TheCall);
  bool SemaBuiltinLongjmp(CallExpr *TheCall);
  bool SemaBuiltinSetjmp(CallExpr *TheCall);
  ExprResult SemaBuiltinAtomicOverloaded(ExprResult TheCallResult);
  ExprResult SemaBuiltinNontemporalOverloaded(ExprResult TheCallResult);
  ExprResult SemaAtomicOpsOverloaded(ExprResult TheCallResult,
                                     AtomicExpr::AtomicOp Op);
  ExprResult SemaBuiltinOperatorNewDeleteOverloaded(ExprResult TheCallResult,
                                                    bool IsDelete);
  bool SemaBuiltinConstantArg(CallExpr *TheCall, int ArgNum,
                              llvm::APSInt &Result);
  bool SemaBuiltinConstantArgRange(CallExpr *TheCall, int ArgNum, int Low,
                                   int High, bool RangeIsError = true);
  bool SemaBuiltinConstantArgMultiple(CallExpr *TheCall, int ArgNum,
                                      unsigned Multiple);
  bool SemaBuiltinConstantArgPower2(CallExpr *TheCall, int ArgNum);
  bool SemaBuiltinConstantArgShiftedByte(CallExpr *TheCall, int ArgNum,
                                         unsigned ArgBits);
  bool SemaBuiltinConstantArgShiftedByteOrXXFF(CallExpr *TheCall, int ArgNum,
                                               unsigned ArgBits);
  bool SemaBuiltinARMSpecialReg(unsigned BuiltinID, CallExpr *TheCall,
                                int ArgNum, unsigned ExpectedFieldNum,
                                bool AllowName);
  bool SemaBuiltinARMMemoryTaggingCall(unsigned BuiltinID, CallExpr *TheCall);
  bool SemaBuiltinPPCMMACall(CallExpr *TheCall, unsigned BuiltinID,
                             const char *TypeDesc);

  bool CheckPPCMMAType(QualType Type, SourceLocation TypeLoc);

  bool SemaBuiltinElementwiseMath(CallExpr *TheCall);
  bool PrepareBuiltinElementwiseMathOneArgCall(CallExpr *TheCall);
  bool PrepareBuiltinReduceMathOneArgCall(CallExpr *TheCall);

  // Matrix builtin handling.
  ExprResult SemaBuiltinMatrixTranspose(CallExpr *TheCall,
                                        ExprResult CallResult);
  ExprResult SemaBuiltinMatrixColumnMajorLoad(CallExpr *TheCall,
                                              ExprResult CallResult);
  ExprResult SemaBuiltinMatrixColumnMajorStore(CallExpr *TheCall,
                                               ExprResult CallResult);

public:
  enum FormatStringType {
    FST_Scanf,
    FST_Printf,
    FST_NSString,
    FST_Strftime,
    FST_Strfmon,
    FST_Kprintf,
    FST_FreeBSDKPrintf,
    FST_OSTrace,
    FST_OSLog,
    FST_Unknown
  };
  static FormatStringType GetFormatStringType(const FormatAttr *Format);

  bool FormatStringHasSArg(const StringLiteral *FExpr);

  static bool GetFormatNSStringIdx(const FormatAttr *Format, unsigned &Idx);

private:
  bool CheckFormatArguments(const FormatAttr *Format,
                            ArrayRef<const Expr *> Args,
                            bool IsCXXMember,
                            VariadicCallType CallType,
                            SourceLocation Loc, SourceRange Range,
                            llvm::SmallBitVector &CheckedVarArgs);
  bool CheckFormatArguments(ArrayRef<const Expr *> Args,
                            bool HasVAListArg, unsigned format_idx,
                            unsigned firstDataArg, FormatStringType Type,
                            VariadicCallType CallType,
                            SourceLocation Loc, SourceRange range,
                            llvm::SmallBitVector &CheckedVarArgs);

  void CheckAbsoluteValueFunction(const CallExpr *Call,
                                  const FunctionDecl *FDecl);

  void CheckMaxUnsignedZero(const CallExpr *Call, const FunctionDecl *FDecl);

  void CheckMemaccessArguments(const CallExpr *Call,
                               unsigned BId,
                               IdentifierInfo *FnName);

  void CheckStrlcpycatArguments(const CallExpr *Call,
                                IdentifierInfo *FnName);

  void CheckStrncatArguments(const CallExpr *Call,
                             IdentifierInfo *FnName);

  void CheckFreeArguments(const CallExpr *E);

  void CheckReturnValExpr(Expr *RetValExp, QualType lhsType,
                          SourceLocation ReturnLoc,
                          bool isObjCMethod = false,
                          const AttrVec *Attrs = nullptr,
                          const FunctionDecl *FD = nullptr);

public:
  void CheckFloatComparison(SourceLocation Loc, Expr *LHS, Expr *RHS);

private:
  void CheckImplicitConversions(Expr *E, SourceLocation CC = SourceLocation());
  void CheckBoolLikeConversion(Expr *E, SourceLocation CC);
  void CheckForIntOverflow(Expr *E);
  void CheckUnsequencedOperations(const Expr *E);

  /// Perform semantic checks on a completed expression. This will either
  /// be a full-expression or a default argument expression.
  void CheckCompletedExpr(Expr *E, SourceLocation CheckLoc = SourceLocation(),
                          bool IsConstexpr = false);

  void CheckBitFieldInitialization(SourceLocation InitLoc, FieldDecl *Field,
                                   Expr *Init);

  /// Check if there is a field shadowing.
  void CheckShadowInheritedFields(const SourceLocation &Loc,
                                  DeclarationName FieldName,
                                  const CXXRecordDecl *RD,
                                  bool DeclIsField = true);

  /// Check if the given expression contains 'break' or 'continue'
  /// statement that produces control flow different from GCC.
  void CheckBreakContinueBinding(Expr *E);

  /// Check whether receiver is mutable ObjC container which
  /// attempts to add itself into the container
  void CheckObjCCircularContainer(ObjCMessageExpr *Message);

  void CheckTCBEnforcement(const CallExpr *TheCall, const FunctionDecl *Callee);

  void AnalyzeDeleteExprMismatch(const CXXDeleteExpr *DE);
  void AnalyzeDeleteExprMismatch(FieldDecl *Field, SourceLocation DeleteLoc,
                                 bool DeleteWasArrayForm);
public:
  /// Register a magic integral constant to be used as a type tag.
  void RegisterTypeTagForDatatype(const IdentifierInfo *ArgumentKind,
                                  uint64_t MagicValue, QualType Type,
                                  bool LayoutCompatible, bool MustBeNull);

  struct TypeTagData {
    TypeTagData() {}

    TypeTagData(QualType Type, bool LayoutCompatible, bool MustBeNull) :
        Type(Type), LayoutCompatible(LayoutCompatible),
        MustBeNull(MustBeNull)
    {}

    QualType Type;

    /// If true, \c Type should be compared with other expression's types for
    /// layout-compatibility.
    unsigned LayoutCompatible : 1;
    unsigned MustBeNull : 1;
  };

  /// A pair of ArgumentKind identifier and magic value.  This uniquely
  /// identifies the magic value.
  typedef std::pair<const IdentifierInfo *, uint64_t> TypeTagMagicValue;

private:
  /// A map from magic value to type information.
  std::unique_ptr<llvm::DenseMap<TypeTagMagicValue, TypeTagData>>
      TypeTagForDatatypeMagicValues;

  /// Peform checks on a call of a function with argument_with_type_tag
  /// or pointer_with_type_tag attributes.
  void CheckArgumentWithTypeTag(const ArgumentWithTypeTagAttr *Attr,
                                const ArrayRef<const Expr *> ExprArgs,
                                SourceLocation CallSiteLoc);

  /// Check if we are taking the address of a packed field
  /// as this may be a problem if the pointer value is dereferenced.
  void CheckAddressOfPackedMember(Expr *rhs);

  /// The parser's current scope.
  ///
  /// The parser maintains this state here.
  Scope *CurScope;

  mutable IdentifierInfo *Ident_super;
  mutable IdentifierInfo *Ident___float128;

  /// Nullability type specifiers.
  IdentifierInfo *Ident__Nonnull = nullptr;
  IdentifierInfo *Ident__Nullable = nullptr;
  IdentifierInfo *Ident__Nullable_result = nullptr;
  IdentifierInfo *Ident__Null_unspecified = nullptr;

  IdentifierInfo *Ident_NSError = nullptr;

  /// The handler for the FileChanged preprocessor events.
  ///
  /// Used for diagnostics that implement custom semantic analysis for #include
  /// directives, like -Wpragma-pack.
  sema::SemaPPCallbacks *SemaPPCallbackHandler;

protected:
  friend class Parser;
  friend class InitializationSequence;
  friend class ASTReader;
  friend class ASTDeclReader;
  friend class ASTWriter;

public:
  /// Retrieve the keyword associated
  IdentifierInfo *getNullabilityKeyword(NullabilityKind nullability);

  /// The struct behind the CFErrorRef pointer.
  RecordDecl *CFError = nullptr;
  bool isCFError(RecordDecl *D);

  /// Retrieve the identifier "NSError".
  IdentifierInfo *getNSErrorIdent();

  /// Retrieve the parser's current scope.
  ///
  /// This routine must only be used when it is certain that semantic analysis
  /// and the parser are in precisely the same context, which is not the case
  /// when, e.g., we are performing any kind of template instantiation.
  /// Therefore, the only safe places to use this scope are in the parser
  /// itself and in routines directly invoked from the parser and *never* from
  /// template substitution or instantiation.
  Scope *getCurScope() const { return CurScope; }

  void incrementMSManglingNumber() const {
    return CurScope->incrementMSManglingNumber();
  }

  IdentifierInfo *getSuperIdentifier() const;
  IdentifierInfo *getFloat128Identifier() const;

  Decl *getObjCDeclContext() const;

  DeclContext *getCurLexicalContext() const {
    return OriginalLexicalContext ? OriginalLexicalContext : CurContext;
  }

  const DeclContext *getCurObjCLexicalContext() const {
    const DeclContext *DC = getCurLexicalContext();
    // A category implicitly has the attribute of the interface.
    if (const ObjCCategoryDecl *CatD = dyn_cast<ObjCCategoryDecl>(DC))
      DC = CatD->getClassInterface();
    return DC;
  }

  /// Determine the number of levels of enclosing template parameters. This is
  /// only usable while parsing. Note that this does not include dependent
  /// contexts in which no template parameters have yet been declared, such as
  /// in a terse function template or generic lambda before the first 'auto' is
  /// encountered.
  unsigned getTemplateDepth(Scope *S) const;

  /// To be used for checking whether the arguments being passed to
  /// function exceeds the number of parameters expected for it.
  static bool TooManyArguments(size_t NumParams, size_t NumArgs,
                               bool PartialOverloading = false) {
    // We check whether we're just after a comma in code-completion.
    if (NumArgs > 0 && PartialOverloading)
      return NumArgs + 1 > NumParams; // If so, we view as an extra argument.
    return NumArgs > NumParams;
  }

  // Emitting members of dllexported classes is delayed until the class
  // (including field initializers) is fully parsed.
  SmallVector<CXXRecordDecl*, 4> DelayedDllExportClasses;
  SmallVector<CXXMethodDecl*, 4> DelayedDllExportMemberFunctions;

private:
  int ParsingClassDepth = 0;

  class SavePendingParsedClassStateRAII {
  public:
    SavePendingParsedClassStateRAII(Sema &S) : S(S) { swapSavedState(); }

    ~SavePendingParsedClassStateRAII() {
      assert(S.DelayedOverridingExceptionSpecChecks.empty() &&
             "there shouldn't be any pending delayed exception spec checks");
      assert(S.DelayedEquivalentExceptionSpecChecks.empty() &&
             "there shouldn't be any pending delayed exception spec checks");
      swapSavedState();
    }

  private:
    Sema &S;
    decltype(DelayedOverridingExceptionSpecChecks)
        SavedOverridingExceptionSpecChecks;
    decltype(DelayedEquivalentExceptionSpecChecks)
        SavedEquivalentExceptionSpecChecks;

    void swapSavedState() {
      SavedOverridingExceptionSpecChecks.swap(
          S.DelayedOverridingExceptionSpecChecks);
      SavedEquivalentExceptionSpecChecks.swap(
          S.DelayedEquivalentExceptionSpecChecks);
    }
  };

  /// Helper class that collects misaligned member designations and
  /// their location info for delayed diagnostics.
  struct MisalignedMember {
    Expr *E;
    RecordDecl *RD;
    ValueDecl *MD;
    CharUnits Alignment;

    MisalignedMember() : E(), RD(), MD() {}
    MisalignedMember(Expr *E, RecordDecl *RD, ValueDecl *MD,
                     CharUnits Alignment)
        : E(E), RD(RD), MD(MD), Alignment(Alignment) {}
    explicit MisalignedMember(Expr *E)
        : MisalignedMember(E, nullptr, nullptr, CharUnits()) {}

    bool operator==(const MisalignedMember &m) { return this->E == m.E; }
  };
  /// Small set of gathered accesses to potentially misaligned members
  /// due to the packed attribute.
  SmallVector<MisalignedMember, 4> MisalignedMembers;

  /// Adds an expression to the set of gathered misaligned members.
  void AddPotentialMisalignedMembers(Expr *E, RecordDecl *RD, ValueDecl *MD,
                                     CharUnits Alignment);

public:
  /// Diagnoses the current set of gathered accesses. This typically
  /// happens at full expression level. The set is cleared after emitting the
  /// diagnostics.
  void DiagnoseMisalignedMembers();

  /// This function checks if the expression is in the sef of potentially
  /// misaligned members and it is converted to some pointer type T with lower
  /// or equal alignment requirements. If so it removes it. This is used when
  /// we do not want to diagnose such misaligned access (e.g. in conversions to
  /// void*).
  void DiscardMisalignedMemberAddress(const Type *T, Expr *E);

  /// This function calls Action when it determines that E designates a
  /// misaligned member due to the packed attribute. This is used to emit
  /// local diagnostics like in reference binding.
  void RefersToMemberWithReducedAlignment(
      Expr *E,
      llvm::function_ref<void(Expr *, RecordDecl *, FieldDecl *, CharUnits)>
          Action);

  /// Describes the reason a calling convention specification was ignored, used
  /// for diagnostics.
  enum class CallingConventionIgnoredReason {
    ForThisTarget = 0,
    VariadicFunction,
    ConstructorDestructor,
    BuiltinFunction
  };
  /// Creates a SemaDiagnosticBuilder that emits the diagnostic if the current
  /// context is "used as device code".
  ///
  /// - If CurLexicalContext is a kernel function or it is known that the
  ///   function will be emitted for the device, emits the diagnostics
  ///   immediately.
  /// - If CurLexicalContext is a function and we are compiling
  ///   for the device, but we don't know that this function will be codegen'ed
  ///   for devive yet, creates a diagnostic which is emitted if and when we
  ///   realize that the function will be codegen'ed.
  ///
  /// Example usage:
  ///
  /// Diagnose __float128 type usage only from SYCL device code if the current
  /// target doesn't support it
  /// if (!S.Context.getTargetInfo().hasFloat128Type() &&
  ///     S.getLangOpts().SYCLIsDevice)
  ///   SYCLDiagIfDeviceCode(Loc, diag::err_type_unsupported) << "__float128";
  SemaDiagnosticBuilder SYCLDiagIfDeviceCode(SourceLocation Loc,
                                             unsigned DiagID);

  /// Check whether we're allowed to call Callee from the current context.
  ///
  /// - If the call is never allowed in a semantically-correct program
  ///   emits an error and returns false.
  ///
  /// - If the call is allowed in semantically-correct programs, but only if
  ///   it's never codegen'ed, creates a deferred diagnostic to be emitted if
  ///   and when the caller is codegen'ed, and returns true.
  ///
  /// - Otherwise, returns true without emitting any diagnostics.
  ///
  /// Adds Callee to DeviceCallGraph if we don't know if its caller will be
  /// codegen'ed yet.
  bool checkSYCLDeviceFunction(SourceLocation Loc, FunctionDecl *Callee);
  void deepTypeCheckForSYCLDevice(SourceLocation UsedAt,
                                  llvm::DenseSet<QualType> Visited,
                                  ValueDecl *DeclToCheck);
};

/// RAII object that enters a new expression evaluation context.
class EnterExpressionEvaluationContext {
  Sema &Actions;
  bool Entered = true;

public:
  EnterExpressionEvaluationContext(
      Sema &Actions, Sema::ExpressionEvaluationContext NewContext,
      Decl *LambdaContextDecl = nullptr,
      Sema::ExpressionEvaluationContextRecord::ExpressionKind ExprContext =
          Sema::ExpressionEvaluationContextRecord::EK_Other,
      bool ShouldEnter = true)
      : Actions(Actions), Entered(ShouldEnter) {
    if (Entered)
      Actions.PushExpressionEvaluationContext(NewContext, LambdaContextDecl,
                                              ExprContext);
  }
  EnterExpressionEvaluationContext(
      Sema &Actions, Sema::ExpressionEvaluationContext NewContext,
      Sema::ReuseLambdaContextDecl_t,
      Sema::ExpressionEvaluationContextRecord::ExpressionKind ExprContext =
          Sema::ExpressionEvaluationContextRecord::EK_Other)
      : Actions(Actions) {
    Actions.PushExpressionEvaluationContext(
        NewContext, Sema::ReuseLambdaContextDecl, ExprContext);
  }

  enum InitListTag { InitList };
  EnterExpressionEvaluationContext(Sema &Actions, InitListTag,
                                   bool ShouldEnter = true)
      : Actions(Actions), Entered(false) {
    // In C++11 onwards, narrowing checks are performed on the contents of
    // braced-init-lists, even when they occur within unevaluated operands.
    // Therefore we still need to instantiate constexpr functions used in such
    // a context.
    if (ShouldEnter && Actions.isUnevaluatedContext() &&
        Actions.getLangOpts().CPlusPlus11) {
      Actions.PushExpressionEvaluationContext(
          Sema::ExpressionEvaluationContext::UnevaluatedList);
      Entered = true;
    }
  }

  ~EnterExpressionEvaluationContext() {
    if (Entered)
      Actions.PopExpressionEvaluationContext();
  }
};

DeductionFailureInfo
MakeDeductionFailureInfo(ASTContext &Context, Sema::TemplateDeductionResult TDK,
                         sema::TemplateDeductionInfo &Info);

/// Contains a late templated function.
/// Will be parsed at the end of the translation unit, used by Sema & Parser.
struct LateParsedTemplate {
  CachedTokens Toks;
  /// The template function declaration to be late parsed.
  Decl *D;
};

template <>
void Sema::PragmaStack<Sema::AlignPackInfo>::Act(SourceLocation PragmaLocation,
                                                 PragmaMsStackAction Action,
                                                 llvm::StringRef StackSlotLabel,
                                                 AlignPackInfo Value);

} // end namespace clang

namespace llvm {
// Hash a FunctionDeclAndLoc by looking at both its FunctionDecl and its
// SourceLocation.
template <> struct DenseMapInfo<clang::Sema::FunctionDeclAndLoc> {
  using FunctionDeclAndLoc = clang::Sema::FunctionDeclAndLoc;
  using FDBaseInfo = DenseMapInfo<clang::CanonicalDeclPtr<clang::FunctionDecl>>;

  static FunctionDeclAndLoc getEmptyKey() {
    return {FDBaseInfo::getEmptyKey(), clang::SourceLocation()};
  }

  static FunctionDeclAndLoc getTombstoneKey() {
    return {FDBaseInfo::getTombstoneKey(), clang::SourceLocation()};
  }

  static unsigned getHashValue(const FunctionDeclAndLoc &FDL) {
    return hash_combine(FDBaseInfo::getHashValue(FDL.FD),
                        FDL.Loc.getHashValue());
  }

  static bool isEqual(const FunctionDeclAndLoc &LHS,
                      const FunctionDeclAndLoc &RHS) {
    return LHS.FD == RHS.FD && LHS.Loc == RHS.Loc;
  }
};
} // namespace llvm

#endif
