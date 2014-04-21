//===-- LLVMContextImpl.cpp - Implement LLVMContextImpl -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the opaque LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include <algorithm>
using namespace llvm;

LLVMContextImpl::LLVMContextImpl(LLVMContext &C)
  : TheTrueVal(nullptr), TheFalseVal(nullptr),
    VoidTy(C, Type::VoidTyID),
    LabelTy(C, Type::LabelTyID),
    HalfTy(C, Type::HalfTyID),
    FloatTy(C, Type::FloatTyID),
    DoubleTy(C, Type::DoubleTyID),
    MetadataTy(C, Type::MetadataTyID),
    X86_FP80Ty(C, Type::X86_FP80TyID),
    FP128Ty(C, Type::FP128TyID),
    PPC_FP128Ty(C, Type::PPC_FP128TyID),
    X86_MMXTy(C, Type::X86_MMXTyID),
    Int1Ty(C, 1),
    Int8Ty(C, 8),
    Int16Ty(C, 16),
    Int32Ty(C, 32),
    Int64Ty(C, 64) {
  InlineAsmDiagHandler = nullptr;
  InlineAsmDiagContext = nullptr;
  DiagnosticHandler = nullptr;
  DiagnosticContext = nullptr;
  NamedStructTypesUniqueID = 0;
}

namespace {

/// \brief Regular expression corresponding to the value given in the
/// command line flag -pass-remarks. Passes whose name matches this
/// regexp will emit a diagnostic when calling
/// LLVMContext::emitOptimizationRemark.
static Regex *OptimizationRemarkPattern = nullptr;

/// \brief String to hold all the values passed via -pass-remarks. Every
/// instance of -pass-remarks on the command line will be concatenated
/// to this string. Values are stored inside braces and concatenated with
/// the '|' operator. This implements the expected semantics that multiple
/// -pass-remarks are additive.
static std::string OptimizationRemarkExpr;

struct PassRemarksOpt {
  void operator=(const std::string &Val) const {
    // Create a regexp object to match pass names for emitOptimizationRemark.
    if (!Val.empty()) {
      if (!OptimizationRemarkExpr.empty())
        OptimizationRemarkExpr += "|";
      OptimizationRemarkExpr += "(" + Val + ")";
      delete OptimizationRemarkPattern;
      OptimizationRemarkPattern = new Regex(OptimizationRemarkExpr);
      std::string RegexError;
      if (!OptimizationRemarkPattern->isValid(RegexError))
        report_fatal_error("Invalid regular expression '" + Val +
                               "' in -pass-remarks: " + RegexError,
                           false);
    }
  };
};

static PassRemarksOpt PassRemarksOptLoc;

// -pass-remarks
//    Command line flag to enable LLVMContext::emitOptimizationRemark()
//    and LLVMContext::emitOptimizationNote() calls.
static cl::opt<PassRemarksOpt, true, cl::parser<std::string>>
PassRemarks("pass-remarks", cl::value_desc("pattern"),
            cl::desc("Enable optimization remarks from passes whose name match "
                     "the given regular expression"),
            cl::Hidden, cl::location(PassRemarksOptLoc), cl::ValueRequired,
            cl::ZeroOrMore);
}

bool
LLVMContextImpl::optimizationRemarksEnabledFor(const char *PassName) const {
  return OptimizationRemarkPattern &&
         OptimizationRemarkPattern->match(PassName);
}


namespace {
struct DropReferences {
  // Takes the value_type of a ConstantUniqueMap's internal map, whose 'second'
  // is a Constant*.
  template<typename PairT>
  void operator()(const PairT &P) {
    P.second->dropAllReferences();
  }
};

// Temporary - drops pair.first instead of second.
struct DropFirst {
  // Takes the value_type of a ConstantUniqueMap's internal map, whose 'second'
  // is a Constant*.
  template<typename PairT>
  void operator()(const PairT &P) {
    P.first->dropAllReferences();
  }
};
}

LLVMContextImpl::~LLVMContextImpl() {
  // NOTE: We need to delete the contents of OwnedModules, but Module's dtor
  // will call LLVMContextImpl::removeModule, thus invalidating iterators into
  // the container. Avoid iterators during this operation:
  while (!OwnedModules.empty())
    delete *OwnedModules.begin();
  
  // Free the constants.  This is important to do here to ensure that they are
  // freed before the LeakDetector is torn down.
  std::for_each(ExprConstants.map_begin(), ExprConstants.map_end(),
                DropReferences());
  std::for_each(ArrayConstants.map_begin(), ArrayConstants.map_end(),
                DropFirst());
  std::for_each(StructConstants.map_begin(), StructConstants.map_end(),
                DropFirst());
  std::for_each(VectorConstants.map_begin(), VectorConstants.map_end(),
                DropFirst());
  ExprConstants.freeConstants();
  ArrayConstants.freeConstants();
  StructConstants.freeConstants();
  VectorConstants.freeConstants();
  DeleteContainerSeconds(CAZConstants);
  DeleteContainerSeconds(CPNConstants);
  DeleteContainerSeconds(UVConstants);
  InlineAsms.freeConstants();
  DeleteContainerSeconds(IntConstants);
  DeleteContainerSeconds(FPConstants);
  
  for (StringMap<ConstantDataSequential*>::iterator I = CDSConstants.begin(),
       E = CDSConstants.end(); I != E; ++I)
    delete I->second;
  CDSConstants.clear();

  // Destroy attributes.
  for (FoldingSetIterator<AttributeImpl> I = AttrsSet.begin(),
         E = AttrsSet.end(); I != E; ) {
    FoldingSetIterator<AttributeImpl> Elem = I++;
    delete &*Elem;
  }

  // Destroy attribute lists.
  for (FoldingSetIterator<AttributeSetImpl> I = AttrsLists.begin(),
         E = AttrsLists.end(); I != E; ) {
    FoldingSetIterator<AttributeSetImpl> Elem = I++;
    delete &*Elem;
  }

  // Destroy attribute node lists.
  for (FoldingSetIterator<AttributeSetNode> I = AttrsSetNodes.begin(),
         E = AttrsSetNodes.end(); I != E; ) {
    FoldingSetIterator<AttributeSetNode> Elem = I++;
    delete &*Elem;
  }

  // Destroy MDNodes.  ~MDNode can move and remove nodes between the MDNodeSet
  // and the NonUniquedMDNodes sets, so copy the values out first.
  SmallVector<MDNode*, 8> MDNodes;
  MDNodes.reserve(MDNodeSet.size() + NonUniquedMDNodes.size());
  for (FoldingSetIterator<MDNode> I = MDNodeSet.begin(), E = MDNodeSet.end();
       I != E; ++I)
    MDNodes.push_back(&*I);
  MDNodes.append(NonUniquedMDNodes.begin(), NonUniquedMDNodes.end());
  for (SmallVectorImpl<MDNode *>::iterator I = MDNodes.begin(),
         E = MDNodes.end(); I != E; ++I)
    (*I)->destroy();
  assert(MDNodeSet.empty() && NonUniquedMDNodes.empty() &&
         "Destroying all MDNodes didn't empty the Context's sets.");

  // Destroy MDStrings.
  DeleteContainerSeconds(MDStringCache);
}

// ConstantsContext anchors
void UnaryConstantExpr::anchor() { }

void BinaryConstantExpr::anchor() { }

void SelectConstantExpr::anchor() { }

void ExtractElementConstantExpr::anchor() { }

void InsertElementConstantExpr::anchor() { }

void ShuffleVectorConstantExpr::anchor() { }

void ExtractValueConstantExpr::anchor() { }

void InsertValueConstantExpr::anchor() { }

void GetElementPtrConstantExpr::anchor() { }

void CompareConstantExpr::anchor() { }
