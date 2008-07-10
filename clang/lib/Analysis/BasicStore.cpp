//== BasicStore.cpp - Basic map from Locations to Values --------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the BasicStore and BasicStoreManager classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/BasicStore.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

namespace {
  
class VISIBILITY_HIDDEN BasicStoreManager : public StoreManager {
  typedef llvm::ImmutableMap<VarDecl*,RVal> VarBindingsTy;  
  VarBindingsTy::Factory VBFactory;
  
public:
  BasicStoreManager(llvm::BumpPtrAllocator& A) : VBFactory(A) {}
  virtual ~BasicStoreManager() {}

  virtual RVal GetRVal(Store St, LVal LV, QualType T);  
  virtual Store SetRVal(Store St, LVal LV, RVal V);  
  virtual Store Remove(Store St, LVal LV);

  virtual Store getInitialStore() {
    return VBFactory.GetEmptyMap().getRoot();
  }
};  
  
} // end anonymous namespace


StoreManager* clang::CreateBasicStoreManager(llvm::BumpPtrAllocator& A) {
  return new BasicStoreManager(A);
}

RVal BasicStoreManager::GetRVal(Store St, LVal LV, QualType T) {
  
  if (isa<UnknownVal>(LV))
    return UnknownVal();
  
  assert (!isa<UndefinedVal>(LV));
  
  switch (LV.getSubKind()) {

    case lval::DeclValKind: {      
      VarBindingsTy B(static_cast<const VarBindingsTy::TreeTy*>(St));      
      VarBindingsTy::data_type* T = B.lookup(cast<lval::DeclVal>(LV).getDecl());      
      return T ? *T : UnknownVal();
    }
      
    case lval::SymbolValKind: {
      
      // FIXME: This is a broken representation of memory, and is prone
      //  to crashing the analyzer when addresses to symbolic values are
      //  passed through casts.  We need a better representation of symbolic
      //  memory (or just memory in general); probably we should do this
      //  as a plugin class (similar to GRTransferFuncs).
      
#if 0      
      const lval::SymbolVal& SV = cast<lval::SymbolVal>(LV);
      assert (T.getTypePtr());
      
      // Punt on "symbolic" function pointers.
      if (T->isFunctionType())
        return UnknownVal();      
      
      if (T->isPointerType())
        return lval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
      else
        return nonlval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
#endif
      
      return UnknownVal();
    }
      
    case lval::ConcreteIntKind:
      // Some clients may call GetRVal with such an option simply because
      // they are doing a quick scan through their LVals (potentially to
      // invalidate their bindings).  Just return Undefined.
      return UndefinedVal();
      
    case lval::ArrayOffsetKind:
    case lval::FieldOffsetKind:
      return UnknownVal();
      
    case lval::FuncValKind:
      return LV;
      
    case lval::StringLiteralValKind:
      // FIXME: Implement better support for fetching characters from strings.
      return UnknownVal();
      
    default:
      assert (false && "Invalid LVal.");
      break;
  }
  
  return UnknownVal();
}

Store BasicStoreManager::SetRVal(Store St, LVal LV, RVal V) {    
  
  VarBindingsTy B(static_cast<const VarBindingsTy::TreeTy*>(St));
  
  switch (LV.getSubKind()) {
      
    case lval::DeclValKind:        
      return V.isUnknown()
        ? VBFactory.Remove(B,cast<lval::DeclVal>(LV).getDecl()).getRoot()
        : VBFactory.Add(B, cast<lval::DeclVal>(LV).getDecl(), V).getRoot();
      
    default:
      assert ("SetRVal for given LVal type not yet implemented.");
      return St;
  }
}

Store BasicStoreManager::Remove(Store St, LVal LV) {
  
  VarBindingsTy B(static_cast<const VarBindingsTy::TreeTy*>(St));
  
  switch (LV.getSubKind()) {
      
    case lval::DeclValKind:
      return VBFactory.Remove(B,cast<lval::DeclVal>(LV).getDecl()).getRoot();

    default:
      assert ("Remove for given LVal type not yet implemented.");
      return St;
  }
}
