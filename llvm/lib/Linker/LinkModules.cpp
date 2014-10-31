//===- lib/Linker/LinkModules.cpp - Module Linker Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM module linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker/Linker.h"
#include "llvm-c/Linker.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cctype>
#include <tuple>
using namespace llvm;


//===----------------------------------------------------------------------===//
// TypeMap implementation.
//===----------------------------------------------------------------------===//

namespace {
typedef SmallPtrSet<StructType *, 32> TypeSet;

class TypeMapTy : public ValueMapTypeRemapper {
  /// This is a mapping from a source type to a destination type to use.
  DenseMap<Type*, Type*> MappedTypes;

  /// When checking to see if two subgraphs are isomorphic, we speculatively
  /// add types to MappedTypes, but keep track of them here in case we need to
  /// roll back.
  SmallVector<Type*, 16> SpeculativeTypes;

  /// This is a list of non-opaque structs in the source module that are mapped
  /// to an opaque struct in the destination module.
  SmallVector<StructType*, 16> SrcDefinitionsToResolve;

  /// This is the set of opaque types in the destination modules who are
  /// getting a body from the source module.
  SmallPtrSet<StructType*, 16> DstResolvedOpaqueTypes;

public:
  TypeMapTy(TypeSet &Set) : DstStructTypesSet(Set) {}

  TypeSet &DstStructTypesSet;
  /// Indicate that the specified type in the destination module is conceptually
  /// equivalent to the specified type in the source module.
  void addTypeMapping(Type *DstTy, Type *SrcTy);

  /// linkDefinedTypeBodies - Produce a body for an opaque type in the dest
  /// module from a type definition in the source module.
  void linkDefinedTypeBodies();

  /// Return the mapped type to use for the specified input type from the
  /// source module.
  Type *get(Type *SrcTy);

  FunctionType *get(FunctionType *T) {return cast<FunctionType>(get((Type*)T));}

  /// Dump out the type map for debugging purposes.
  void dump() const {
    for (DenseMap<Type*, Type*>::const_iterator
           I = MappedTypes.begin(), E = MappedTypes.end(); I != E; ++I) {
      dbgs() << "TypeMap: ";
      I->first->print(dbgs());
      dbgs() << " => ";
      I->second->print(dbgs());
      dbgs() << '\n';
    }
  }

private:
  Type *getImpl(Type *T);
  /// Implement the ValueMapTypeRemapper interface.
  Type *remapType(Type *SrcTy) override {
    return get(SrcTy);
  }

  bool areTypesIsomorphic(Type *DstTy, Type *SrcTy);
};
}

void TypeMapTy::addTypeMapping(Type *DstTy, Type *SrcTy) {
  Type *&Entry = MappedTypes[SrcTy];
  if (Entry) return;

  if (DstTy == SrcTy) {
    Entry = DstTy;
    return;
  }

  // Check to see if these types are recursively isomorphic and establish a
  // mapping between them if so.
  if (!areTypesIsomorphic(DstTy, SrcTy)) {
    // Oops, they aren't isomorphic.  Just discard this request by rolling out
    // any speculative mappings we've established.
    for (unsigned i = 0, e = SpeculativeTypes.size(); i != e; ++i)
      MappedTypes.erase(SpeculativeTypes[i]);
  }
  SpeculativeTypes.clear();
}

/// Recursively walk this pair of types, returning true if they are isomorphic,
/// false if they are not.
bool TypeMapTy::areTypesIsomorphic(Type *DstTy, Type *SrcTy) {
  // Two types with differing kinds are clearly not isomorphic.
  if (DstTy->getTypeID() != SrcTy->getTypeID()) return false;

  // If we have an entry in the MappedTypes table, then we have our answer.
  Type *&Entry = MappedTypes[SrcTy];
  if (Entry)
    return Entry == DstTy;

  // Two identical types are clearly isomorphic.  Remember this
  // non-speculatively.
  if (DstTy == SrcTy) {
    Entry = DstTy;
    return true;
  }

  // Okay, we have two types with identical kinds that we haven't seen before.

  // If this is an opaque struct type, special case it.
  if (StructType *SSTy = dyn_cast<StructType>(SrcTy)) {
    // Mapping an opaque type to any struct, just keep the dest struct.
    if (SSTy->isOpaque()) {
      Entry = DstTy;
      SpeculativeTypes.push_back(SrcTy);
      return true;
    }

    // Mapping a non-opaque source type to an opaque dest.  If this is the first
    // type that we're mapping onto this destination type then we succeed.  Keep
    // the dest, but fill it in later.  This doesn't need to be speculative.  If
    // this is the second (different) type that we're trying to map onto the
    // same opaque type then we fail.
    if (cast<StructType>(DstTy)->isOpaque()) {
      // We can only map one source type onto the opaque destination type.
      if (!DstResolvedOpaqueTypes.insert(cast<StructType>(DstTy)))
        return false;
      SrcDefinitionsToResolve.push_back(SSTy);
      Entry = DstTy;
      return true;
    }
  }

  // If the number of subtypes disagree between the two types, then we fail.
  if (SrcTy->getNumContainedTypes() != DstTy->getNumContainedTypes())
    return false;

  // Fail if any of the extra properties (e.g. array size) of the type disagree.
  if (isa<IntegerType>(DstTy))
    return false;  // bitwidth disagrees.
  if (PointerType *PT = dyn_cast<PointerType>(DstTy)) {
    if (PT->getAddressSpace() != cast<PointerType>(SrcTy)->getAddressSpace())
      return false;

  } else if (FunctionType *FT = dyn_cast<FunctionType>(DstTy)) {
    if (FT->isVarArg() != cast<FunctionType>(SrcTy)->isVarArg())
      return false;
  } else if (StructType *DSTy = dyn_cast<StructType>(DstTy)) {
    StructType *SSTy = cast<StructType>(SrcTy);
    if (DSTy->isLiteral() != SSTy->isLiteral() ||
        DSTy->isPacked() != SSTy->isPacked())
      return false;
  } else if (ArrayType *DATy = dyn_cast<ArrayType>(DstTy)) {
    if (DATy->getNumElements() != cast<ArrayType>(SrcTy)->getNumElements())
      return false;
  } else if (VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
    if (DVTy->getNumElements() != cast<VectorType>(SrcTy)->getNumElements())
      return false;
  }

  // Otherwise, we speculate that these two types will line up and recursively
  // check the subelements.
  Entry = DstTy;
  SpeculativeTypes.push_back(SrcTy);

  for (unsigned i = 0, e = SrcTy->getNumContainedTypes(); i != e; ++i)
    if (!areTypesIsomorphic(DstTy->getContainedType(i),
                            SrcTy->getContainedType(i)))
      return false;

  // If everything seems to have lined up, then everything is great.
  return true;
}

/// Produce a body for an opaque type in the dest module from a type definition
/// in the source module.
void TypeMapTy::linkDefinedTypeBodies() {
  SmallVector<Type*, 16> Elements;
  SmallString<16> TmpName;

  // Note that processing entries in this loop (calling 'get') can add new
  // entries to the SrcDefinitionsToResolve vector.
  while (!SrcDefinitionsToResolve.empty()) {
    StructType *SrcSTy = SrcDefinitionsToResolve.pop_back_val();
    StructType *DstSTy = cast<StructType>(MappedTypes[SrcSTy]);

    // TypeMap is a many-to-one mapping, if there were multiple types that
    // provide a body for DstSTy then previous iterations of this loop may have
    // already handled it.  Just ignore this case.
    if (!DstSTy->isOpaque()) continue;
    assert(!SrcSTy->isOpaque() && "Not resolving a definition?");

    // Map the body of the source type over to a new body for the dest type.
    Elements.resize(SrcSTy->getNumElements());
    for (unsigned i = 0, e = Elements.size(); i != e; ++i)
      Elements[i] = getImpl(SrcSTy->getElementType(i));

    DstSTy->setBody(Elements, SrcSTy->isPacked());

    // If DstSTy has no name or has a longer name than STy, then viciously steal
    // STy's name.
    if (!SrcSTy->hasName()) continue;
    StringRef SrcName = SrcSTy->getName();

    if (!DstSTy->hasName() || DstSTy->getName().size() > SrcName.size()) {
      TmpName.insert(TmpName.end(), SrcName.begin(), SrcName.end());
      SrcSTy->setName("");
      DstSTy->setName(TmpName.str());
      TmpName.clear();
    }
  }

  DstResolvedOpaqueTypes.clear();
}

Type *TypeMapTy::get(Type *Ty) {
  Type *Result = getImpl(Ty);

  // If this caused a reference to any struct type, resolve it before returning.
  if (!SrcDefinitionsToResolve.empty())
    linkDefinedTypeBodies();
  return Result;
}

/// This is the recursive version of get().
Type *TypeMapTy::getImpl(Type *Ty) {
  // If we already have an entry for this type, return it.
  Type **Entry = &MappedTypes[Ty];
  if (*Entry) return *Entry;

  // If this is not a named struct type, then just map all of the elements and
  // then rebuild the type from inside out.
  if (!isa<StructType>(Ty) || cast<StructType>(Ty)->isLiteral()) {
    // If there are no element types to map, then the type is itself.  This is
    // true for the anonymous {} struct, things like 'float', integers, etc.
    if (Ty->getNumContainedTypes() == 0)
      return *Entry = Ty;

    // Remap all of the elements, keeping track of whether any of them change.
    bool AnyChange = false;
    SmallVector<Type*, 4> ElementTypes;
    ElementTypes.resize(Ty->getNumContainedTypes());
    for (unsigned i = 0, e = Ty->getNumContainedTypes(); i != e; ++i) {
      ElementTypes[i] = getImpl(Ty->getContainedType(i));
      AnyChange |= ElementTypes[i] != Ty->getContainedType(i);
    }

    // If we found our type while recursively processing stuff, just use it.
    Entry = &MappedTypes[Ty];
    if (*Entry) return *Entry;

    // If all of the element types mapped directly over, then the type is usable
    // as-is.
    if (!AnyChange)
      return *Entry = Ty;

    // Otherwise, rebuild a modified type.
    switch (Ty->getTypeID()) {
    default: llvm_unreachable("unknown derived type to remap");
    case Type::ArrayTyID:
      return *Entry = ArrayType::get(ElementTypes[0],
                                     cast<ArrayType>(Ty)->getNumElements());
    case Type::VectorTyID:
      return *Entry = VectorType::get(ElementTypes[0],
                                      cast<VectorType>(Ty)->getNumElements());
    case Type::PointerTyID:
      return *Entry = PointerType::get(ElementTypes[0],
                                      cast<PointerType>(Ty)->getAddressSpace());
    case Type::FunctionTyID:
      return *Entry = FunctionType::get(ElementTypes[0],
                                        makeArrayRef(ElementTypes).slice(1),
                                        cast<FunctionType>(Ty)->isVarArg());
    case Type::StructTyID:
      // Note that this is only reached for anonymous structs.
      return *Entry = StructType::get(Ty->getContext(), ElementTypes,
                                      cast<StructType>(Ty)->isPacked());
    }
  }

  // Otherwise, this is an unmapped named struct.  If the struct can be directly
  // mapped over, just use it as-is.  This happens in a case when the linked-in
  // module has something like:
  //   %T = type {%T*, i32}
  //   @GV = global %T* null
  // where T does not exist at all in the destination module.
  //
  // The other case we watch for is when the type is not in the destination
  // module, but that it has to be rebuilt because it refers to something that
  // is already mapped.  For example, if the destination module has:
  //  %A = type { i32 }
  // and the source module has something like
  //  %A' = type { i32 }
  //  %B = type { %A'* }
  //  @GV = global %B* null
  // then we want to create a new type: "%B = type { %A*}" and have it take the
  // pristine "%B" name from the source module.
  //
  // To determine which case this is, we have to recursively walk the type graph
  // speculating that we'll be able to reuse it unmodified.  Only if this is
  // safe would we map the entire thing over.  Because this is an optimization,
  // and is not required for the prettiness of the linked module, we just skip
  // it and always rebuild a type here.
  StructType *STy = cast<StructType>(Ty);

  // If the type is opaque, we can just use it directly.
  if (STy->isOpaque()) {
    // A named structure type from src module is used. Add it to the Set of
    // identified structs in the destination module.
    DstStructTypesSet.insert(STy);
    return *Entry = STy;
  }

  // Otherwise we create a new type and resolve its body later.  This will be
  // resolved by the top level of get().
  SrcDefinitionsToResolve.push_back(STy);
  StructType *DTy = StructType::create(STy->getContext());
  // A new identified structure type was created. Add it to the set of
  // identified structs in the destination module.
  DstStructTypesSet.insert(DTy);
  DstResolvedOpaqueTypes.insert(DTy);
  return *Entry = DTy;
}

//===----------------------------------------------------------------------===//
// ModuleLinker implementation.
//===----------------------------------------------------------------------===//

namespace {
  class ModuleLinker;

  /// Creates prototypes for functions that are lazily linked on the fly. This
  /// speeds up linking for modules with many/ lazily linked functions of which
  /// few get used.
  class ValueMaterializerTy : public ValueMaterializer {
    TypeMapTy &TypeMap;
    Module *DstM;
    std::vector<Function*> &LazilyLinkFunctions;
  public:
    ValueMaterializerTy(TypeMapTy &TypeMap, Module *DstM,
                        std::vector<Function*> &LazilyLinkFunctions) :
      ValueMaterializer(), TypeMap(TypeMap), DstM(DstM),
      LazilyLinkFunctions(LazilyLinkFunctions) {
    }

    Value *materializeValueFor(Value *V) override;
  };

  namespace {
  class LinkDiagnosticInfo : public DiagnosticInfo {
    const Twine &Msg;

  public:
    LinkDiagnosticInfo(DiagnosticSeverity Severity, const Twine &Msg);
    void print(DiagnosticPrinter &DP) const override;
  };
  LinkDiagnosticInfo::LinkDiagnosticInfo(DiagnosticSeverity Severity,
                                         const Twine &Msg)
      : DiagnosticInfo(DK_Linker, Severity), Msg(Msg) {}
  void LinkDiagnosticInfo::print(DiagnosticPrinter &DP) const { DP << Msg; }
  }

  /// This is an implementation class for the LinkModules function, which is the
  /// entrypoint for this file.
  class ModuleLinker {
    Module *DstM, *SrcM;

    TypeMapTy TypeMap;
    ValueMaterializerTy ValMaterializer;

    /// Mapping of values from what they used to be in Src, to what they are now
    /// in DstM.  ValueToValueMapTy is a ValueMap, which involves some overhead
    /// due to the use of Value handles which the Linker doesn't actually need,
    /// but this allows us to reuse the ValueMapper code.
    ValueToValueMapTy ValueMap;

    struct AppendingVarInfo {
      GlobalVariable *NewGV;   // New aggregate global in dest module.
      const Constant *DstInit; // Old initializer from dest module.
      const Constant *SrcInit; // Old initializer from src module.
    };

    std::vector<AppendingVarInfo> AppendingVars;

    // Set of items not to link in from source.
    SmallPtrSet<const Value*, 16> DoNotLinkFromSource;

    // Vector of functions to lazily link in.
    std::vector<Function*> LazilyLinkFunctions;

    Linker::DiagnosticHandlerFunction DiagnosticHandler;

  public:
    ModuleLinker(Module *dstM, TypeSet &Set, Module *srcM,
                 Linker::DiagnosticHandlerFunction DiagnosticHandler)
        : DstM(dstM), SrcM(srcM), TypeMap(Set),
          ValMaterializer(TypeMap, DstM, LazilyLinkFunctions),
          DiagnosticHandler(DiagnosticHandler) {}

    bool run();

  private:
    bool shouldLinkFromSource(bool &LinkFromSrc, const GlobalValue &Dest,
                              const GlobalValue &Src);

    /// Helper method for setting a message and returning an error code.
    bool emitError(const Twine &Message) {
      DiagnosticHandler(LinkDiagnosticInfo(DS_Error, Message));
      return true;
    }

    void emitWarning(const Twine &Message) {
      DiagnosticHandler(LinkDiagnosticInfo(DS_Warning, Message));
    }

    bool getComdatLeader(Module *M, StringRef ComdatName,
                         const GlobalVariable *&GVar);
    bool computeResultingSelectionKind(StringRef ComdatName,
                                       Comdat::SelectionKind Src,
                                       Comdat::SelectionKind Dst,
                                       Comdat::SelectionKind &Result,
                                       bool &LinkFromSrc);
    std::map<const Comdat *, std::pair<Comdat::SelectionKind, bool>>
        ComdatsChosen;
    bool getComdatResult(const Comdat *SrcC, Comdat::SelectionKind &SK,
                         bool &LinkFromSrc);

    /// This analyzes the two global values and determines what the result will
    /// look like in the destination module.
    bool getLinkageResult(GlobalValue *Dest, const GlobalValue *Src,
                          GlobalValue::LinkageTypes &LT,
                          GlobalValue::VisibilityTypes &Vis,
                          bool &LinkFromSrc);

    /// Given a global in the source module, return the global in the
    /// destination module that is being linked to, if any.
    GlobalValue *getLinkedToGlobal(const GlobalValue *SrcGV) {
      // If the source has no name it can't link.  If it has local linkage,
      // there is no name match-up going on.
      if (!SrcGV->hasName() || SrcGV->hasLocalLinkage())
        return nullptr;

      // Otherwise see if we have a match in the destination module's symtab.
      GlobalValue *DGV = DstM->getNamedValue(SrcGV->getName());
      if (!DGV) return nullptr;

      // If we found a global with the same name in the dest module, but it has
      // internal linkage, we are really not doing any linkage here.
      if (DGV->hasLocalLinkage())
        return nullptr;

      // Otherwise, we do in fact link to the destination global.
      return DGV;
    }

    void computeTypeMapping();

    void upgradeMismatchedGlobalArray(StringRef Name);
    void upgradeMismatchedGlobals();

    bool linkAppendingVarProto(GlobalVariable *DstGV,
                               const GlobalVariable *SrcGV);
    bool linkGlobalProto(const GlobalVariable *SrcGV);
    bool linkFunctionProto(Function *SrcF);
    bool linkAliasProto(GlobalAlias *SrcA);
    bool linkModuleFlagsMetadata();

    void linkAppendingVarInit(const AppendingVarInfo &AVI);
    void linkGlobalInits();
    void linkFunctionBody(Function *Dst, Function *Src);
    void linkAliasBodies();
    void linkNamedMDNodes();
  };
}

/// The LLVM SymbolTable class autorenames globals that conflict in the symbol
/// table. This is good for all clients except for us. Go through the trouble
/// to force this back.
static void forceRenaming(GlobalValue *GV, StringRef Name) {
  // If the global doesn't force its name or if it already has the right name,
  // there is nothing for us to do.
  if (GV->hasLocalLinkage() || GV->getName() == Name)
    return;

  Module *M = GV->getParent();

  // If there is a conflict, rename the conflict.
  if (GlobalValue *ConflictGV = M->getNamedValue(Name)) {
    GV->takeName(ConflictGV);
    ConflictGV->setName(Name);    // This will cause ConflictGV to get renamed
    assert(ConflictGV->getName() != Name && "forceRenaming didn't work");
  } else {
    GV->setName(Name);              // Force the name back
  }
}

/// copy additional attributes (those not needed to construct a GlobalValue)
/// from the SrcGV to the DestGV.
static void copyGVAttributes(GlobalValue *DestGV, const GlobalValue *SrcGV) {
  // Use the maximum alignment, rather than just copying the alignment of SrcGV.
  auto *DestGO = dyn_cast<GlobalObject>(DestGV);
  unsigned Alignment;
  if (DestGO)
    Alignment = std::max(DestGO->getAlignment(), SrcGV->getAlignment());

  DestGV->copyAttributesFrom(SrcGV);

  if (DestGO)
    DestGO->setAlignment(Alignment);

  forceRenaming(DestGV, SrcGV->getName());
}

static bool isLessConstraining(GlobalValue::VisibilityTypes a,
                               GlobalValue::VisibilityTypes b) {
  if (a == GlobalValue::HiddenVisibility)
    return false;
  if (b == GlobalValue::HiddenVisibility)
    return true;
  if (a == GlobalValue::ProtectedVisibility)
    return false;
  if (b == GlobalValue::ProtectedVisibility)
    return true;
  return false;
}

Value *ValueMaterializerTy::materializeValueFor(Value *V) {
  Function *SF = dyn_cast<Function>(V);
  if (!SF)
    return nullptr;

  Function *DF = Function::Create(TypeMap.get(SF->getFunctionType()),
                                  SF->getLinkage(), SF->getName(), DstM);
  copyGVAttributes(DF, SF);

  if (Comdat *SC = SF->getComdat()) {
    Comdat *DC = DstM->getOrInsertComdat(SC->getName());
    DF->setComdat(DC);
  }

  LazilyLinkFunctions.push_back(SF);
  return DF;
}

bool ModuleLinker::getComdatLeader(Module *M, StringRef ComdatName,
                                   const GlobalVariable *&GVar) {
  const GlobalValue *GVal = M->getNamedValue(ComdatName);
  if (const auto *GA = dyn_cast_or_null<GlobalAlias>(GVal)) {
    GVal = GA->getBaseObject();
    if (!GVal)
      // We cannot resolve the size of the aliasee yet.
      return emitError("Linking COMDATs named '" + ComdatName +
                       "': COMDAT key involves incomputable alias size.");
  }

  GVar = dyn_cast_or_null<GlobalVariable>(GVal);
  if (!GVar)
    return emitError(
        "Linking COMDATs named '" + ComdatName +
        "': GlobalVariable required for data dependent selection!");

  return false;
}

bool ModuleLinker::computeResultingSelectionKind(StringRef ComdatName,
                                                 Comdat::SelectionKind Src,
                                                 Comdat::SelectionKind Dst,
                                                 Comdat::SelectionKind &Result,
                                                 bool &LinkFromSrc) {
  // The ability to mix Comdat::SelectionKind::Any with
  // Comdat::SelectionKind::Largest is a behavior that comes from COFF.
  bool DstAnyOrLargest = Dst == Comdat::SelectionKind::Any ||
                         Dst == Comdat::SelectionKind::Largest;
  bool SrcAnyOrLargest = Src == Comdat::SelectionKind::Any ||
                         Src == Comdat::SelectionKind::Largest;
  if (DstAnyOrLargest && SrcAnyOrLargest) {
    if (Dst == Comdat::SelectionKind::Largest ||
        Src == Comdat::SelectionKind::Largest)
      Result = Comdat::SelectionKind::Largest;
    else
      Result = Comdat::SelectionKind::Any;
  } else if (Src == Dst) {
    Result = Dst;
  } else {
    return emitError("Linking COMDATs named '" + ComdatName +
                     "': invalid selection kinds!");
  }

  switch (Result) {
  case Comdat::SelectionKind::Any:
    // Go with Dst.
    LinkFromSrc = false;
    break;
  case Comdat::SelectionKind::NoDuplicates:
    return emitError("Linking COMDATs named '" + ComdatName +
                     "': noduplicates has been violated!");
  case Comdat::SelectionKind::ExactMatch:
  case Comdat::SelectionKind::Largest:
  case Comdat::SelectionKind::SameSize: {
    const GlobalVariable *DstGV;
    const GlobalVariable *SrcGV;
    if (getComdatLeader(DstM, ComdatName, DstGV) ||
        getComdatLeader(SrcM, ComdatName, SrcGV))
      return true;

    const DataLayout *DstDL = DstM->getDataLayout();
    const DataLayout *SrcDL = SrcM->getDataLayout();
    if (!DstDL || !SrcDL) {
      return emitError(
          "Linking COMDATs named '" + ComdatName +
          "': can't do size dependent selection without DataLayout!");
    }
    uint64_t DstSize =
        DstDL->getTypeAllocSize(DstGV->getType()->getPointerElementType());
    uint64_t SrcSize =
        SrcDL->getTypeAllocSize(SrcGV->getType()->getPointerElementType());
    if (Result == Comdat::SelectionKind::ExactMatch) {
      if (SrcGV->getInitializer() != DstGV->getInitializer())
        return emitError("Linking COMDATs named '" + ComdatName +
                         "': ExactMatch violated!");
      LinkFromSrc = false;
    } else if (Result == Comdat::SelectionKind::Largest) {
      LinkFromSrc = SrcSize > DstSize;
    } else if (Result == Comdat::SelectionKind::SameSize) {
      if (SrcSize != DstSize)
        return emitError("Linking COMDATs named '" + ComdatName +
                         "': SameSize violated!");
      LinkFromSrc = false;
    } else {
      llvm_unreachable("unknown selection kind");
    }
    break;
  }
  }

  return false;
}

bool ModuleLinker::getComdatResult(const Comdat *SrcC,
                                   Comdat::SelectionKind &Result,
                                   bool &LinkFromSrc) {
  Comdat::SelectionKind SSK = SrcC->getSelectionKind();
  StringRef ComdatName = SrcC->getName();
  Module::ComdatSymTabType &ComdatSymTab = DstM->getComdatSymbolTable();
  Module::ComdatSymTabType::iterator DstCI = ComdatSymTab.find(ComdatName);

  if (DstCI == ComdatSymTab.end()) {
    // Use the comdat if it is only available in one of the modules.
    LinkFromSrc = true;
    Result = SSK;
    return false;
  }

  const Comdat *DstC = &DstCI->second;
  Comdat::SelectionKind DSK = DstC->getSelectionKind();
  return computeResultingSelectionKind(ComdatName, SSK, DSK, Result,
                                       LinkFromSrc);
}

bool ModuleLinker::shouldLinkFromSource(bool &LinkFromSrc,
                                        const GlobalValue &Dest,
                                        const GlobalValue &Src) {
  bool SrcIsDeclaration = Src.isDeclarationForLinker();
  bool DestIsDeclaration = Dest.isDeclarationForLinker();

  if (SrcIsDeclaration) {
    // If Src is external or if both Src & Dest are external..  Just link the
    // external globals, we aren't adding anything.
    if (Src.hasDLLImportStorageClass()) {
      // If one of GVs is marked as DLLImport, result should be dllimport'ed.
      LinkFromSrc = DestIsDeclaration;
      return false;
    }
    // If the Dest is weak, use the source linkage.
    LinkFromSrc = Dest.hasExternalWeakLinkage();
    return false;
  }

  if (DestIsDeclaration) {
    // If Dest is external but Src is not:
    LinkFromSrc = true;
    return false;
  }

  if (Src.hasCommonLinkage()) {
    if (Dest.hasLinkOnceLinkage() || Dest.hasWeakLinkage()) {
      LinkFromSrc = true;
      return false;
    }

    if (!Dest.hasCommonLinkage()) {
      LinkFromSrc = false;
      return false;
    }

    // FIXME: Make datalayout mandatory and just use getDataLayout().
    DataLayout DL(Dest.getParent());

    uint64_t DestSize = DL.getTypeAllocSize(Dest.getType()->getElementType());
    uint64_t SrcSize = DL.getTypeAllocSize(Src.getType()->getElementType());
    LinkFromSrc = SrcSize > DestSize;
    return false;
  }

  if (Src.isWeakForLinker()) {
    assert(!Dest.hasExternalWeakLinkage());
    assert(!Dest.hasAvailableExternallyLinkage());

    if (Dest.hasLinkOnceLinkage() && Src.hasWeakLinkage()) {
      LinkFromSrc = true;
      return false;
    }

    LinkFromSrc = false;
    return false;
  }

  if (Dest.isWeakForLinker()) {
    assert(Src.hasExternalLinkage());
    LinkFromSrc = true;
    return false;
  }

  assert(!Src.hasExternalWeakLinkage());
  assert(!Dest.hasExternalWeakLinkage());
  assert(Dest.hasExternalLinkage() && Src.hasExternalLinkage() &&
         "Unexpected linkage type!");
  return emitError("Linking globals named '" + Src.getName() +
                   "': symbol multiply defined!");
}

/// This analyzes the two global values and determines what the result will look
/// like in the destination module. In particular, it computes the resultant
/// linkage type and visibility, computes whether the global in the source
/// should be copied over to the destination (replacing the existing one), and
/// computes whether this linkage is an error or not.
bool ModuleLinker::getLinkageResult(GlobalValue *Dest, const GlobalValue *Src,
                                    GlobalValue::LinkageTypes &LT,
                                    GlobalValue::VisibilityTypes &Vis,
                                    bool &LinkFromSrc) {
  assert(Dest && "Must have two globals being queried");
  assert(!Src->hasLocalLinkage() &&
         "If Src has internal linkage, Dest shouldn't be set!");

  if (shouldLinkFromSource(LinkFromSrc, *Dest, *Src))
    return true;

  if (LinkFromSrc)
    LT = Src->getLinkage();
  else
    LT = Dest->getLinkage();

  // Compute the visibility. We follow the rules in the System V Application
  // Binary Interface.
  assert(!GlobalValue::isLocalLinkage(LT) &&
         "Symbols with local linkage should not be merged");
  Vis = isLessConstraining(Src->getVisibility(), Dest->getVisibility()) ?
    Dest->getVisibility() : Src->getVisibility();
  return false;
}

/// Loop over all of the linked values to compute type mappings.  For example,
/// if we link "extern Foo *x" and "Foo *x = NULL", then we have two struct
/// types 'Foo' but one got renamed when the module was loaded into the same
/// LLVMContext.
void ModuleLinker::computeTypeMapping() {
  // Incorporate globals.
  for (Module::global_iterator I = SrcM->global_begin(),
       E = SrcM->global_end(); I != E; ++I) {
    GlobalValue *DGV = getLinkedToGlobal(I);
    if (!DGV) continue;

    if (!DGV->hasAppendingLinkage() || !I->hasAppendingLinkage()) {
      TypeMap.addTypeMapping(DGV->getType(), I->getType());
      continue;
    }

    // Unify the element type of appending arrays.
    ArrayType *DAT = cast<ArrayType>(DGV->getType()->getElementType());
    ArrayType *SAT = cast<ArrayType>(I->getType()->getElementType());
    TypeMap.addTypeMapping(DAT->getElementType(), SAT->getElementType());
  }

  // Incorporate functions.
  for (Module::iterator I = SrcM->begin(), E = SrcM->end(); I != E; ++I) {
    if (GlobalValue *DGV = getLinkedToGlobal(I))
      TypeMap.addTypeMapping(DGV->getType(), I->getType());
  }

  // Incorporate types by name, scanning all the types in the source module.
  // At this point, the destination module may have a type "%foo = { i32 }" for
  // example.  When the source module got loaded into the same LLVMContext, if
  // it had the same type, it would have been renamed to "%foo.42 = { i32 }".
  TypeFinder SrcStructTypes;
  SrcStructTypes.run(*SrcM, true);
  SmallPtrSet<StructType*, 32> SrcStructTypesSet(SrcStructTypes.begin(),
                                                 SrcStructTypes.end());

  for (unsigned i = 0, e = SrcStructTypes.size(); i != e; ++i) {
    StructType *ST = SrcStructTypes[i];
    if (!ST->hasName()) continue;

    // Check to see if there is a dot in the name followed by a digit.
    size_t DotPos = ST->getName().rfind('.');
    if (DotPos == 0 || DotPos == StringRef::npos ||
        ST->getName().back() == '.' ||
        !isdigit(static_cast<unsigned char>(ST->getName()[DotPos+1])))
      continue;

    // Check to see if the destination module has a struct with the prefix name.
    if (StructType *DST = DstM->getTypeByName(ST->getName().substr(0, DotPos)))
      // Don't use it if this actually came from the source module. They're in
      // the same LLVMContext after all. Also don't use it unless the type is
      // actually used in the destination module. This can happen in situations
      // like this:
      //
      //      Module A                         Module B
      //      --------                         --------
      //   %Z = type { %A }                %B = type { %C.1 }
      //   %A = type { %B.1, [7 x i8] }    %C.1 = type { i8* }
      //   %B.1 = type { %C }              %A.2 = type { %B.3, [5 x i8] }
      //   %C = type { i8* }               %B.3 = type { %C.1 }
      //
      // When we link Module B with Module A, the '%B' in Module B is
      // used. However, that would then use '%C.1'. But when we process '%C.1',
      // we prefer to take the '%C' version. So we are then left with both
      // '%C.1' and '%C' being used for the same types. This leads to some
      // variables using one type and some using the other.
      if (!SrcStructTypesSet.count(DST) && TypeMap.DstStructTypesSet.count(DST))
        TypeMap.addTypeMapping(DST, ST);
  }

  // Don't bother incorporating aliases, they aren't generally typed well.

  // Now that we have discovered all of the type equivalences, get a body for
  // any 'opaque' types in the dest module that are now resolved.
  TypeMap.linkDefinedTypeBodies();
}

static void upgradeGlobalArray(GlobalVariable *GV) {
  ArrayType *ATy = cast<ArrayType>(GV->getType()->getElementType());
  StructType *OldTy = cast<StructType>(ATy->getElementType());
  assert(OldTy->getNumElements() == 2 && "Expected to upgrade from 2 elements");

  // Get the upgraded 3 element type.
  PointerType *VoidPtrTy = Type::getInt8Ty(GV->getContext())->getPointerTo();
  Type *Tys[3] = {OldTy->getElementType(0), OldTy->getElementType(1),
                  VoidPtrTy};
  StructType *NewTy = StructType::get(GV->getContext(), Tys, false);

  // Build new constants with a null third field filled in.
  Constant *OldInitC = GV->getInitializer();
  ConstantArray *OldInit = dyn_cast<ConstantArray>(OldInitC);
  if (!OldInit && !isa<ConstantAggregateZero>(OldInitC))
    // Invalid initializer; give up.
    return;
  std::vector<Constant *> Initializers;
  if (OldInit && OldInit->getNumOperands()) {
    Value *Null = Constant::getNullValue(VoidPtrTy);
    for (Use &U : OldInit->operands()) {
      ConstantStruct *Init = cast<ConstantStruct>(U.get());
      Initializers.push_back(ConstantStruct::get(
          NewTy, Init->getOperand(0), Init->getOperand(1), Null, nullptr));
    }
  }
  assert(Initializers.size() == ATy->getNumElements() &&
         "Failed to copy all array elements");

  // Replace the old GV with a new one.
  ATy = ArrayType::get(NewTy, Initializers.size());
  Constant *NewInit = ConstantArray::get(ATy, Initializers);
  GlobalVariable *NewGV = new GlobalVariable(
      *GV->getParent(), ATy, GV->isConstant(), GV->getLinkage(), NewInit, "",
      GV, GV->getThreadLocalMode(), GV->getType()->getAddressSpace(),
      GV->isExternallyInitialized());
  NewGV->copyAttributesFrom(GV);
  NewGV->takeName(GV);
  assert(GV->use_empty() && "program cannot use initializer list");
  GV->eraseFromParent();
}

void ModuleLinker::upgradeMismatchedGlobalArray(StringRef Name) {
  // Look for the global arrays.
  auto *DstGV = dyn_cast_or_null<GlobalVariable>(DstM->getNamedValue(Name));
  if (!DstGV)
    return;
  auto *SrcGV = dyn_cast_or_null<GlobalVariable>(SrcM->getNamedValue(Name));
  if (!SrcGV)
    return;

  // Check if the types already match.
  auto *DstTy = cast<ArrayType>(DstGV->getType()->getElementType());
  auto *SrcTy =
      cast<ArrayType>(TypeMap.get(SrcGV->getType()->getElementType()));
  if (DstTy == SrcTy)
    return;

  // Grab the element types.  We can only upgrade an array of a two-field
  // struct.  Only bother if the other one has three-fields.
  auto *DstEltTy = cast<StructType>(DstTy->getElementType());
  auto *SrcEltTy = cast<StructType>(SrcTy->getElementType());
  if (DstEltTy->getNumElements() == 2 && SrcEltTy->getNumElements() == 3) {
    upgradeGlobalArray(DstGV);
    return;
  }
  if (DstEltTy->getNumElements() == 3 && SrcEltTy->getNumElements() == 2)
    upgradeGlobalArray(SrcGV);

  // We can't upgrade any other differences.
}

void ModuleLinker::upgradeMismatchedGlobals() {
  upgradeMismatchedGlobalArray("llvm.global_ctors");
  upgradeMismatchedGlobalArray("llvm.global_dtors");
}

/// If there were any appending global variables, link them together now.
/// Return true on error.
bool ModuleLinker::linkAppendingVarProto(GlobalVariable *DstGV,
                                         const GlobalVariable *SrcGV) {

  if (!SrcGV->hasAppendingLinkage() || !DstGV->hasAppendingLinkage())
    return emitError("Linking globals named '" + SrcGV->getName() +
           "': can only link appending global with another appending global!");

  ArrayType *DstTy = cast<ArrayType>(DstGV->getType()->getElementType());
  ArrayType *SrcTy =
    cast<ArrayType>(TypeMap.get(SrcGV->getType()->getElementType()));
  Type *EltTy = DstTy->getElementType();

  // Check to see that they two arrays agree on type.
  if (EltTy != SrcTy->getElementType())
    return emitError("Appending variables with different element types!");
  if (DstGV->isConstant() != SrcGV->isConstant())
    return emitError("Appending variables linked with different const'ness!");

  if (DstGV->getAlignment() != SrcGV->getAlignment())
    return emitError(
             "Appending variables with different alignment need to be linked!");

  if (DstGV->getVisibility() != SrcGV->getVisibility())
    return emitError(
            "Appending variables with different visibility need to be linked!");

  if (DstGV->hasUnnamedAddr() != SrcGV->hasUnnamedAddr())
    return emitError(
        "Appending variables with different unnamed_addr need to be linked!");

  if (StringRef(DstGV->getSection()) != SrcGV->getSection())
    return emitError(
          "Appending variables with different section name need to be linked!");

  uint64_t NewSize = DstTy->getNumElements() + SrcTy->getNumElements();
  ArrayType *NewType = ArrayType::get(EltTy, NewSize);

  // Create the new global variable.
  GlobalVariable *NG =
    new GlobalVariable(*DstGV->getParent(), NewType, SrcGV->isConstant(),
                       DstGV->getLinkage(), /*init*/nullptr, /*name*/"", DstGV,
                       DstGV->getThreadLocalMode(),
                       DstGV->getType()->getAddressSpace());

  // Propagate alignment, visibility and section info.
  copyGVAttributes(NG, DstGV);

  AppendingVarInfo AVI;
  AVI.NewGV = NG;
  AVI.DstInit = DstGV->getInitializer();
  AVI.SrcInit = SrcGV->getInitializer();
  AppendingVars.push_back(AVI);

  // Replace any uses of the two global variables with uses of the new
  // global.
  ValueMap[SrcGV] = ConstantExpr::getBitCast(NG, TypeMap.get(SrcGV->getType()));

  DstGV->replaceAllUsesWith(ConstantExpr::getBitCast(NG, DstGV->getType()));
  DstGV->eraseFromParent();

  // Track the source variable so we don't try to link it.
  DoNotLinkFromSource.insert(SrcGV);

  return false;
}

/// Loop through the global variables in the src module and merge them into the
/// dest module.
bool ModuleLinker::linkGlobalProto(const GlobalVariable *SGV) {
  GlobalValue *DGV = getLinkedToGlobal(SGV);
  llvm::Optional<GlobalValue::VisibilityTypes> NewVisibility;
  bool HasUnnamedAddr = SGV->hasUnnamedAddr();
  unsigned Alignment = SGV->getAlignment();

  bool LinkFromSrc = false;
  Comdat *DC = nullptr;
  if (const Comdat *SC = SGV->getComdat()) {
    Comdat::SelectionKind SK;
    std::tie(SK, LinkFromSrc) = ComdatsChosen[SC];
    DC = DstM->getOrInsertComdat(SC->getName());
    DC->setSelectionKind(SK);
  }

  if (DGV) {
    if (!DC) {
      // Concatenation of appending linkage variables is magic and handled later.
      if (DGV->hasAppendingLinkage() || SGV->hasAppendingLinkage())
        return linkAppendingVarProto(cast<GlobalVariable>(DGV), SGV);

      // Determine whether linkage of these two globals follows the source
      // module's definition or the destination module's definition.
      GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
      GlobalValue::VisibilityTypes NV;
      if (getLinkageResult(DGV, SGV, NewLinkage, NV, LinkFromSrc))
        return true;
      NewVisibility = NV;
      HasUnnamedAddr = HasUnnamedAddr && DGV->hasUnnamedAddr();
      if (DGV->hasCommonLinkage() && SGV->hasCommonLinkage())
        Alignment = std::max(Alignment, DGV->getAlignment());
      else if (!LinkFromSrc)
        Alignment = DGV->getAlignment();

      // If we're not linking from the source, then keep the definition that we
      // have.
      if (!LinkFromSrc) {
        // Special case for const propagation.
        if (GlobalVariable *DGVar = dyn_cast<GlobalVariable>(DGV)) {
          DGVar->setAlignment(Alignment);

          if (DGVar->isDeclaration() && !SGV->isConstant())
            DGVar->setConstant(false);
        }

        // Set calculated linkage, visibility and unnamed_addr.
        DGV->setLinkage(NewLinkage);
        DGV->setVisibility(*NewVisibility);
        DGV->setUnnamedAddr(HasUnnamedAddr);
      }
    }

    if (!LinkFromSrc) {
      // Make sure to remember this mapping.
      ValueMap[SGV] = ConstantExpr::getBitCast(DGV,TypeMap.get(SGV->getType()));

      // Track the source global so that we don't attempt to copy it over when
      // processing global initializers.
      DoNotLinkFromSource.insert(SGV);

      return false;
    }
  }

  // If the Comdat this variable was inside of wasn't selected, skip it.
  if (DC && !DGV && !LinkFromSrc) {
    DoNotLinkFromSource.insert(SGV);
    return false;
  }

  // No linking to be performed or linking from the source: simply create an
  // identical version of the symbol over in the dest module... the
  // initializer will be filled in later by LinkGlobalInits.
  GlobalVariable *NewDGV =
    new GlobalVariable(*DstM, TypeMap.get(SGV->getType()->getElementType()),
                       SGV->isConstant(), SGV->getLinkage(), /*init*/nullptr,
                       SGV->getName(), /*insertbefore*/nullptr,
                       SGV->getThreadLocalMode(),
                       SGV->getType()->getAddressSpace());
  // Propagate alignment, visibility and section info.
  copyGVAttributes(NewDGV, SGV);
  NewDGV->setAlignment(Alignment);
  if (NewVisibility)
    NewDGV->setVisibility(*NewVisibility);
  NewDGV->setUnnamedAddr(HasUnnamedAddr);

  if (DC)
    NewDGV->setComdat(DC);

  if (DGV) {
    DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDGV, DGV->getType()));
    DGV->eraseFromParent();
  }

  // Make sure to remember this mapping.
  ValueMap[SGV] = NewDGV;
  return false;
}

/// Link the function in the source module into the destination module if
/// needed, setting up mapping information.
bool ModuleLinker::linkFunctionProto(Function *SF) {
  GlobalValue *DGV = getLinkedToGlobal(SF);
  llvm::Optional<GlobalValue::VisibilityTypes> NewVisibility;
  bool HasUnnamedAddr = SF->hasUnnamedAddr();

  bool LinkFromSrc = false;
  Comdat *DC = nullptr;
  if (const Comdat *SC = SF->getComdat()) {
    Comdat::SelectionKind SK;
    std::tie(SK, LinkFromSrc) = ComdatsChosen[SC];
    DC = DstM->getOrInsertComdat(SC->getName());
    DC->setSelectionKind(SK);
  }

  if (DGV) {
    if (!DC) {
      GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
      GlobalValue::VisibilityTypes NV;
      if (getLinkageResult(DGV, SF, NewLinkage, NV, LinkFromSrc))
        return true;
      NewVisibility = NV;
      HasUnnamedAddr = HasUnnamedAddr && DGV->hasUnnamedAddr();

      if (!LinkFromSrc) {
        // Set calculated linkage
        DGV->setLinkage(NewLinkage);
        DGV->setVisibility(*NewVisibility);
        DGV->setUnnamedAddr(HasUnnamedAddr);
      }
    }

    if (!LinkFromSrc) {
      // Make sure to remember this mapping.
      ValueMap[SF] = ConstantExpr::getBitCast(DGV, TypeMap.get(SF->getType()));

      // Track the function from the source module so we don't attempt to remap
      // it.
      DoNotLinkFromSource.insert(SF);

      return false;
    }
  }

  // If the function is to be lazily linked, don't create it just yet.
  // The ValueMaterializerTy will deal with creating it if it's used.
  if (!DGV && (SF->hasLocalLinkage() || SF->hasLinkOnceLinkage() ||
               SF->hasAvailableExternallyLinkage())) {
    DoNotLinkFromSource.insert(SF);
    return false;
  }

  // If the Comdat this function was inside of wasn't selected, skip it.
  if (DC && !DGV && !LinkFromSrc) {
    DoNotLinkFromSource.insert(SF);
    return false;
  }

  // If there is no linkage to be performed or we are linking from the source,
  // bring SF over.
  Function *NewDF = Function::Create(TypeMap.get(SF->getFunctionType()),
                                     SF->getLinkage(), SF->getName(), DstM);
  copyGVAttributes(NewDF, SF);
  if (NewVisibility)
    NewDF->setVisibility(*NewVisibility);
  NewDF->setUnnamedAddr(HasUnnamedAddr);

  if (DC)
    NewDF->setComdat(DC);

  if (DGV) {
    // Any uses of DF need to change to NewDF, with cast.
    DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDF, DGV->getType()));
    DGV->eraseFromParent();
  }

  ValueMap[SF] = NewDF;
  return false;
}

/// Set up prototypes for any aliases that come over from the source module.
bool ModuleLinker::linkAliasProto(GlobalAlias *SGA) {
  GlobalValue *DGV = getLinkedToGlobal(SGA);
  llvm::Optional<GlobalValue::VisibilityTypes> NewVisibility;
  bool HasUnnamedAddr = SGA->hasUnnamedAddr();

  bool LinkFromSrc = false;
  Comdat *DC = nullptr;
  if (const Comdat *SC = SGA->getComdat()) {
    Comdat::SelectionKind SK;
    std::tie(SK, LinkFromSrc) = ComdatsChosen[SC];
    DC = DstM->getOrInsertComdat(SC->getName());
    DC->setSelectionKind(SK);
  }

  if (DGV) {
    if (!DC) {
      GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
      GlobalValue::VisibilityTypes NV;
      if (getLinkageResult(DGV, SGA, NewLinkage, NV, LinkFromSrc))
        return true;
      NewVisibility = NV;
      HasUnnamedAddr = HasUnnamedAddr && DGV->hasUnnamedAddr();

      if (!LinkFromSrc) {
        // Set calculated linkage.
        DGV->setLinkage(NewLinkage);
        DGV->setVisibility(*NewVisibility);
        DGV->setUnnamedAddr(HasUnnamedAddr);
      }
    }

    if (!LinkFromSrc) {
      // Make sure to remember this mapping.
      ValueMap[SGA] = ConstantExpr::getBitCast(DGV,TypeMap.get(SGA->getType()));

      // Track the alias from the source module so we don't attempt to remap it.
      DoNotLinkFromSource.insert(SGA);

      return false;
    }
  }

  // If the Comdat this alias was inside of wasn't selected, skip it.
  if (DC && !DGV && !LinkFromSrc) {
    DoNotLinkFromSource.insert(SGA);
    return false;
  }

  // If there is no linkage to be performed or we're linking from the source,
  // bring over SGA.
  auto *PTy = cast<PointerType>(TypeMap.get(SGA->getType()));
  auto *NewDA =
      GlobalAlias::create(PTy->getElementType(), PTy->getAddressSpace(),
                          SGA->getLinkage(), SGA->getName(), DstM);
  copyGVAttributes(NewDA, SGA);
  if (NewVisibility)
    NewDA->setVisibility(*NewVisibility);
  NewDA->setUnnamedAddr(HasUnnamedAddr);

  if (DGV) {
    // Any uses of DGV need to change to NewDA, with cast.
    DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDA, DGV->getType()));
    DGV->eraseFromParent();
  }

  ValueMap[SGA] = NewDA;
  return false;
}

static void getArrayElements(const Constant *C,
                             SmallVectorImpl<Constant *> &Dest) {
  unsigned NumElements = cast<ArrayType>(C->getType())->getNumElements();

  for (unsigned i = 0; i != NumElements; ++i)
    Dest.push_back(C->getAggregateElement(i));
}

void ModuleLinker::linkAppendingVarInit(const AppendingVarInfo &AVI) {
  // Merge the initializer.
  SmallVector<Constant *, 16> DstElements;
  getArrayElements(AVI.DstInit, DstElements);

  SmallVector<Constant *, 16> SrcElements;
  getArrayElements(AVI.SrcInit, SrcElements);

  ArrayType *NewType = cast<ArrayType>(AVI.NewGV->getType()->getElementType());

  StringRef Name = AVI.NewGV->getName();
  bool IsNewStructor =
      (Name == "llvm.global_ctors" || Name == "llvm.global_dtors") &&
      cast<StructType>(NewType->getElementType())->getNumElements() == 3;

  for (auto *V : SrcElements) {
    if (IsNewStructor) {
      Constant *Key = V->getAggregateElement(2);
      if (DoNotLinkFromSource.count(Key))
        continue;
    }
    DstElements.push_back(
        MapValue(V, ValueMap, RF_None, &TypeMap, &ValMaterializer));
  }
  if (IsNewStructor) {
    NewType = ArrayType::get(NewType->getElementType(), DstElements.size());
    AVI.NewGV->mutateType(PointerType::get(NewType, 0));
  }

  AVI.NewGV->setInitializer(ConstantArray::get(NewType, DstElements));
}

/// Update the initializers in the Dest module now that all globals that may be
/// referenced are in Dest.
void ModuleLinker::linkGlobalInits() {
  // Loop over all of the globals in the src module, mapping them over as we go
  for (Module::const_global_iterator I = SrcM->global_begin(),
       E = SrcM->global_end(); I != E; ++I) {

    // Only process initialized GV's or ones not already in dest.
    if (!I->hasInitializer() || DoNotLinkFromSource.count(I)) continue;

    // Grab destination global variable.
    GlobalVariable *DGV = cast<GlobalVariable>(ValueMap[I]);
    // Figure out what the initializer looks like in the dest module.
    DGV->setInitializer(MapValue(I->getInitializer(), ValueMap,
                                 RF_None, &TypeMap, &ValMaterializer));
  }
}

/// Copy the source function over into the dest function and fix up references
/// to values. At this point we know that Dest is an external function, and
/// that Src is not.
void ModuleLinker::linkFunctionBody(Function *Dst, Function *Src) {
  assert(Src && Dst && Dst->isDeclaration() && !Src->isDeclaration());

  // Go through and convert function arguments over, remembering the mapping.
  Function::arg_iterator DI = Dst->arg_begin();
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I, ++DI) {
    DI->setName(I->getName());  // Copy the name over.

    // Add a mapping to our mapping.
    ValueMap[I] = DI;
  }

  // Splice the body of the source function into the dest function.
  Dst->getBasicBlockList().splice(Dst->end(), Src->getBasicBlockList());

  // At this point, all of the instructions and values of the function are now
  // copied over.  The only problem is that they are still referencing values in
  // the Source function as operands.  Loop through all of the operands of the
  // functions and patch them up to point to the local versions.
  for (Function::iterator BB = Dst->begin(), BE = Dst->end(); BB != BE; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      RemapInstruction(I, ValueMap, RF_IgnoreMissingEntries, &TypeMap,
                       &ValMaterializer);

  // There is no need to map the arguments anymore.
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I)
    ValueMap.erase(I);

}

/// Insert all of the aliases in Src into the Dest module.
void ModuleLinker::linkAliasBodies() {
  for (Module::alias_iterator I = SrcM->alias_begin(), E = SrcM->alias_end();
       I != E; ++I) {
    if (DoNotLinkFromSource.count(I))
      continue;
    if (Constant *Aliasee = I->getAliasee()) {
      GlobalAlias *DA = cast<GlobalAlias>(ValueMap[I]);
      Constant *Val =
          MapValue(Aliasee, ValueMap, RF_None, &TypeMap, &ValMaterializer);
      DA->setAliasee(Val);
    }
  }
}

/// Insert all of the named MDNodes in Src into the Dest module.
void ModuleLinker::linkNamedMDNodes() {
  const NamedMDNode *SrcModFlags = SrcM->getModuleFlagsMetadata();
  for (Module::const_named_metadata_iterator I = SrcM->named_metadata_begin(),
       E = SrcM->named_metadata_end(); I != E; ++I) {
    // Don't link module flags here. Do them separately.
    if (&*I == SrcModFlags) continue;
    NamedMDNode *DestNMD = DstM->getOrInsertNamedMetadata(I->getName());
    // Add Src elements into Dest node.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      DestNMD->addOperand(MapValue(I->getOperand(i), ValueMap,
                                   RF_None, &TypeMap, &ValMaterializer));
  }
}

/// Merge the linker flags in Src into the Dest module.
bool ModuleLinker::linkModuleFlagsMetadata() {
  // If the source module has no module flags, we are done.
  const NamedMDNode *SrcModFlags = SrcM->getModuleFlagsMetadata();
  if (!SrcModFlags) return false;

  // If the destination module doesn't have module flags yet, then just copy
  // over the source module's flags.
  NamedMDNode *DstModFlags = DstM->getOrInsertModuleFlagsMetadata();
  if (DstModFlags->getNumOperands() == 0) {
    for (unsigned I = 0, E = SrcModFlags->getNumOperands(); I != E; ++I)
      DstModFlags->addOperand(SrcModFlags->getOperand(I));

    return false;
  }

  // First build a map of the existing module flags and requirements.
  DenseMap<MDString*, MDNode*> Flags;
  SmallSetVector<MDNode*, 16> Requirements;
  for (unsigned I = 0, E = DstModFlags->getNumOperands(); I != E; ++I) {
    MDNode *Op = DstModFlags->getOperand(I);
    ConstantInt *Behavior = cast<ConstantInt>(Op->getOperand(0));
    MDString *ID = cast<MDString>(Op->getOperand(1));

    if (Behavior->getZExtValue() == Module::Require) {
      Requirements.insert(cast<MDNode>(Op->getOperand(2)));
    } else {
      Flags[ID] = Op;
    }
  }

  // Merge in the flags from the source module, and also collect its set of
  // requirements.
  bool HasErr = false;
  for (unsigned I = 0, E = SrcModFlags->getNumOperands(); I != E; ++I) {
    MDNode *SrcOp = SrcModFlags->getOperand(I);
    ConstantInt *SrcBehavior = cast<ConstantInt>(SrcOp->getOperand(0));
    MDString *ID = cast<MDString>(SrcOp->getOperand(1));
    MDNode *DstOp = Flags.lookup(ID);
    unsigned SrcBehaviorValue = SrcBehavior->getZExtValue();

    // If this is a requirement, add it and continue.
    if (SrcBehaviorValue == Module::Require) {
      // If the destination module does not already have this requirement, add
      // it.
      if (Requirements.insert(cast<MDNode>(SrcOp->getOperand(2)))) {
        DstModFlags->addOperand(SrcOp);
      }
      continue;
    }

    // If there is no existing flag with this ID, just add it.
    if (!DstOp) {
      Flags[ID] = SrcOp;
      DstModFlags->addOperand(SrcOp);
      continue;
    }

    // Otherwise, perform a merge.
    ConstantInt *DstBehavior = cast<ConstantInt>(DstOp->getOperand(0));
    unsigned DstBehaviorValue = DstBehavior->getZExtValue();

    // If either flag has override behavior, handle it first.
    if (DstBehaviorValue == Module::Override) {
      // Diagnose inconsistent flags which both have override behavior.
      if (SrcBehaviorValue == Module::Override &&
          SrcOp->getOperand(2) != DstOp->getOperand(2)) {
        HasErr |= emitError("linking module flags '" + ID->getString() +
                            "': IDs have conflicting override values");
      }
      continue;
    } else if (SrcBehaviorValue == Module::Override) {
      // Update the destination flag to that of the source.
      DstOp->replaceOperandWith(0, SrcBehavior);
      DstOp->replaceOperandWith(2, SrcOp->getOperand(2));
      continue;
    }

    // Diagnose inconsistent merge behavior types.
    if (SrcBehaviorValue != DstBehaviorValue) {
      HasErr |= emitError("linking module flags '" + ID->getString() +
                          "': IDs have conflicting behaviors");
      continue;
    }

    // Perform the merge for standard behavior types.
    switch (SrcBehaviorValue) {
    case Module::Require:
    case Module::Override: llvm_unreachable("not possible");
    case Module::Error: {
      // Emit an error if the values differ.
      if (SrcOp->getOperand(2) != DstOp->getOperand(2)) {
        HasErr |= emitError("linking module flags '" + ID->getString() +
                            "': IDs have conflicting values");
      }
      continue;
    }
    case Module::Warning: {
      // Emit a warning if the values differ.
      if (SrcOp->getOperand(2) != DstOp->getOperand(2)) {
        emitWarning("linking module flags '" + ID->getString() +
                    "': IDs have conflicting values");
      }
      continue;
    }
    case Module::Append: {
      MDNode *DstValue = cast<MDNode>(DstOp->getOperand(2));
      MDNode *SrcValue = cast<MDNode>(SrcOp->getOperand(2));
      unsigned NumOps = DstValue->getNumOperands() + SrcValue->getNumOperands();
      Value **VP, **Values = VP = new Value*[NumOps];
      for (unsigned i = 0, e = DstValue->getNumOperands(); i != e; ++i, ++VP)
        *VP = DstValue->getOperand(i);
      for (unsigned i = 0, e = SrcValue->getNumOperands(); i != e; ++i, ++VP)
        *VP = SrcValue->getOperand(i);
      DstOp->replaceOperandWith(2, MDNode::get(DstM->getContext(),
                                               ArrayRef<Value*>(Values,
                                                                NumOps)));
      delete[] Values;
      break;
    }
    case Module::AppendUnique: {
      SmallSetVector<Value*, 16> Elts;
      MDNode *DstValue = cast<MDNode>(DstOp->getOperand(2));
      MDNode *SrcValue = cast<MDNode>(SrcOp->getOperand(2));
      for (unsigned i = 0, e = DstValue->getNumOperands(); i != e; ++i)
        Elts.insert(DstValue->getOperand(i));
      for (unsigned i = 0, e = SrcValue->getNumOperands(); i != e; ++i)
        Elts.insert(SrcValue->getOperand(i));
      DstOp->replaceOperandWith(2, MDNode::get(DstM->getContext(),
                                               ArrayRef<Value*>(Elts.begin(),
                                                                Elts.end())));
      break;
    }
    }
  }

  // Check all of the requirements.
  for (unsigned I = 0, E = Requirements.size(); I != E; ++I) {
    MDNode *Requirement = Requirements[I];
    MDString *Flag = cast<MDString>(Requirement->getOperand(0));
    Value *ReqValue = Requirement->getOperand(1);

    MDNode *Op = Flags[Flag];
    if (!Op || Op->getOperand(2) != ReqValue) {
      HasErr |= emitError("linking module flags '" + Flag->getString() +
                          "': does not have the required value");
      continue;
    }
  }

  return HasErr;
}

bool ModuleLinker::run() {
  assert(DstM && "Null destination module");
  assert(SrcM && "Null source module");

  // Inherit the target data from the source module if the destination module
  // doesn't have one already.
  if (!DstM->getDataLayout() && SrcM->getDataLayout())
    DstM->setDataLayout(SrcM->getDataLayout());

  // Copy the target triple from the source to dest if the dest's is empty.
  if (DstM->getTargetTriple().empty() && !SrcM->getTargetTriple().empty())
    DstM->setTargetTriple(SrcM->getTargetTriple());

  if (SrcM->getDataLayout() && DstM->getDataLayout() &&
      *SrcM->getDataLayout() != *DstM->getDataLayout()) {
    emitWarning("Linking two modules of different data layouts: '" +
                SrcM->getModuleIdentifier() + "' is '" +
                SrcM->getDataLayoutStr() + "' whereas '" +
                DstM->getModuleIdentifier() + "' is '" +
                DstM->getDataLayoutStr() + "'\n");
  }
  if (!SrcM->getTargetTriple().empty() &&
      DstM->getTargetTriple() != SrcM->getTargetTriple()) {
    emitWarning("Linking two modules of different target triples: " +
                SrcM->getModuleIdentifier() + "' is '" +
                SrcM->getTargetTriple() + "' whereas '" +
                DstM->getModuleIdentifier() + "' is '" +
                DstM->getTargetTriple() + "'\n");
  }

  // Append the module inline asm string.
  if (!SrcM->getModuleInlineAsm().empty()) {
    if (DstM->getModuleInlineAsm().empty())
      DstM->setModuleInlineAsm(SrcM->getModuleInlineAsm());
    else
      DstM->setModuleInlineAsm(DstM->getModuleInlineAsm()+"\n"+
                               SrcM->getModuleInlineAsm());
  }

  // Loop over all of the linked values to compute type mappings.
  computeTypeMapping();

  ComdatsChosen.clear();
  for (const StringMapEntry<llvm::Comdat> &SMEC : SrcM->getComdatSymbolTable()) {
    const Comdat &C = SMEC.getValue();
    if (ComdatsChosen.count(&C))
      continue;
    Comdat::SelectionKind SK;
    bool LinkFromSrc;
    if (getComdatResult(&C, SK, LinkFromSrc))
      return true;
    ComdatsChosen[&C] = std::make_pair(SK, LinkFromSrc);
  }

  // Upgrade mismatched global arrays.
  upgradeMismatchedGlobals();

  // Insert all of the globals in src into the DstM module... without linking
  // initializers (which could refer to functions not yet mapped over).
  for (Module::global_iterator I = SrcM->global_begin(),
       E = SrcM->global_end(); I != E; ++I)
    if (linkGlobalProto(I))
      return true;

  // Link the functions together between the two modules, without doing function
  // bodies... this just adds external function prototypes to the DstM
  // function...  We do this so that when we begin processing function bodies,
  // all of the global values that may be referenced are available in our
  // ValueMap.
  for (Module::iterator I = SrcM->begin(), E = SrcM->end(); I != E; ++I)
    if (linkFunctionProto(I))
      return true;

  // If there were any aliases, link them now.
  for (Module::alias_iterator I = SrcM->alias_begin(),
       E = SrcM->alias_end(); I != E; ++I)
    if (linkAliasProto(I))
      return true;

  for (unsigned i = 0, e = AppendingVars.size(); i != e; ++i)
    linkAppendingVarInit(AppendingVars[i]);

  // Link in the function bodies that are defined in the source module into
  // DstM.
  for (Module::iterator SF = SrcM->begin(), E = SrcM->end(); SF != E; ++SF) {
    // Skip if not linking from source.
    if (DoNotLinkFromSource.count(SF)) continue;

    Function *DF = cast<Function>(ValueMap[SF]);
    if (SF->hasPrefixData()) {
      // Link in the prefix data.
      DF->setPrefixData(MapValue(
          SF->getPrefixData(), ValueMap, RF_None, &TypeMap, &ValMaterializer));
    }

    // Materialize if needed.
    if (SF->isMaterializable()) {
      if (std::error_code EC = SF->materialize())
        return emitError(EC.message());
    }

    // Skip if no body (function is external).
    if (SF->isDeclaration())
      continue;

    linkFunctionBody(DF, SF);
    SF->Dematerialize();
  }

  // Resolve all uses of aliases with aliasees.
  linkAliasBodies();

  // Remap all of the named MDNodes in Src into the DstM module. We do this
  // after linking GlobalValues so that MDNodes that reference GlobalValues
  // are properly remapped.
  linkNamedMDNodes();

  // Merge the module flags into the DstM module.
  if (linkModuleFlagsMetadata())
    return true;

  // Update the initializers in the DstM module now that all globals that may
  // be referenced are in DstM.
  linkGlobalInits();

  // Process vector of lazily linked in functions.
  bool LinkedInAnyFunctions;
  do {
    LinkedInAnyFunctions = false;

    for(std::vector<Function*>::iterator I = LazilyLinkFunctions.begin(),
        E = LazilyLinkFunctions.end(); I != E; ++I) {
      Function *SF = *I;
      if (!SF)
        continue;

      Function *DF = cast<Function>(ValueMap[SF]);
      if (SF->hasPrefixData()) {
        // Link in the prefix data.
        DF->setPrefixData(MapValue(SF->getPrefixData(),
                                   ValueMap,
                                   RF_None,
                                   &TypeMap,
                                   &ValMaterializer));
      }

      // Materialize if needed.
      if (SF->isMaterializable()) {
        if (std::error_code EC = SF->materialize())
          return emitError(EC.message());
      }

      // Skip if no body (function is external).
      if (SF->isDeclaration())
        continue;

      // Erase from vector *before* the function body is linked - linkFunctionBody could
      // invalidate I.
      LazilyLinkFunctions.erase(I);

      // Link in function body.
      linkFunctionBody(DF, SF);
      SF->Dematerialize();

      // Set flag to indicate we may have more functions to lazily link in
      // since we linked in a function.
      LinkedInAnyFunctions = true;
      break;
    }
  } while (LinkedInAnyFunctions);

  // Now that all of the types from the source are used, resolve any structs
  // copied over to the dest that didn't exist there.
  TypeMap.linkDefinedTypeBodies();

  return false;
}

Linker::Linker(Module *M, DiagnosticHandlerFunction DiagnosticHandler)
    : Composite(M), DiagnosticHandler(DiagnosticHandler) {}

Linker::Linker(Module *M)
    : Composite(M), DiagnosticHandler([this](const DiagnosticInfo &DI) {
                      Composite->getContext().diagnose(DI);
                    }) {
  TypeFinder StructTypes;
  StructTypes.run(*M, true);
  IdentifiedStructTypes.insert(StructTypes.begin(), StructTypes.end());
}

Linker::~Linker() {
}

void Linker::deleteModule() {
  delete Composite;
  Composite = nullptr;
}

bool Linker::linkInModule(Module *Src) {
  ModuleLinker TheLinker(Composite, IdentifiedStructTypes, Src,
                         DiagnosticHandler);
  return TheLinker.run();
}

//===----------------------------------------------------------------------===//
// LinkModules entrypoint.
//===----------------------------------------------------------------------===//

/// This function links two modules together, with the resulting Dest module
/// modified to be the composite of the two input modules. If an error occurs,
/// true is returned and ErrorMsg (if not null) is set to indicate the problem.
/// Upon failure, the Dest module could be in a modified state, and shouldn't be
/// relied on to be consistent.
bool Linker::LinkModules(Module *Dest, Module *Src,
                         DiagnosticHandlerFunction DiagnosticHandler) {
  Linker L(Dest, DiagnosticHandler);
  return L.linkInModule(Src);
}

bool Linker::LinkModules(Module *Dest, Module *Src) {
  Linker L(Dest);
  return L.linkInModule(Src);
}

//===----------------------------------------------------------------------===//
// C API.
//===----------------------------------------------------------------------===//

LLVMBool LLVMLinkModules(LLVMModuleRef Dest, LLVMModuleRef Src,
                         LLVMLinkerMode Mode, char **OutMessages) {
  Module *D = unwrap(Dest);
  std::string Message;
  raw_string_ostream Stream(Message);
  DiagnosticPrinterRawOStream DP(Stream);

  LLVMBool Result = Linker::LinkModules(
      D, unwrap(Src), [&](const DiagnosticInfo &DI) { DI.print(DP); });

  if (OutMessages && Result)
    *OutMessages = strdup(Message.c_str());
  return Result;
}
