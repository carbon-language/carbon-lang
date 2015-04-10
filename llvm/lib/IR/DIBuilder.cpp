//===--- DIBuilder.cpp - Debug Information Builder ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DIBuilder.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DIBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

namespace {
class HeaderBuilder {
  /// \brief Whether there are any fields yet.
  ///
  /// Note that this is not equivalent to \c Chars.empty(), since \a concat()
  /// may have been called already with an empty string.
  bool IsEmpty;
  SmallVector<char, 256> Chars;

public:
  HeaderBuilder() : IsEmpty(true) {}
  HeaderBuilder(const HeaderBuilder &X) : IsEmpty(X.IsEmpty), Chars(X.Chars) {}
  HeaderBuilder(HeaderBuilder &&X)
      : IsEmpty(X.IsEmpty), Chars(std::move(X.Chars)) {}

  template <class Twineable> HeaderBuilder &concat(Twineable &&X) {
    if (IsEmpty)
      IsEmpty = false;
    else
      Chars.push_back(0);
    Twine(X).toVector(Chars);
    return *this;
  }

  MDString *get(LLVMContext &Context) const {
    return MDString::get(Context, StringRef(Chars.begin(), Chars.size()));
  }

  static HeaderBuilder get(unsigned Tag) {
    return HeaderBuilder().concat("0x" + Twine::utohexstr(Tag));
  }
};
}

DIBuilder::DIBuilder(Module &m, bool AllowUnresolvedNodes)
    : M(m), VMContext(M.getContext()), TempEnumTypes(nullptr),
      TempRetainTypes(nullptr), TempSubprograms(nullptr), TempGVs(nullptr),
      DeclareFn(nullptr), ValueFn(nullptr),
      AllowUnresolvedNodes(AllowUnresolvedNodes) {}

void DIBuilder::trackIfUnresolved(MDNode *N) {
  if (!N)
    return;
  if (N->isResolved())
    return;

  assert(AllowUnresolvedNodes && "Cannot handle unresolved nodes");
  UnresolvedNodes.emplace_back(N);
}

void DIBuilder::finalize() {
  DIArray Enums = getOrCreateArray(AllEnumTypes);
  TempEnumTypes->replaceAllUsesWith(Enums.get());

  SmallVector<Metadata *, 16> RetainValues;
  // Declarations and definitions of the same type may be retained. Some
  // clients RAUW these pairs, leaving duplicates in the retained types
  // list. Use a set to remove the duplicates while we transform the
  // TrackingVHs back into Values.
  SmallPtrSet<Metadata *, 16> RetainSet;
  for (unsigned I = 0, E = AllRetainTypes.size(); I < E; I++)
    if (RetainSet.insert(AllRetainTypes[I]).second)
      RetainValues.push_back(AllRetainTypes[I]);
  DIArray RetainTypes = getOrCreateArray(RetainValues);
  TempRetainTypes->replaceAllUsesWith(RetainTypes.get());

  DIArray SPs = getOrCreateArray(AllSubprograms);
  TempSubprograms->replaceAllUsesWith(SPs.get());
  for (unsigned i = 0, e = SPs.size(); i != e; ++i) {
    DISubprogram SP = cast<MDSubprogram>(SPs[i]);
    if (MDTuple *Temp = SP.getVariables().get()) {
      const auto &PV = PreservedVariables.lookup(SP);
      SmallVector<Metadata *, 4> Variables(PV.begin(), PV.end());
      DIArray AV = getOrCreateArray(Variables);
      TempMDTuple(Temp)->replaceAllUsesWith(AV.get());
    }
  }

  DIArray GVs = getOrCreateArray(AllGVs);
  TempGVs->replaceAllUsesWith(GVs.get());

  SmallVector<Metadata *, 16> RetainValuesI(AllImportedModules.begin(),
                                            AllImportedModules.end());
  DIArray IMs = getOrCreateArray(RetainValuesI);
  TempImportedModules->replaceAllUsesWith(IMs.get());

  // Now that all temp nodes have been replaced or deleted, resolve remaining
  // cycles.
  for (const auto &N : UnresolvedNodes)
    if (N && !N->isResolved())
      N->resolveCycles();
  UnresolvedNodes.clear();

  // Can't handle unresolved nodes anymore.
  AllowUnresolvedNodes = false;
}

/// If N is compile unit return NULL otherwise return N.
static MDScope *getNonCompileUnitScope(MDNode *N) {
  if (!N || isa<MDCompileUnit>(N))
    return nullptr;
  return cast<MDScope>(N);
}

DICompileUnit DIBuilder::createCompileUnit(unsigned Lang, StringRef Filename,
                                           StringRef Directory,
                                           StringRef Producer, bool isOptimized,
                                           StringRef Flags, unsigned RunTimeVer,
                                           StringRef SplitName,
                                           DebugEmissionKind Kind,
                                           bool EmitDebugInfo) {

  assert(((Lang <= dwarf::DW_LANG_Fortran08 && Lang >= dwarf::DW_LANG_C89) ||
          (Lang <= dwarf::DW_LANG_hi_user && Lang >= dwarf::DW_LANG_lo_user)) &&
         "Invalid Language tag");
  assert(!Filename.empty() &&
         "Unable to create compile unit without filename");

  // TODO: Once we make MDCompileUnit distinct, stop using temporaries here
  // (just start with operands assigned to nullptr).
  TempEnumTypes = MDTuple::getTemporary(VMContext, None);
  TempRetainTypes = MDTuple::getTemporary(VMContext, None);
  TempSubprograms = MDTuple::getTemporary(VMContext, None);
  TempGVs = MDTuple::getTemporary(VMContext, None);
  TempImportedModules = MDTuple::getTemporary(VMContext, None);

  // TODO: Switch to getDistinct().  We never want to merge compile units based
  // on contents.
  MDCompileUnit *CUNode = MDCompileUnit::get(
      VMContext, Lang, MDFile::get(VMContext, Filename, Directory), Producer,
      isOptimized, Flags, RunTimeVer, SplitName, Kind, TempEnumTypes.get(),
      TempRetainTypes.get(), TempSubprograms.get(), TempGVs.get(),
      TempImportedModules.get());

  // Create a named metadata so that it is easier to find cu in a module.
  // Note that we only generate this when the caller wants to actually
  // emit debug information. When we are only interested in tracking
  // source line locations throughout the backend, we prevent codegen from
  // emitting debug info in the final output by not generating llvm.dbg.cu.
  if (EmitDebugInfo) {
    NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
    NMD->addOperand(CUNode);
  }

  trackIfUnresolved(CUNode);
  return CUNode;
}

static DIImportedEntity
createImportedModule(LLVMContext &C, dwarf::Tag Tag, DIScope Context,
                     Metadata *NS, unsigned Line, StringRef Name,
                     SmallVectorImpl<TrackingMDNodeRef> &AllImportedModules) {
  DIImportedEntity M =
      MDImportedEntity::get(C, Tag, Context, DebugNodeRef(NS), Line, Name);
  AllImportedModules.emplace_back(M.get());
  return M;
}

DIImportedEntity DIBuilder::createImportedModule(DIScope Context,
                                                 DINameSpace NS,
                                                 unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, NS, Line, StringRef(), AllImportedModules);
}

DIImportedEntity DIBuilder::createImportedModule(DIScope Context,
                                                 DIImportedEntity NS,
                                                 unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, NS, Line, StringRef(), AllImportedModules);
}

DIImportedEntity DIBuilder::createImportedDeclaration(DIScope Context,
                                                      DIDescriptor Decl,
                                                      unsigned Line,
                                                      StringRef Name) {
  // Make sure to use the unique identifier based metadata reference for
  // types that have one.
  return ::createImportedModule(
      VMContext, dwarf::DW_TAG_imported_declaration, Context,
      DebugNodeRef::get(cast_or_null<DebugNode>(Decl.get())), Line, Name,
      AllImportedModules);
}

DIImportedEntity DIBuilder::createImportedDeclaration(DIScope Context,
                                                      DIImportedEntity Imp,
                                                      unsigned Line, StringRef Name) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_declaration,
                                Context, Imp, Line, Name, AllImportedModules);
}

DIFile DIBuilder::createFile(StringRef Filename, StringRef Directory) {
  return MDFile::get(VMContext, Filename, Directory);
}

DIEnumerator DIBuilder::createEnumerator(StringRef Name, int64_t Val) {
  assert(!Name.empty() && "Unable to create enumerator without name");
  return MDEnumerator::get(VMContext, Val, Name);
}

DIBasicType DIBuilder::createUnspecifiedType(StringRef Name) {
  assert(!Name.empty() && "Unable to create type without name");
  return MDBasicType::get(VMContext, dwarf::DW_TAG_unspecified_type, Name);
}

DIBasicType DIBuilder::createNullPtrType() {
  return createUnspecifiedType("decltype(nullptr)");
}

DIBasicType
DIBuilder::createBasicType(StringRef Name, uint64_t SizeInBits,
                           uint64_t AlignInBits, unsigned Encoding) {
  assert(!Name.empty() && "Unable to create type without name");
  return MDBasicType::get(VMContext, dwarf::DW_TAG_base_type, Name, SizeInBits,
                          AlignInBits, Encoding);
}

DIDerivedType DIBuilder::createQualifiedType(unsigned Tag, DIType FromTy) {
  return MDDerivedType::get(VMContext, Tag, "", nullptr, 0, nullptr,
                            MDTypeRef::get(FromTy), 0, 0, 0, 0);
}

DIDerivedType
DIBuilder::createPointerType(DIType PointeeTy, uint64_t SizeInBits,
                             uint64_t AlignInBits, StringRef Name) {
  // FIXME: Why is there a name here?
  return MDDerivedType::get(VMContext, dwarf::DW_TAG_pointer_type, Name,
                            nullptr, 0, nullptr, MDTypeRef::get(PointeeTy),
                            SizeInBits, AlignInBits, 0, 0);
}

DIDerivedType
DIBuilder::createMemberPointerType(DIType PointeeTy, DIType Base,
                                   uint64_t SizeInBits, uint64_t AlignInBits) {
  return MDDerivedType::get(VMContext, dwarf::DW_TAG_ptr_to_member_type, "",
                            nullptr, 0, nullptr, MDTypeRef::get(PointeeTy),
                            SizeInBits, AlignInBits, 0, 0, MDTypeRef::get(Base));
}

DIDerivedType DIBuilder::createReferenceType(unsigned Tag, DIType RTy) {
  assert(RTy && "Unable to create reference type");
  return MDDerivedType::get(VMContext, Tag, "", nullptr, 0, nullptr,
                            MDTypeRef::get(RTy), 0, 0, 0, 0);
}

DIDerivedType DIBuilder::createTypedef(DIType Ty, StringRef Name, DIFile File,
                                       unsigned LineNo, DIDescriptor Context) {
  return MDDerivedType::get(
      VMContext, dwarf::DW_TAG_typedef, Name, File, LineNo,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Context))),
      MDTypeRef::get(Ty), 0, 0, 0, 0);
}

DIDerivedType DIBuilder::createFriend(DIType Ty, DIType FriendTy) {
  // typedefs are encoded in DIDerivedType format.
  assert(Ty && "Invalid type!");
  assert(FriendTy && "Invalid friend type!");
  return MDDerivedType::get(VMContext, dwarf::DW_TAG_friend, "", nullptr, 0,
                            MDTypeRef::get(Ty), MDTypeRef::get(FriendTy), 0, 0,
                            0, 0);
}

DIDerivedType DIBuilder::createInheritance(DIType Ty, DIType BaseTy,
                                           uint64_t BaseOffset,
                                           unsigned Flags) {
  assert(Ty && "Unable to create inheritance");
  return MDDerivedType::get(VMContext, dwarf::DW_TAG_inheritance, "", nullptr,
                            0, MDTypeRef::get(Ty), MDTypeRef::get(BaseTy), 0, 0,
                            BaseOffset, Flags);
}

DIDerivedType DIBuilder::createMemberType(DIDescriptor Scope, StringRef Name,
                                          DIFile File, unsigned LineNumber,
                                          uint64_t SizeInBits,
                                          uint64_t AlignInBits,
                                          uint64_t OffsetInBits, unsigned Flags,
                                          DIType Ty) {
  return MDDerivedType::get(
      VMContext, dwarf::DW_TAG_member, Name, File, LineNumber,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Scope))),
      MDTypeRef::get(Ty), SizeInBits, AlignInBits, OffsetInBits, Flags);
}

static ConstantAsMetadata *getConstantOrNull(Constant *C) {
  if (C)
    return ConstantAsMetadata::get(C);
  return nullptr;
}

DIDerivedType DIBuilder::createStaticMemberType(DIDescriptor Scope,
                                                StringRef Name, DIFile File,
                                                unsigned LineNumber, DIType Ty,
                                                unsigned Flags,
                                                llvm::Constant *Val) {
  // TAG_member is encoded in DIDerivedType format.
  Flags |= DIDescriptor::FlagStaticMember;
  return MDDerivedType::get(
      VMContext, dwarf::DW_TAG_member, Name, File, LineNumber,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Scope))),
      MDTypeRef::get(Ty), 0, 0, 0, Flags, getConstantOrNull(Val));
}

DIDerivedType DIBuilder::createObjCIVar(StringRef Name, DIFile File,
                                        unsigned LineNumber,
                                        uint64_t SizeInBits,
                                        uint64_t AlignInBits,
                                        uint64_t OffsetInBits, unsigned Flags,
                                        DIType Ty, MDNode *PropertyNode) {
  return MDDerivedType::get(
      VMContext, dwarf::DW_TAG_member, Name, File, LineNumber,
      MDScopeRef::get(getNonCompileUnitScope(File)), MDTypeRef::get(Ty),
      SizeInBits, AlignInBits, OffsetInBits, Flags, PropertyNode);
}

DIObjCProperty
DIBuilder::createObjCProperty(StringRef Name, DIFile File, unsigned LineNumber,
                              StringRef GetterName, StringRef SetterName,
                              unsigned PropertyAttributes, DIType Ty) {
  return MDObjCProperty::get(VMContext, Name, File, LineNumber, GetterName,
                             SetterName, PropertyAttributes, Ty);
}

DITemplateTypeParameter
DIBuilder::createTemplateTypeParameter(DIDescriptor Context, StringRef Name,
                                       DIType Ty) {
  assert((!Context || isa<MDCompileUnit>(Context.get())) &&
         "Expected compile unit");
  return MDTemplateTypeParameter::get(VMContext, Name, MDTypeRef::get(Ty));
}

static DITemplateValueParameter
createTemplateValueParameterHelper(LLVMContext &VMContext, unsigned Tag,
                                   DIDescriptor Context, StringRef Name,
                                   DIType Ty, Metadata *MD) {
  assert((!Context || isa<MDCompileUnit>(Context.get())) &&
         "Expected compile unit");
  return MDTemplateValueParameter::get(VMContext, Tag, Name, MDTypeRef::get(Ty),
                                       MD);
}

DITemplateValueParameter
DIBuilder::createTemplateValueParameter(DIDescriptor Context, StringRef Name,
                                        DIType Ty, Constant *Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_template_value_parameter, Context, Name, Ty,
      getConstantOrNull(Val));
}

DITemplateValueParameter
DIBuilder::createTemplateTemplateParameter(DIDescriptor Context, StringRef Name,
                                           DIType Ty, StringRef Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_GNU_template_template_param, Context, Name, Ty,
      MDString::get(VMContext, Val));
}

DITemplateValueParameter
DIBuilder::createTemplateParameterPack(DIDescriptor Context, StringRef Name,
                                       DIType Ty, DIArray Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_GNU_template_parameter_pack, Context, Name, Ty,
      Val.get());
}

DICompositeType DIBuilder::createClassType(DIDescriptor Context, StringRef Name,
                                           DIFile File, unsigned LineNumber,
                                           uint64_t SizeInBits,
                                           uint64_t AlignInBits,
                                           uint64_t OffsetInBits,
                                           unsigned Flags, DIType DerivedFrom,
                                           DIArray Elements,
                                           DIType VTableHolder,
                                           MDNode *TemplateParams,
                                           StringRef UniqueIdentifier) {
  assert((!Context || isa<MDScope>(Context)) &&
         "createClassType should be called with a valid Context");
  // TAG_class_type is encoded in DICompositeType format.
  DICompositeType R = MDCompositeType::get(
      VMContext, dwarf::DW_TAG_structure_type, Name, File, LineNumber,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Context))),
      MDTypeRef::get(DerivedFrom), SizeInBits, AlignInBits, OffsetInBits, Flags,
      Elements, 0, MDTypeRef::get(VTableHolder),
      cast_or_null<MDTuple>(TemplateParams), UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(R);
  trackIfUnresolved(R);
  return R;
}

DICompositeType DIBuilder::createStructType(DIDescriptor Context,
                                            StringRef Name, DIFile File,
                                            unsigned LineNumber,
                                            uint64_t SizeInBits,
                                            uint64_t AlignInBits,
                                            unsigned Flags, DIType DerivedFrom,
                                            DIArray Elements,
                                            unsigned RunTimeLang,
                                            DIType VTableHolder,
                                            StringRef UniqueIdentifier) {
  DICompositeType R = MDCompositeType::get(
      VMContext, dwarf::DW_TAG_structure_type, Name, File, LineNumber,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Context))),
      MDTypeRef::get(DerivedFrom), SizeInBits, AlignInBits, 0, Flags, Elements,
      RunTimeLang, MDTypeRef::get(VTableHolder), nullptr, UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(R);
  trackIfUnresolved(R);
  return R;
}

DICompositeType DIBuilder::createUnionType(DIDescriptor Scope, StringRef Name,
                                           DIFile File, unsigned LineNumber,
                                           uint64_t SizeInBits,
                                           uint64_t AlignInBits, unsigned Flags,
                                           DIArray Elements,
                                           unsigned RunTimeLang,
                                           StringRef UniqueIdentifier) {
  DICompositeType R = MDCompositeType::get(
      VMContext, dwarf::DW_TAG_union_type, Name, File, LineNumber,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Scope))), nullptr,
      SizeInBits, AlignInBits, 0, Flags, Elements, RunTimeLang, nullptr,
      nullptr, UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(R);
  trackIfUnresolved(R);
  return R;
}

DISubroutineType DIBuilder::createSubroutineType(DIFile File,
                                                 DITypeArray ParameterTypes,
                                                 unsigned Flags) {
  return MDSubroutineType::get(VMContext, Flags, ParameterTypes);
}

DICompositeType DIBuilder::createEnumerationType(
    DIDescriptor Scope, StringRef Name, DIFile File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, DIArray Elements,
    DIType UnderlyingType, StringRef UniqueIdentifier) {
  DICompositeType CTy = MDCompositeType::get(
      VMContext, dwarf::DW_TAG_enumeration_type, Name, File, LineNumber,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Scope))),
      MDTypeRef::get(UnderlyingType), SizeInBits, AlignInBits, 0, 0, Elements,
      0, nullptr, nullptr, UniqueIdentifier);
  AllEnumTypes.push_back(CTy);
  if (!UniqueIdentifier.empty())
    retainType(CTy);
  trackIfUnresolved(CTy);
  return CTy;
}

DICompositeType DIBuilder::createArrayType(uint64_t Size, uint64_t AlignInBits,
                                           DIType Ty, DIArray Subscripts) {
  auto *R = MDCompositeType::get(VMContext, dwarf::DW_TAG_array_type, "",
                                 nullptr, 0, nullptr, MDTypeRef::get(Ty), Size,
                                 AlignInBits, 0, 0, Subscripts, 0, nullptr);
  trackIfUnresolved(R);
  return R;
}

DICompositeType DIBuilder::createVectorType(uint64_t Size, uint64_t AlignInBits,
                                            DIType Ty, DIArray Subscripts) {
  auto *R =
      MDCompositeType::get(VMContext, dwarf::DW_TAG_array_type, "", nullptr, 0,
                           nullptr, MDTypeRef::get(Ty), Size, AlignInBits, 0,
                           DIType::FlagVector, Subscripts, 0, nullptr);
  trackIfUnresolved(R);
  return R;
}

static DIType createTypeWithFlags(LLVMContext &Context, DIType Ty,
                                  unsigned FlagsToSet) {
  TempMDType NewTy = cast<MDType>(static_cast<MDNode *>(Ty))->clone();
  NewTy->setFlags(NewTy->getFlags() | FlagsToSet);
  return MDNode::replaceWithUniqued(std::move(NewTy));
}

DIType DIBuilder::createArtificialType(DIType Ty) {
  // FIXME: Restrict this to the nodes where it's valid.
  if (Ty.isArtificial())
    return Ty;
  return createTypeWithFlags(VMContext, Ty, DIType::FlagArtificial);
}

DIType DIBuilder::createObjectPointerType(DIType Ty) {
  // FIXME: Restrict this to the nodes where it's valid.
  if (Ty.isObjectPointer())
    return Ty;
  unsigned Flags = DIType::FlagObjectPointer | DIType::FlagArtificial;
  return createTypeWithFlags(VMContext, Ty, Flags);
}

void DIBuilder::retainType(DIType T) {
  assert(T.get() && "Expected non-null type");
  AllRetainTypes.emplace_back(T);
}

DIBasicType DIBuilder::createUnspecifiedParameter() {
  return DIBasicType();
}

DICompositeType
DIBuilder::createForwardDecl(unsigned Tag, StringRef Name, DIDescriptor Scope,
                             DIFile F, unsigned Line, unsigned RuntimeLang,
                             uint64_t SizeInBits, uint64_t AlignInBits,
                             StringRef UniqueIdentifier) {
  // FIXME: Define in terms of createReplaceableForwardDecl() by calling
  // replaceWithUniqued().
  DICompositeType RetTy = MDCompositeType::get(
      VMContext, Tag, Name, F, Line,
      MDScopeRef::get(DIScope(getNonCompileUnitScope(Scope))), nullptr,
      SizeInBits, AlignInBits, 0, DIDescriptor::FlagFwdDecl, nullptr,
      RuntimeLang, nullptr, nullptr, UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(RetTy);
  trackIfUnresolved(RetTy);
  return RetTy;
}

DICompositeType DIBuilder::createReplaceableCompositeType(
    unsigned Tag, StringRef Name, DIDescriptor Scope, DIFile F, unsigned Line,
    unsigned RuntimeLang, uint64_t SizeInBits, uint64_t AlignInBits,
    unsigned Flags, StringRef UniqueIdentifier) {
  DICompositeType RetTy =
      MDCompositeType::getTemporary(
          VMContext, Tag, Name, F, Line,
          MDScopeRef::get(DIScope(getNonCompileUnitScope(Scope))), nullptr,
          SizeInBits, AlignInBits, 0, Flags, nullptr, RuntimeLang, nullptr,
          nullptr, UniqueIdentifier)
          .release();
  if (!UniqueIdentifier.empty())
    retainType(RetTy);
  trackIfUnresolved(RetTy);
  return RetTy;
}

DIArray DIBuilder::getOrCreateArray(ArrayRef<Metadata *> Elements) {
  return DIArray(MDNode::get(VMContext, Elements));
}

DITypeArray DIBuilder::getOrCreateTypeArray(ArrayRef<Metadata *> Elements) {
  SmallVector<llvm::Metadata *, 16> Elts;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] && isa<MDNode>(Elements[i]))
      Elts.push_back(MDTypeRef::get(cast<MDType>(Elements[i])));
    else
      Elts.push_back(Elements[i]);
  }
  return DITypeArray(MDNode::get(VMContext, Elts));
}

DISubrange DIBuilder::getOrCreateSubrange(int64_t Lo, int64_t Count) {
  return MDSubrange::get(VMContext, Count, Lo);
}

static void checkGlobalVariableScope(DIDescriptor Context) {
#ifndef NDEBUG
  if (DICompositeType CT =
          dyn_cast_or_null<MDCompositeType>(getNonCompileUnitScope(Context)))
    assert(!CT.getIdentifier() &&
           "Context of a global variable should not be a type with identifier");
#endif
}

DIGlobalVariable DIBuilder::createGlobalVariable(
    DIDescriptor Context, StringRef Name, StringRef LinkageName, DIFile F,
    unsigned LineNumber, DIType Ty, bool isLocalToUnit, Constant *Val,
    MDNode *Decl) {
  checkGlobalVariableScope(Context);

  auto *N = MDGlobalVariable::get(
      VMContext, cast_or_null<MDScope>(Context.get()), Name, LinkageName, F,
      LineNumber, MDTypeRef::get(Ty), isLocalToUnit, true,
      getConstantOrNull(Val), cast_or_null<MDDerivedType>(Decl));
  AllGVs.push_back(N);
  return N;
}

DIGlobalVariable DIBuilder::createTempGlobalVariableFwdDecl(
    DIDescriptor Context, StringRef Name, StringRef LinkageName, DIFile F,
    unsigned LineNumber, DIType Ty, bool isLocalToUnit, Constant *Val,
    MDNode *Decl) {
  checkGlobalVariableScope(Context);

  return MDGlobalVariable::getTemporary(
             VMContext, cast_or_null<MDScope>(Context.get()), Name, LinkageName,
             F, LineNumber, MDTypeRef::get(Ty), isLocalToUnit, false, getConstantOrNull(Val),
             cast_or_null<MDDerivedType>(Decl)).release();
}

DIVariable DIBuilder::createLocalVariable(unsigned Tag, DIDescriptor Scope,
                                          StringRef Name, DIFile File,
                                          unsigned LineNo, DIType Ty,
                                          bool AlwaysPreserve, unsigned Flags,
                                          unsigned ArgNo) {
  // FIXME: Why getNonCompileUnitScope()?
  // FIXME: Why is "!Context" okay here?
  // FIXME: WHy doesn't this check for a subprogram or lexical block (AFAICT
  // the only valid scopes)?
  DIScope Context = getNonCompileUnitScope(Scope);

  auto *Node = MDLocalVariable::get(
      VMContext, Tag, cast_or_null<MDLocalScope>(Context.get()), Name, File,
      LineNo, MDTypeRef::get(Ty), ArgNo, Flags);
  if (AlwaysPreserve) {
    // The optimizer may remove local variable. If there is an interest
    // to preserve variable info in such situation then stash it in a
    // named mdnode.
    DISubprogram Fn(getDISubprogram(Scope));
    assert(Fn && "Missing subprogram for local variable");
    PreservedVariables[Fn].emplace_back(Node);
  }
  return Node;
}

DIExpression DIBuilder::createExpression(ArrayRef<uint64_t> Addr) {
  return MDExpression::get(VMContext, Addr);
}

DIExpression DIBuilder::createExpression(ArrayRef<int64_t> Signed) {
  // TODO: Remove the callers of this signed version and delete.
  SmallVector<uint64_t, 8> Addr(Signed.begin(), Signed.end());
  return createExpression(Addr);
}

DIExpression DIBuilder::createBitPieceExpression(unsigned OffsetInBytes,
                                                 unsigned SizeInBytes) {
  uint64_t Addr[] = {dwarf::DW_OP_bit_piece, OffsetInBytes, SizeInBytes};
  return MDExpression::get(VMContext, Addr);
}

DISubprogram DIBuilder::createFunction(DIScopeRef Context, StringRef Name,
                                       StringRef LinkageName, DIFile File,
                                       unsigned LineNo, DICompositeType Ty,
                                       bool isLocalToUnit, bool isDefinition,
                                       unsigned ScopeLine, unsigned Flags,
                                       bool isOptimized, Function *Fn,
                                       MDNode *TParams, MDNode *Decl) {
  // dragonegg does not generate identifier for types, so using an empty map
  // to resolve the context should be fine.
  DITypeIdentifierMap EmptyMap;
  return createFunction(Context.resolve(EmptyMap), Name, LinkageName, File,
                        LineNo, Ty, isLocalToUnit, isDefinition, ScopeLine,
                        Flags, isOptimized, Fn, TParams, Decl);
}

DISubprogram DIBuilder::createFunction(DIDescriptor Context, StringRef Name,
                                       StringRef LinkageName, DIFile File,
                                       unsigned LineNo, DICompositeType Ty,
                                       bool isLocalToUnit, bool isDefinition,
                                       unsigned ScopeLine, unsigned Flags,
                                       bool isOptimized, Function *Fn,
                                       MDNode *TParams, MDNode *Decl) {
  assert(Ty.getTag() == dwarf::DW_TAG_subroutine_type &&
         "function types should be subroutines");
  auto *Node = MDSubprogram::get(
      VMContext, MDScopeRef::get(DIScope(getNonCompileUnitScope(Context))),
      Name, LinkageName, File.get(), LineNo,
      cast_or_null<MDSubroutineType>(Ty.get()), isLocalToUnit, isDefinition,
      ScopeLine, nullptr, 0, 0, Flags, isOptimized, getConstantOrNull(Fn),
      cast_or_null<MDTuple>(TParams), cast_or_null<MDSubprogram>(Decl),
      MDTuple::getTemporary(VMContext, None).release());

  if (isDefinition)
    AllSubprograms.push_back(Node);
  trackIfUnresolved(Node);
  return Node;
}

DISubprogram
DIBuilder::createTempFunctionFwdDecl(DIDescriptor Context, StringRef Name,
                                     StringRef LinkageName, DIFile File,
                                     unsigned LineNo, DICompositeType Ty,
                                     bool isLocalToUnit, bool isDefinition,
                                     unsigned ScopeLine, unsigned Flags,
                                     bool isOptimized, Function *Fn,
                                     MDNode *TParams, MDNode *Decl) {
  return MDSubprogram::getTemporary(
             VMContext,
             MDScopeRef::get(DIScope(getNonCompileUnitScope(Context))), Name,
             LinkageName, File.get(), LineNo,
             cast_or_null<MDSubroutineType>(Ty.get()), isLocalToUnit,
             isDefinition, ScopeLine, nullptr, 0, 0, Flags, isOptimized,
             getConstantOrNull(Fn), cast_or_null<MDTuple>(TParams),
             cast_or_null<MDSubprogram>(Decl), nullptr)
      .release();
}

DISubprogram DIBuilder::createMethod(DIDescriptor Context, StringRef Name,
                                     StringRef LinkageName, DIFile F,
                                     unsigned LineNo, DICompositeType Ty,
                                     bool isLocalToUnit, bool isDefinition,
                                     unsigned VK, unsigned VIndex,
                                     DIType VTableHolder, unsigned Flags,
                                     bool isOptimized, Function *Fn,
                                     MDNode *TParam) {
  assert(Ty.getTag() == dwarf::DW_TAG_subroutine_type &&
         "function types should be subroutines");
  assert(getNonCompileUnitScope(Context) &&
         "Methods should have both a Context and a context that isn't "
         "the compile unit.");
  // FIXME: Do we want to use different scope/lines?
  auto *SP = MDSubprogram::get(
      VMContext, MDScopeRef::get(cast<MDScope>(Context)), Name, LinkageName,
      F.get(), LineNo, cast_or_null<MDSubroutineType>(Ty.get()), isLocalToUnit,
      isDefinition, LineNo, MDTypeRef::get(VTableHolder), VK, VIndex, Flags,
      isOptimized, getConstantOrNull(Fn), cast_or_null<MDTuple>(TParam),
      nullptr, nullptr);

  if (isDefinition)
    AllSubprograms.push_back(SP);
  trackIfUnresolved(SP);
  return SP;
}

DINameSpace DIBuilder::createNameSpace(DIDescriptor Scope, StringRef Name,
                                       DIFile File, unsigned LineNo) {
  return MDNamespace::get(VMContext, getNonCompileUnitScope(Scope), File, Name,
                          LineNo);
}

DILexicalBlockFile DIBuilder::createLexicalBlockFile(DIDescriptor Scope,
                                                     DIFile File,
                                                     unsigned Discriminator) {
  return MDLexicalBlockFile::get(VMContext, Scope, File.getFileNode(),
                                 Discriminator);
}

DILexicalBlock DIBuilder::createLexicalBlock(DIDescriptor Scope, DIFile File,
                                             unsigned Line, unsigned Col) {
  // Make these distinct, to avoid merging two lexical blocks on the same
  // file/line/column.
  return MDLexicalBlock::getDistinct(VMContext, getNonCompileUnitScope(Scope),
                                     File.getFileNode(), Line, Col);
}

static Value *getDbgIntrinsicValueImpl(LLVMContext &VMContext, Value *V) {
  assert(V && "no value passed to dbg intrinsic");
  return MetadataAsValue::get(VMContext, ValueAsMetadata::get(V));
}

Instruction *DIBuilder::insertDeclare(Value *Storage, DIVariable VarInfo,
                                      DIExpression Expr,
                                      Instruction *InsertBefore) {
  assert(VarInfo && "empty or invalid DIVariable passed to dbg.declare");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, Storage),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};
  return CallInst::Create(DeclareFn, Args, "", InsertBefore);
}

Instruction *DIBuilder::insertDeclare(Value *Storage, DIVariable VarInfo,
                                      DIExpression Expr,
                                      BasicBlock *InsertAtEnd) {
  assert(VarInfo && "empty or invalid DIVariable passed to dbg.declare");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, Storage),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};

  // If this block already has a terminator then insert this intrinsic
  // before the terminator.
  if (TerminatorInst *T = InsertAtEnd->getTerminator())
    return CallInst::Create(DeclareFn, Args, "", T);
  else
    return CallInst::Create(DeclareFn, Args, "", InsertAtEnd);
}

Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DIVariable VarInfo,
                                                DIExpression Expr,
                                                Instruction *InsertBefore) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo && "empty or invalid DIVariable passed to dbg.value");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, V),
                   ConstantInt::get(Type::getInt64Ty(VMContext), Offset),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};
  return CallInst::Create(ValueFn, Args, "", InsertBefore);
}

Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DIVariable VarInfo,
                                                DIExpression Expr,
                                                BasicBlock *InsertAtEnd) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo && "empty or invalid DIVariable passed to dbg.value");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, V),
                   ConstantInt::get(Type::getInt64Ty(VMContext), Offset),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};
  return CallInst::Create(ValueFn, Args, "", InsertAtEnd);
}

void DIBuilder::replaceVTableHolder(DICompositeType &T, DICompositeType VTableHolder) {
  {
    TypedTrackingMDRef<MDCompositeTypeBase> N(T);
    N->replaceVTableHolder(MDTypeRef::get(VTableHolder));
    T = N.get();
  }

  // If this didn't create a self-reference, just return.
  if (T != VTableHolder)
    return;

  // Look for unresolved operands.  T will drop RAUW support, orphaning any
  // cycles underneath it.
  if (T->isResolved())
    for (const MDOperand &O : T->operands())
      if (auto *N = dyn_cast_or_null<MDNode>(O))
        trackIfUnresolved(N);
}

void DIBuilder::replaceArrays(DICompositeType &T, DIArray Elements,
                              DIArray TParams) {
  {
    TypedTrackingMDRef<MDCompositeTypeBase> N(T);
    if (Elements)
      N->replaceElements(Elements);
    if (TParams)
      N->replaceTemplateParams(MDTemplateParameterArray(TParams));
    T = N.get();
  }

  // If T isn't resolved, there's no problem.
  if (!T->isResolved())
    return;

  // If "T" is resolved, it may be due to a self-reference cycle.  Track the
  // arrays explicitly if they're unresolved, or else the cycles will be
  // orphaned.
  if (Elements)
    trackIfUnresolved(Elements.get());
  if (TParams)
    trackIfUnresolved(TParams.get());
}
