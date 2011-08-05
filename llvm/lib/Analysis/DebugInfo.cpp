//===--- DebugInfo.cpp - Debug Information Helper Classes -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the helper classes used to build and interpret debug
// information in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::dwarf;

//===----------------------------------------------------------------------===//
// DIDescriptor
//===----------------------------------------------------------------------===//

DIDescriptor::DIDescriptor(const DIFile F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DISubprogram F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DILexicalBlock F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DIVariable F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DIType F) : DbgNode(F.DbgNode) {
}

StringRef
DIDescriptor::getStringField(unsigned Elt) const {
  if (DbgNode == 0)
    return StringRef();

  if (Elt < DbgNode->getNumOperands())
    if (MDString *MDS = dyn_cast_or_null<MDString>(DbgNode->getOperand(Elt)))
      return MDS->getString();

  return StringRef();
}

uint64_t DIDescriptor::getUInt64Field(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
    if (ConstantInt *CI = dyn_cast<ConstantInt>(DbgNode->getOperand(Elt)))
      return CI->getZExtValue();

  return 0;
}

DIDescriptor DIDescriptor::getDescriptorField(unsigned Elt) const {
  if (DbgNode == 0)
    return DIDescriptor();

  if (Elt < DbgNode->getNumOperands())
    return
      DIDescriptor(dyn_cast_or_null<const MDNode>(DbgNode->getOperand(Elt)));
  return DIDescriptor();
}

GlobalVariable *DIDescriptor::getGlobalVariableField(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
      return dyn_cast_or_null<GlobalVariable>(DbgNode->getOperand(Elt));
  return 0;
}

Constant *DIDescriptor::getConstantField(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
      return dyn_cast_or_null<Constant>(DbgNode->getOperand(Elt));
  return 0;
}

Function *DIDescriptor::getFunctionField(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
      return dyn_cast_or_null<Function>(DbgNode->getOperand(Elt));
  return 0;
}

unsigned DIVariable::getNumAddrElements() const {
  if (getVersion() <= llvm::LLVMDebugVersion8)
    return DbgNode->getNumOperands()-6;
  if (getVersion() == llvm::LLVMDebugVersion9)
    return DbgNode->getNumOperands()-7;
  return DbgNode->getNumOperands()-8;
}

/// getInlinedAt - If this variable is inlined then return inline location.
MDNode *DIVariable::getInlinedAt() {
  if (getVersion() <= llvm::LLVMDebugVersion9)
    return NULL;
  return dyn_cast_or_null<MDNode>(DbgNode->getOperand(7));
}

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// isBasicType - Return true if the specified tag is legal for
/// DIBasicType.
bool DIDescriptor::isBasicType() const {
  return DbgNode && getTag() == dwarf::DW_TAG_base_type;
}

/// isDerivedType - Return true if the specified tag is legal for DIDerivedType.
bool DIDescriptor::isDerivedType() const {
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_pointer_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_member:
  case dwarf::DW_TAG_inheritance:
  case dwarf::DW_TAG_friend:
    return true;
  default:
    // CompositeTypes are currently modelled as DerivedTypes.
    return isCompositeType();
  }
}

/// isCompositeType - Return true if the specified tag is legal for
/// DICompositeType.
bool DIDescriptor::isCompositeType() const {
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_vector_type:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_class_type:
    return true;
  default:
    return false;
  }
}

/// isVariable - Return true if the specified tag is legal for DIVariable.
bool DIDescriptor::isVariable() const {
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_auto_variable:
  case dwarf::DW_TAG_arg_variable:
  case dwarf::DW_TAG_return_variable:
    return true;
  default:
    return false;
  }
}

/// isType - Return true if the specified tag is legal for DIType.
bool DIDescriptor::isType() const {
  return isBasicType() || isCompositeType() || isDerivedType();
}

/// isSubprogram - Return true if the specified tag is legal for
/// DISubprogram.
bool DIDescriptor::isSubprogram() const {
  return DbgNode && getTag() == dwarf::DW_TAG_subprogram;
}

/// isGlobalVariable - Return true if the specified tag is legal for
/// DIGlobalVariable.
bool DIDescriptor::isGlobalVariable() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_variable ||
                     getTag() == dwarf::DW_TAG_constant);
}

/// isGlobal - Return true if the specified tag is legal for DIGlobal.
bool DIDescriptor::isGlobal() const {
  return isGlobalVariable();
}

/// isUnspecifiedParmeter - Return true if the specified tag is
/// DW_TAG_unspecified_parameters.
bool DIDescriptor::isUnspecifiedParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_unspecified_parameters;
}

/// isScope - Return true if the specified tag is one of the scope
/// related tag.
bool DIDescriptor::isScope() const {
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_compile_unit:
  case dwarf::DW_TAG_lexical_block:
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_namespace:
    return true;
  default:
    break;
  }
  return false;
}

/// isTemplateTypeParameter - Return true if the specified tag is
/// DW_TAG_template_type_parameter.
bool DIDescriptor::isTemplateTypeParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_template_type_parameter;
}

/// isTemplateValueParameter - Return true if the specified tag is
/// DW_TAG_template_value_parameter.
bool DIDescriptor::isTemplateValueParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_template_value_parameter;
}

/// isCompileUnit - Return true if the specified tag is DW_TAG_compile_unit.
bool DIDescriptor::isCompileUnit() const {
  return DbgNode && getTag() == dwarf::DW_TAG_compile_unit;
}

/// isFile - Return true if the specified tag is DW_TAG_file_type.
bool DIDescriptor::isFile() const {
  return DbgNode && getTag() == dwarf::DW_TAG_file_type;
}

/// isNameSpace - Return true if the specified tag is DW_TAG_namespace.
bool DIDescriptor::isNameSpace() const {
  return DbgNode && getTag() == dwarf::DW_TAG_namespace;
}

/// isLexicalBlock - Return true if the specified tag is DW_TAG_lexical_block.
bool DIDescriptor::isLexicalBlock() const {
  return DbgNode && getTag() == dwarf::DW_TAG_lexical_block;
}

/// isSubrange - Return true if the specified tag is DW_TAG_subrange_type.
bool DIDescriptor::isSubrange() const {
  return DbgNode && getTag() == dwarf::DW_TAG_subrange_type;
}

/// isEnumerator - Return true if the specified tag is DW_TAG_enumerator.
bool DIDescriptor::isEnumerator() const {
  return DbgNode && getTag() == dwarf::DW_TAG_enumerator;
}

//===----------------------------------------------------------------------===//
// Simple Descriptor Constructors and other Methods
//===----------------------------------------------------------------------===//

DIType::DIType(const MDNode *N) : DIScope(N) {
  if (!N) return;
  if (!isBasicType() && !isDerivedType() && !isCompositeType()) {
    DbgNode = 0;
  }
}

unsigned DIArray::getNumElements() const {
  if (!DbgNode)
    return 0;
  return DbgNode->getNumOperands();
}

/// replaceAllUsesWith - Replace all uses of debug info referenced by
/// this descriptor.
void DIType::replaceAllUsesWith(DIDescriptor &D) {
  if (!DbgNode)
    return;

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  if (DbgNode != D) {
    MDNode *Node = const_cast<MDNode*>(DbgNode);
    const MDNode *DN = D;
    const Value *V = cast_or_null<Value>(DN);
    Node->replaceAllUsesWith(const_cast<Value*>(V));
    MDNode::deleteTemporary(Node);
  }
}

/// replaceAllUsesWith - Replace all uses of debug info referenced by
/// this descriptor.
void DIType::replaceAllUsesWith(MDNode *D) {
  if (!DbgNode)
    return;

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  if (DbgNode != D) {
    MDNode *Node = const_cast<MDNode*>(DbgNode);
    const MDNode *DN = D;
    const Value *V = cast_or_null<Value>(DN);
    Node->replaceAllUsesWith(const_cast<Value*>(V));
    MDNode::deleteTemporary(Node);
  }
}

/// Verify - Verify that a compile unit is well formed.
bool DICompileUnit::Verify() const {
  if (!DbgNode)
    return false;
  StringRef N = getFilename();
  if (N.empty())
    return false;
  // It is possible that directory and produce string is empty.
  return true;
}

/// Verify - Verify that a type descriptor is well formed.
bool DIType::Verify() const {
  if (!DbgNode)
    return false;
  if (!getContext().Verify())
    return false;
  unsigned Tag = getTag();
  if (!isBasicType() && Tag != dwarf::DW_TAG_const_type &&
      Tag != dwarf::DW_TAG_volatile_type && Tag != dwarf::DW_TAG_pointer_type &&
      Tag != dwarf::DW_TAG_reference_type && Tag != dwarf::DW_TAG_restrict_type 
      && Tag != dwarf::DW_TAG_vector_type && Tag != dwarf::DW_TAG_array_type
      && Tag != dwarf::DW_TAG_enumeration_type 
      && getFilename().empty())
    return false;
  return true;
}

/// Verify - Verify that a basic type descriptor is well formed.
bool DIBasicType::Verify() const {
  return isBasicType();
}

/// Verify - Verify that a derived type descriptor is well formed.
bool DIDerivedType::Verify() const {
  return isDerivedType();
}

/// Verify - Verify that a composite type descriptor is well formed.
bool DICompositeType::Verify() const {
  if (!DbgNode)
    return false;
  if (!getContext().Verify())
    return false;

  DICompileUnit CU = getCompileUnit();
  if (!CU.Verify())
    return false;
  return true;
}

/// Verify - Verify that a subprogram descriptor is well formed.
bool DISubprogram::Verify() const {
  if (!DbgNode)
    return false;

  if (!getContext().Verify())
    return false;

  DICompileUnit CU = getCompileUnit();
  if (!CU.Verify())
    return false;

  DICompositeType Ty = getType();
  if (!Ty.Verify())
    return false;
  return true;
}

/// Verify - Verify that a global variable descriptor is well formed.
bool DIGlobalVariable::Verify() const {
  if (!DbgNode)
    return false;

  if (getDisplayName().empty())
    return false;

  if (!getContext().Verify())
    return false;

  DICompileUnit CU = getCompileUnit();
  if (!CU.Verify())
    return false;

  DIType Ty = getType();
  if (!Ty.Verify())
    return false;

  if (!getGlobal() && !getConstant())
    return false;

  return true;
}

/// Verify - Verify that a variable descriptor is well formed.
bool DIVariable::Verify() const {
  if (!DbgNode)
    return false;

  if (!getContext().Verify())
    return false;

  if (!getCompileUnit().Verify())
    return false;

  DIType Ty = getType();
  if (!Ty.Verify())
    return false;

  return true;
}

/// Verify - Verify that a location descriptor is well formed.
bool DILocation::Verify() const {
  if (!DbgNode)
    return false;

  return DbgNode->getNumOperands() == 4;
}

/// Verify - Verify that a namespace descriptor is well formed.
bool DINameSpace::Verify() const {
  if (!DbgNode)
    return false;
  if (getName().empty())
    return false;
  if (!getCompileUnit().Verify())
    return false;
  return true;
}

/// getOriginalTypeSize - If this type is derived from a base type then
/// return base type size.
uint64_t DIDerivedType::getOriginalTypeSize() const {
  unsigned Tag = getTag();
  if (Tag == dwarf::DW_TAG_member || Tag == dwarf::DW_TAG_typedef ||
      Tag == dwarf::DW_TAG_const_type || Tag == dwarf::DW_TAG_volatile_type ||
      Tag == dwarf::DW_TAG_restrict_type) {
    DIType BaseType = getTypeDerivedFrom();
    // If this type is not derived from any type then take conservative
    // approach.
    if (!BaseType.isValid())
      return getSizeInBits();
    if (BaseType.isDerivedType())
      return DIDerivedType(BaseType).getOriginalTypeSize();
    else
      return BaseType.getSizeInBits();
  }

  return getSizeInBits();
}

/// isInlinedFnArgument - Return true if this variable provides debugging
/// information for an inlined function arguments.
bool DIVariable::isInlinedFnArgument(const Function *CurFn) {
  assert(CurFn && "Invalid function");
  if (!getContext().isSubprogram())
    return false;
  // This variable is not inlined function argument if its scope
  // does not describe current function.
  return !(DISubprogram(getContext()).describes(CurFn));
}

/// describes - Return true if this subprogram provides debugging
/// information for the function F.
bool DISubprogram::describes(const Function *F) {
  assert(F && "Invalid function");
  if (F == getFunction())
    return true;
  StringRef Name = getLinkageName();
  if (Name.empty())
    Name = getName();
  if (F->getName() == Name)
    return true;
  return false;
}

unsigned DISubprogram::isOptimized() const {
  assert (DbgNode && "Invalid subprogram descriptor!");
  if (DbgNode->getNumOperands() == 16)
    return getUnsignedField(15);
  return 0;
}

StringRef DIScope::getFilename() const {
  if (!DbgNode)
    return StringRef();
  if (isLexicalBlock())
    return DILexicalBlock(DbgNode).getFilename();
  if (isSubprogram())
    return DISubprogram(DbgNode).getFilename();
  if (isCompileUnit())
    return DICompileUnit(DbgNode).getFilename();
  if (isNameSpace())
    return DINameSpace(DbgNode).getFilename();
  if (isType())
    return DIType(DbgNode).getFilename();
  if (isFile())
    return DIFile(DbgNode).getFilename();
  assert(0 && "Invalid DIScope!");
  return StringRef();
}

StringRef DIScope::getDirectory() const {
  if (!DbgNode)
    return StringRef();
  if (isLexicalBlock())
    return DILexicalBlock(DbgNode).getDirectory();
  if (isSubprogram())
    return DISubprogram(DbgNode).getDirectory();
  if (isCompileUnit())
    return DICompileUnit(DbgNode).getDirectory();
  if (isNameSpace())
    return DINameSpace(DbgNode).getDirectory();
  if (isType())
    return DIType(DbgNode).getDirectory();
  if (isFile())
    return DIFile(DbgNode).getDirectory();
  assert(0 && "Invalid DIScope!");
  return StringRef();
}

//===----------------------------------------------------------------------===//
// DIDescriptor: dump routines for all descriptors.
//===----------------------------------------------------------------------===//


/// print - Print descriptor.
void DIDescriptor::print(raw_ostream &OS) const {
  OS << "[" << dwarf::TagString(getTag()) << "] ";
  OS.write_hex((intptr_t) &*DbgNode) << ']';
}

/// print - Print compile unit.
void DICompileUnit::print(raw_ostream &OS) const {
  if (getLanguage())
    OS << " [" << dwarf::LanguageString(getLanguage()) << "] ";

  OS << " [" << getDirectory() << "/" << getFilename() << "]";
}

/// print - Print type.
void DIType::print(raw_ostream &OS) const {
  if (!DbgNode) return;

  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << "] ";

  unsigned Tag = getTag();
  OS << " [" << dwarf::TagString(Tag) << "] ";

  // TODO : Print context
  getCompileUnit().print(OS);
  OS << " ["
         << "line " << getLineNumber() << ", "
         << getSizeInBits() << " bits, "
         << getAlignInBits() << " bit alignment, "
         << getOffsetInBits() << " bit offset"
         << "] ";

  if (isPrivate())
    OS << " [private] ";
  else if (isProtected())
    OS << " [protected] ";

  if (isForwardDecl())
    OS << " [fwd] ";

  if (isBasicType())
    DIBasicType(DbgNode).print(OS);
  else if (isDerivedType())
    DIDerivedType(DbgNode).print(OS);
  else if (isCompositeType())
    DICompositeType(DbgNode).print(OS);
  else {
    OS << "Invalid DIType\n";
    return;
  }

  OS << "\n";
}

/// print - Print basic type.
void DIBasicType::print(raw_ostream &OS) const {
  OS << " [" << dwarf::AttributeEncodingString(getEncoding()) << "] ";
}

/// print - Print derived type.
void DIDerivedType::print(raw_ostream &OS) const {
  OS << "\n\t Derived From: "; getTypeDerivedFrom().print(OS);
}

/// print - Print composite type.
void DICompositeType::print(raw_ostream &OS) const {
  DIArray A = getTypeArray();
  OS << " [" << A.getNumElements() << " elements]";
}

/// print - Print subprogram.
void DISubprogram::print(raw_ostream &OS) const {
  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << "] ";

  unsigned Tag = getTag();
  OS << " [" << dwarf::TagString(Tag) << "] ";

  // TODO : Print context
  getCompileUnit().print(OS);
  OS << " [" << getLineNumber() << "] ";

  if (isLocalToUnit())
    OS << " [local] ";

  if (isDefinition())
    OS << " [def] ";

  OS << "\n";
}

/// print - Print global variable.
void DIGlobalVariable::print(raw_ostream &OS) const {
  OS << " [";
  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << "] ";

  unsigned Tag = getTag();
  OS << " [" << dwarf::TagString(Tag) << "] ";

  // TODO : Print context
  getCompileUnit().print(OS);
  OS << " [" << getLineNumber() << "] ";

  if (isLocalToUnit())
    OS << " [local] ";

  if (isDefinition())
    OS << " [def] ";

  if (isGlobalVariable())
    DIGlobalVariable(DbgNode).print(OS);
  OS << "]\n";
}

/// print - Print variable.
void DIVariable::print(raw_ostream &OS) const {
  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << "] ";

  getCompileUnit().print(OS);
  OS << " [" << getLineNumber() << "] ";
  getType().print(OS);
  OS << "\n";

  // FIXME: Dump complex addresses
}

/// dump - Print descriptor to dbgs() with a newline.
void DIDescriptor::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print compile unit to dbgs() with a newline.
void DICompileUnit::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print type to dbgs() with a newline.
void DIType::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print basic type to dbgs() with a newline.
void DIBasicType::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print derived type to dbgs() with a newline.
void DIDerivedType::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print composite type to dbgs() with a newline.
void DICompositeType::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print subprogram to dbgs() with a newline.
void DISubprogram::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print global variable.
void DIGlobalVariable::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// dump - Print variable.
void DIVariable::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// fixupObjcLikeName - Replace contains special characters used
/// in a typical Objective-C names with '.' in a given string.
static void fixupObjcLikeName(StringRef Str, SmallVectorImpl<char> &Out) {
  bool isObjCLike = false;
  for (size_t i = 0, e = Str.size(); i < e; ++i) {
    char C = Str[i];
    if (C == '[')
      isObjCLike = true;

    if (isObjCLike && (C == '[' || C == ']' || C == ' ' || C == ':' ||
                       C == '+' || C == '(' || C == ')'))
      Out.push_back('.');
    else
      Out.push_back(C);
  }
}

/// getFnSpecificMDNode - Return a NameMDNode, if available, that is 
/// suitable to hold function specific information.
NamedMDNode *llvm::getFnSpecificMDNode(const Module &M, StringRef FuncName) {
  SmallString<32> Name = StringRef("llvm.dbg.lv.");
  fixupObjcLikeName(FuncName, Name);

  return M.getNamedMetadata(Name.str());
}

/// getOrInsertFnSpecificMDNode - Return a NameMDNode that is suitable
/// to hold function specific information.
NamedMDNode *llvm::getOrInsertFnSpecificMDNode(Module &M, StringRef FuncName) {
  SmallString<32> Name = StringRef("llvm.dbg.lv.");
  fixupObjcLikeName(FuncName, Name);

  return M.getOrInsertNamedMetadata(Name.str());
}

/// createInlinedVariable - Create a new inlined variable based on current
/// variable.
/// @param DV            Current Variable.
/// @param InlinedScope  Location at current variable is inlined.
DIVariable llvm::createInlinedVariable(MDNode *DV, MDNode *InlinedScope,
                                       LLVMContext &VMContext) {
  SmallVector<Value *, 16> Elts;
  // Insert inlined scope as 7th element.
  for (unsigned i = 0, e = DV->getNumOperands(); i != e; ++i)
    i == 7 ? Elts.push_back(InlinedScope) :
             Elts.push_back(DV->getOperand(i));
  return DIVariable(MDNode::get(VMContext, Elts));
}

//===----------------------------------------------------------------------===//
// DebugInfoFinder implementations.
//===----------------------------------------------------------------------===//

/// processModule - Process entire module and collect debug info.
void DebugInfoFinder::processModule(Module &M) {
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    for (Function::iterator FI = (*I).begin(), FE = (*I).end(); FI != FE; ++FI)
      for (BasicBlock::iterator BI = (*FI).begin(), BE = (*FI).end(); BI != BE;
           ++BI) {
        if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(BI))
          processDeclare(DDI);

        DebugLoc Loc = BI->getDebugLoc();
        if (Loc.isUnknown())
          continue;

        LLVMContext &Ctx = BI->getContext();
        DIDescriptor Scope(Loc.getScope(Ctx));

        if (Scope.isCompileUnit())
          addCompileUnit(DICompileUnit(Scope));
        else if (Scope.isSubprogram())
          processSubprogram(DISubprogram(Scope));
        else if (Scope.isLexicalBlock())
          processLexicalBlock(DILexicalBlock(Scope));

        if (MDNode *IA = Loc.getInlinedAt(Ctx))
          processLocation(DILocation(IA));
      }

  if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.gv")) {
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
      DIGlobalVariable DIG(cast<MDNode>(NMD->getOperand(i)));
      if (addGlobalVariable(DIG)) {
        addCompileUnit(DIG.getCompileUnit());
        processType(DIG.getType());
      }
    }
  }

  if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.sp"))
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      processSubprogram(DISubprogram(NMD->getOperand(i)));
}

/// processLocation - Process DILocation.
void DebugInfoFinder::processLocation(DILocation Loc) {
  if (!Loc.Verify()) return;
  DIDescriptor S(Loc.getScope());
  if (S.isCompileUnit())
    addCompileUnit(DICompileUnit(S));
  else if (S.isSubprogram())
    processSubprogram(DISubprogram(S));
  else if (S.isLexicalBlock())
    processLexicalBlock(DILexicalBlock(S));
  processLocation(Loc.getOrigLocation());
}

/// processType - Process DIType.
void DebugInfoFinder::processType(DIType DT) {
  if (!addType(DT))
    return;

  addCompileUnit(DT.getCompileUnit());
  if (DT.isCompositeType()) {
    DICompositeType DCT(DT);
    processType(DCT.getTypeDerivedFrom());
    DIArray DA = DCT.getTypeArray();
    for (unsigned i = 0, e = DA.getNumElements(); i != e; ++i) {
      DIDescriptor D = DA.getElement(i);
      if (D.isType())
        processType(DIType(D));
      else if (D.isSubprogram())
        processSubprogram(DISubprogram(D));
    }
  } else if (DT.isDerivedType()) {
    DIDerivedType DDT(DT);
    processType(DDT.getTypeDerivedFrom());
  }
}

/// processLexicalBlock
void DebugInfoFinder::processLexicalBlock(DILexicalBlock LB) {
  DIScope Context = LB.getContext();
  if (Context.isLexicalBlock())
    return processLexicalBlock(DILexicalBlock(Context));
  else
    return processSubprogram(DISubprogram(Context));
}

/// processSubprogram - Process DISubprogram.
void DebugInfoFinder::processSubprogram(DISubprogram SP) {
  if (!addSubprogram(SP))
    return;
  addCompileUnit(SP.getCompileUnit());
  processType(SP.getType());
}

/// processDeclare - Process DbgDeclareInst.
void DebugInfoFinder::processDeclare(DbgDeclareInst *DDI) {
  MDNode *N = dyn_cast<MDNode>(DDI->getVariable());
  if (!N) return;

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV))
    return;

  addCompileUnit(DIVariable(N).getCompileUnit());
  processType(DIVariable(N).getType());
}

/// addType - Add type into Tys.
bool DebugInfoFinder::addType(DIType DT) {
  if (!DT.isValid())
    return false;

  if (!NodesSeen.insert(DT))
    return false;

  TYs.push_back(DT);
  return true;
}

/// addCompileUnit - Add compile unit into CUs.
bool DebugInfoFinder::addCompileUnit(DICompileUnit CU) {
  if (!CU.Verify())
    return false;

  if (!NodesSeen.insert(CU))
    return false;

  CUs.push_back(CU);
  return true;
}

/// addGlobalVariable - Add global variable into GVs.
bool DebugInfoFinder::addGlobalVariable(DIGlobalVariable DIG) {
  if (!DIDescriptor(DIG).isGlobalVariable())
    return false;

  if (!NodesSeen.insert(DIG))
    return false;

  GVs.push_back(DIG);
  return true;
}

// addSubprogram - Add subprgoram into SPs.
bool DebugInfoFinder::addSubprogram(DISubprogram SP) {
  if (!DIDescriptor(SP).isSubprogram())
    return false;

  if (!NodesSeen.insert(SP))
    return false;

  SPs.push_back(SP);
  return true;
}

/// getDISubprogram - Find subprogram that is enclosing this scope.
DISubprogram llvm::getDISubprogram(const MDNode *Scope) {
  DIDescriptor D(Scope);
  if (D.isSubprogram())
    return DISubprogram(Scope);

  if (D.isLexicalBlock())
    return getDISubprogram(DILexicalBlock(Scope).getContext());

  return DISubprogram();
}

/// getDICompositeType - Find underlying composite type.
DICompositeType llvm::getDICompositeType(DIType T) {
  if (T.isCompositeType())
    return DICompositeType(T);

  if (T.isDerivedType())
    return getDICompositeType(DIDerivedType(T).getTypeDerivedFrom());

  return DICompositeType();
}
