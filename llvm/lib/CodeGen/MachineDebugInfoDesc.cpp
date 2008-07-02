//===-- llvm/CodeGen/MachineDebugInfoDesc.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDebugInfoDesc.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Streams.h"

using namespace llvm;
using namespace llvm::dwarf;

/// getUIntOperand - Return ith operand if it is an unsigned integer.
///
static ConstantInt *getUIntOperand(const GlobalVariable *GV, unsigned i) {
  // Make sure the GlobalVariable has an initializer.
  if (!GV->hasInitializer()) return NULL;
  
  // Get the initializer constant.
  ConstantStruct *CI = dyn_cast<ConstantStruct>(GV->getInitializer());
  if (!CI) return NULL;
  
  // Check if there is at least i + 1 operands.
  unsigned N = CI->getNumOperands();
  if (i >= N) return NULL;

  // Check constant.
  return dyn_cast<ConstantInt>(CI->getOperand(i));
}

//===----------------------------------------------------------------------===//

/// Supply a home for the DebugInfoDesc's v-table.
DebugInfoDesc::~DebugInfoDesc() {}

/// TagFromGlobal - Returns the tag number from a debug info descriptor
/// GlobalVariable. Return DIIValid if operand is not an unsigned int.
unsigned DebugInfoDesc::TagFromGlobal(GlobalVariable *GV) {
  ConstantInt *C = getUIntOperand(GV, 0);
  return C ? ((unsigned)C->getZExtValue() & ~LLVMDebugVersionMask) :
             (unsigned)DW_TAG_invalid;
}

/// VersionFromGlobal - Returns the version number from a debug info
/// descriptor GlobalVariable. Return DIIValid if operand is not an unsigned
/// int.
unsigned  DebugInfoDesc::VersionFromGlobal(GlobalVariable *GV) {
  ConstantInt *C = getUIntOperand(GV, 0);
  return C ? ((unsigned)C->getZExtValue() & LLVMDebugVersionMask) :
             (unsigned)DW_TAG_invalid;
}

/// DescFactory - Create an instance of debug info descriptor based on Tag.
/// Return NULL if not a recognized Tag.
DebugInfoDesc *DebugInfoDesc::DescFactory(unsigned Tag) {
  switch (Tag) {
  case DW_TAG_anchor:           return new AnchorDesc();
  case DW_TAG_compile_unit:     return new CompileUnitDesc();
  case DW_TAG_variable:         return new GlobalVariableDesc();
  case DW_TAG_subprogram:       return new SubprogramDesc();
  case DW_TAG_lexical_block:    return new BlockDesc();
  case DW_TAG_base_type:        return new BasicTypeDesc();
  case DW_TAG_typedef:
  case DW_TAG_pointer_type:        
  case DW_TAG_reference_type:
  case DW_TAG_const_type:
  case DW_TAG_volatile_type:        
  case DW_TAG_restrict_type:
  case DW_TAG_member:
  case DW_TAG_inheritance:      return new DerivedTypeDesc(Tag);
  case DW_TAG_array_type:
  case DW_TAG_structure_type:
  case DW_TAG_union_type:
  case DW_TAG_enumeration_type:
  case DW_TAG_vector_type:
  case DW_TAG_subroutine_type:  return new CompositeTypeDesc(Tag);
  case DW_TAG_subrange_type:    return new SubrangeDesc();
  case DW_TAG_enumerator:       return new EnumeratorDesc();
  case DW_TAG_return_variable:
  case DW_TAG_arg_variable:
  case DW_TAG_auto_variable:    return new VariableDesc(Tag);
  default: break;
  }
  return NULL;
}

/// getLinkage - get linkage appropriate for this type of descriptor.
GlobalValue::LinkageTypes DebugInfoDesc::getLinkage() const {
  return GlobalValue::InternalLinkage;
}

/// ApplyToFields - Target the vistor to the fields of the descriptor.
void DebugInfoDesc::ApplyToFields(DIVisitor *Visitor) {
  Visitor->Apply(Tag);
}

//===----------------------------------------------------------------------===//

AnchorDesc::AnchorDesc()
  : DebugInfoDesc(DW_TAG_anchor), AnchorTag(0) {}

AnchorDesc::AnchorDesc(AnchoredDesc *D)
  : DebugInfoDesc(DW_TAG_anchor), AnchorTag(D->getTag()) {}

// Implement isa/cast/dyncast.
bool AnchorDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_anchor;
}
  
/// getLinkage - get linkage appropriate for this type of descriptor.
GlobalValue::LinkageTypes AnchorDesc::getLinkage() const {
  return GlobalValue::LinkOnceLinkage;
}

/// ApplyToFields - Target the visitor to the fields of the TransUnitDesc.
void AnchorDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  Visitor->Apply(AnchorTag);
}

/// getDescString - Return a string used to compose global names and labels. A
/// global variable name needs to be defined for each debug descriptor that is
/// anchored. NOTE: that each global variable named here also needs to be added
/// to the list of names left external in the internalizer.
///
///   ExternalNames.insert("llvm.dbg.compile_units");
///   ExternalNames.insert("llvm.dbg.global_variables");
///   ExternalNames.insert("llvm.dbg.subprograms");
const char *AnchorDesc::getDescString() const {
  switch (AnchorTag) {
  case DW_TAG_compile_unit: {
    CompileUnitDesc CUD;
    return CUD.getAnchorString();
  }
  case DW_TAG_variable: {
    GlobalVariableDesc GVD;
    return GVD.getAnchorString();
  }
  case DW_TAG_subprogram: {
    SubprogramDesc SPD;
    return SPD.getAnchorString();
  }
  default: break;
  }

  assert(0 && "Tag does not have a case for anchor string");
  return "";
}

#ifndef NDEBUG
void AnchorDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "AnchorTag(" << AnchorTag << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

AnchoredDesc::AnchoredDesc(unsigned T)
  : DebugInfoDesc(T), Anchor(NULL) {}

/// ApplyToFields - Target the visitor to the fields of the AnchoredDesc.
void AnchoredDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  Visitor->Apply(Anchor);
}

//===----------------------------------------------------------------------===//

CompileUnitDesc::CompileUnitDesc()
  : AnchoredDesc(DW_TAG_compile_unit), Language(0), FileName(""),
    Directory(""), Producer("") {}

// Implement isa/cast/dyncast.
bool CompileUnitDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_compile_unit;
}

/// ApplyToFields - Target the visitor to the fields of the CompileUnitDesc.
///
void CompileUnitDesc::ApplyToFields(DIVisitor *Visitor) {
  AnchoredDesc::ApplyToFields(Visitor);
  
  // Handle cases out of sync with compiler.
  if (getVersion() == 0) {
    unsigned DebugVersion;
    Visitor->Apply(DebugVersion);
  }

  Visitor->Apply(Language);
  Visitor->Apply(FileName);
  Visitor->Apply(Directory);
  Visitor->Apply(Producer);
}

#ifndef NDEBUG
void CompileUnitDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Anchor(" << getAnchor() << "), "
       << "Language(" << Language << "), "
       << "FileName(\"" << FileName << "\"), "
       << "Directory(\"" << Directory << "\"), "
       << "Producer(\"" << Producer << "\")\n";
}
#endif

//===----------------------------------------------------------------------===//

TypeDesc::TypeDesc(unsigned T)
  : DebugInfoDesc(T), Context(NULL), Name(""), File(NULL), Line(0), Size(0),
    Align(0), Offset(0), Flags(0) {}

/// ApplyToFields - Target the visitor to the fields of the TypeDesc.
///
void TypeDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(File);
  Visitor->Apply(Line);
  Visitor->Apply(Size);
  Visitor->Apply(Align);
  Visitor->Apply(Offset);
  if (getVersion() > LLVMDebugVersion4) Visitor->Apply(Flags);
}

#ifndef NDEBUG
void TypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << Context << "), "
       << "Name(\"" << Name << "\"), "
       << "File(" << File << "), "
       << "Line(" << Line << "), "
       << "Size(" << Size << "), "
       << "Align(" << Align << "), "
       << "Offset(" << Offset << "), "
       << "Flags(" << Flags << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

BasicTypeDesc::BasicTypeDesc()
  : TypeDesc(DW_TAG_base_type), Encoding(0) {}

// Implement isa/cast/dyncast.
bool BasicTypeDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_base_type;
}

/// ApplyToFields - Target the visitor to the fields of the BasicTypeDesc.
void BasicTypeDesc::ApplyToFields(DIVisitor *Visitor) {
  TypeDesc::ApplyToFields(Visitor);
  Visitor->Apply(Encoding);
}

#ifndef NDEBUG
void BasicTypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << getContext() << "), "
       << "Name(\"" << getName() << "\"), "
       << "Size(" << getSize() << "), "
       << "Encoding(" << Encoding << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

DerivedTypeDesc::DerivedTypeDesc(unsigned T)
  : TypeDesc(T), FromType(NULL) {}

// Implement isa/cast/dyncast.
bool DerivedTypeDesc::classof(const DebugInfoDesc *D) {
  unsigned T =  D->getTag();
  switch (T) {
  case DW_TAG_typedef:
  case DW_TAG_pointer_type:
  case DW_TAG_reference_type:
  case DW_TAG_const_type:
  case DW_TAG_volatile_type:
  case DW_TAG_restrict_type:
  case DW_TAG_member:
  case DW_TAG_inheritance:
    return true;
  default: break;
  }
  return false;
}

/// ApplyToFields - Target the visitor to the fields of the DerivedTypeDesc.
void DerivedTypeDesc::ApplyToFields(DIVisitor *Visitor) {
  TypeDesc::ApplyToFields(Visitor);
  Visitor->Apply(FromType);
}

#ifndef NDEBUG
void DerivedTypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << getContext() << "), "
       << "Name(\"" << getName() << "\"), "
       << "Size(" << getSize() << "), "
       << "File(" << getFile() << "), "
       << "Line(" << getLine() << "), "
       << "FromType(" << FromType << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

CompositeTypeDesc::CompositeTypeDesc(unsigned T)
  : DerivedTypeDesc(T), Elements() {}
  
// Implement isa/cast/dyncast.
bool CompositeTypeDesc::classof(const DebugInfoDesc *D) {
  unsigned T =  D->getTag();
  switch (T) {
  case DW_TAG_array_type:
  case DW_TAG_structure_type:
  case DW_TAG_union_type:
  case DW_TAG_enumeration_type:
  case DW_TAG_vector_type:
  case DW_TAG_subroutine_type:
    return true;
  default: break;
  }
  return false;
}

/// ApplyToFields - Target the visitor to the fields of the CompositeTypeDesc.
///
void CompositeTypeDesc::ApplyToFields(DIVisitor *Visitor) {
  DerivedTypeDesc::ApplyToFields(Visitor);  
  Visitor->Apply(Elements);
}

#ifndef NDEBUG
void CompositeTypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << getContext() << "), "
       << "Name(\"" << getName() << "\"), "
       << "Size(" << getSize() << "), "
       << "File(" << getFile() << "), "
       << "Line(" << getLine() << "), "
       << "FromType(" << getFromType() << "), "
       << "Elements.size(" << Elements.size() << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

SubrangeDesc::SubrangeDesc()
  : DebugInfoDesc(DW_TAG_subrange_type), Lo(0), Hi(0) {}

// Implement isa/cast/dyncast.
bool SubrangeDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_subrange_type;
}

/// ApplyToFields - Target the visitor to the fields of the SubrangeDesc.
void SubrangeDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  Visitor->Apply(Lo);
  Visitor->Apply(Hi);
}

#ifndef NDEBUG
void SubrangeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Lo(" << Lo << "), "
       << "Hi(" << Hi << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

EnumeratorDesc::EnumeratorDesc()
  : DebugInfoDesc(DW_TAG_enumerator), Name(""), Value(0) {}

// Implement isa/cast/dyncast.
bool EnumeratorDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_enumerator;
}

/// ApplyToFields - Target the visitor to the fields of the EnumeratorDesc.
void EnumeratorDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  Visitor->Apply(Name);
  Visitor->Apply(Value);
}

#ifndef NDEBUG
void EnumeratorDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Name(" << Name << "), "
       << "Value(" << Value << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

VariableDesc::VariableDesc(unsigned T)
  : DebugInfoDesc(T), Context(NULL), Name(""), File(NULL), Line(0), TyDesc(0)
{}

// Implement isa/cast/dyncast.
bool VariableDesc::classof(const DebugInfoDesc *D) {
  unsigned T =  D->getTag();
  switch (T) {
  case DW_TAG_auto_variable:
  case DW_TAG_arg_variable:
  case DW_TAG_return_variable:
    return true;
  default: break;
  }
  return false;
}

/// ApplyToFields - Target the visitor to the fields of the VariableDesc.
void VariableDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(File);
  Visitor->Apply(Line);
  Visitor->Apply(TyDesc);
}

#ifndef NDEBUG
void VariableDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << Context << "), "
       << "Name(\"" << Name << "\"), "
       << "File(" << File << "), "
       << "Line(" << Line << "), "
       << "TyDesc(" << TyDesc << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

GlobalDesc::GlobalDesc(unsigned T)
  : AnchoredDesc(T), Context(0), Name(""), FullName(""), LinkageName(""),
    File(NULL), Line(0), TyDesc(NULL), IsStatic(false), IsDefinition(false) {}

/// ApplyToFields - Target the visitor to the fields of the global.
///
void GlobalDesc::ApplyToFields(DIVisitor *Visitor) {
  AnchoredDesc::ApplyToFields(Visitor);
  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(FullName);
  Visitor->Apply(LinkageName);
  Visitor->Apply(File);
  Visitor->Apply(Line);
  Visitor->Apply(TyDesc);
  Visitor->Apply(IsStatic);
  Visitor->Apply(IsDefinition);
}

//===----------------------------------------------------------------------===//

GlobalVariableDesc::GlobalVariableDesc()
  : GlobalDesc(DW_TAG_variable), Global(NULL) {}

// Implement isa/cast/dyncast.
bool GlobalVariableDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_variable; 
}

/// ApplyToFields - Target the visitor to the fields of the GlobalVariableDesc.
void GlobalVariableDesc::ApplyToFields(DIVisitor *Visitor) {
  GlobalDesc::ApplyToFields(Visitor);
  Visitor->Apply(Global);
}

#ifndef NDEBUG
void GlobalVariableDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Anchor(" << getAnchor() << "), "
       << "Name(\"" << getName() << "\"), "
       << "FullName(\"" << getFullName() << "\"), "
       << "LinkageName(\"" << getLinkageName() << "\"), "
       << "File(" << getFile() << "),"
       << "Line(" << getLine() << "),"
       << "Type(" << getType() << "), "
       << "IsStatic(" << (isStatic() ? "true" : "false") << "), "
       << "IsDefinition(" << (isDefinition() ? "true" : "false") << "), "
       << "Global(" << Global << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

SubprogramDesc::SubprogramDesc()
  : GlobalDesc(DW_TAG_subprogram) {}

// Implement isa/cast/dyncast.
bool SubprogramDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_subprogram;
}

/// ApplyToFields - Target the visitor to the fields of the SubprogramDesc.
void SubprogramDesc::ApplyToFields(DIVisitor *Visitor) {
  GlobalDesc::ApplyToFields(Visitor);
}

#ifndef NDEBUG
void SubprogramDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Anchor(" << getAnchor() << "), "
       << "Name(\"" << getName() << "\"), "
       << "FullName(\"" << getFullName() << "\"), "
       << "LinkageName(\"" << getLinkageName() << "\"), "
       << "File(" << getFile() << "),"
       << "Line(" << getLine() << "),"
       << "Type(" << getType() << "), "
       << "IsStatic(" << (isStatic() ? "true" : "false") << "), "
       << "IsDefinition(" << (isDefinition() ? "true" : "false") << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

BlockDesc::BlockDesc()
  : DebugInfoDesc(DW_TAG_lexical_block), Context(NULL) {}

// Implement isa/cast/dyncast.
bool BlockDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_lexical_block;
}

/// ApplyToFields - Target the visitor to the fields of the BlockDesc.
void BlockDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);

  Visitor->Apply(Context);
}

#ifndef NDEBUG
void BlockDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "),"
       << "Context(" << Context << ")\n";
}
#endif
