//===--- CGDebugInfo.cpp - Emit Debug Information for a Module ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the debug information generation while generating code.
//
//===----------------------------------------------------------------------===//

#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Target/TargetMachine.h"
using namespace clang;
using namespace clang::CodeGen;

CGDebugInfo::CGDebugInfo(CodeGenModule *m)
  : M(m), DebugFactory(M->getModule()) {
}

CGDebugInfo::~CGDebugInfo() {
  assert(RegionStack.empty() && "Region stack mismatch, stack not empty!");
}

void CGDebugInfo::setLocation(SourceLocation Loc) {
  if (Loc.isValid())
    CurLoc = M->getContext().getSourceManager().getLogicalLoc(Loc);
}

/// getOrCreateCompileUnit - Get the compile unit from the cache or create a new
/// one if necessary. This returns null for invalid source locations.
llvm::DICompileUnit CGDebugInfo::getOrCreateCompileUnit(SourceLocation Loc) {
  if (Loc.isInvalid())
    return llvm::DICompileUnit();

  SourceManager &SM = M->getContext().getSourceManager();
  const FileEntry *FE = SM.getFileEntryForLoc(Loc);
  if (FE == 0) return llvm::DICompileUnit();
    
  // See if this compile unit has been used before.
  llvm::DICompileUnit &Unit = CompileUnitCache[FE];
  if (!Unit.isNull()) return Unit;
  
  // Get source file information.
  const char *FileName = FE->getName();
  const char *DirName = FE->getDir()->getName();
  
  // Create new compile unit.
  // FIXME: Handle other language IDs as well.
  // FIXME: Do not know how to get clang version yet.
  return Unit = DebugFactory.CreateCompileUnit(llvm::dwarf::DW_LANG_C89,
                                               FileName, DirName, "clang");
}

/// getOrCreateBuiltinType - Get the Basic type from the cache or create a new
/// one if necessary.
llvm::DIType CGDebugInfo::CreateType(const BuiltinType *BT,
                                     llvm::DICompileUnit Unit){
  unsigned Encoding = 0;
  switch (BT->getKind()) {
  default:
  case BuiltinType::Void:
    return llvm::DIType();
  case BuiltinType::UChar:
  case BuiltinType::Char_U: Encoding = llvm::dwarf::DW_ATE_unsigned_char; break;
  case BuiltinType::Char_S:
  case BuiltinType::SChar: Encoding = llvm::dwarf::DW_ATE_signed_char; break;
  case BuiltinType::UShort:
  case BuiltinType::UInt:
  case BuiltinType::ULong:
  case BuiltinType::ULongLong: Encoding = llvm::dwarf::DW_ATE_unsigned; break;
  case BuiltinType::Short:
  case BuiltinType::Int:
  case BuiltinType::Long:
  case BuiltinType::LongLong:  Encoding = llvm::dwarf::DW_ATE_signed; break;
  case BuiltinType::Bool:      Encoding = llvm::dwarf::DW_ATE_boolean; break;
  case BuiltinType::Float:
  case BuiltinType::Double:    Encoding = llvm::dwarf::DW_ATE_float; break;
  } 
  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(BT);
  uint64_t Align = M->getContext().getTypeAlign(BT);
  uint64_t Offset = 0;
  
  return DebugFactory.CreateBasicType(Unit, BT->getName(), Unit, 0, Size, Align,
                                      Offset, /*flags*/ 0, Encoding);
}

/// getOrCreateCVRType - Get the CVR qualified type from the cache or create 
/// a new one if necessary.
llvm::DIType CGDebugInfo::CreateCVRType(QualType Ty, llvm::DICompileUnit Unit) {
  // We will create one Derived type for one qualifier and recurse to handle any
  // additional ones.
  llvm::DIType FromTy;
  unsigned Tag;
  if (Ty.isConstQualified()) {
    Tag = llvm::dwarf::DW_TAG_const_type;
    Ty.removeConst(); 
    FromTy = getOrCreateType(Ty, Unit);
  } else if (Ty.isVolatileQualified()) {
    Tag = llvm::dwarf::DW_TAG_volatile_type;
    Ty.removeVolatile(); 
    FromTy = getOrCreateType(Ty, Unit);
  } else {
    assert(Ty.isRestrictQualified() && "Unknown type qualifier for debug info");
    Tag = llvm::dwarf::DW_TAG_restrict_type;
    Ty.removeRestrict(); 
    FromTy = getOrCreateType(Ty, Unit);
  }
  
  // No need to fill in the Name, Line, Size, Alignment, Offset in case of
  // CVR derived types.
  return DebugFactory.CreateDerivedType(Tag, Unit, "", llvm::DICompileUnit(),
                                        0, 0, 0, 0, 0, FromTy);
}

llvm::DIType CGDebugInfo::CreateType(const PointerType *Ty,
                                     llvm::DICompileUnit Unit) {
  llvm::DIType EltTy = getOrCreateType(Ty->getPointeeType(), Unit);
 
  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);
                                                                               
  return DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type, Unit,
                                        "", llvm::DICompileUnit(),
                                        0, Size, Align, 0, 0, EltTy);
}

llvm::DIType CGDebugInfo::CreateType(const TypedefType *Ty,
                                     llvm::DICompileUnit Unit) {
  // Typedefs are derived from some other type.  If we have a typedef of a
  // typedef, make sure to emit the whole chain.
  llvm::DIType Src = getOrCreateType(Ty->getDecl()->getUnderlyingType(), Unit);
  
  // We don't set size information, but do specify where the typedef was
  // declared.
  const char *TyName = Ty->getDecl()->getName();
  SourceLocation DefLoc = Ty->getDecl()->getLocation();
  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(DefLoc);

  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(DefLoc);

  return DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_typedef, Unit,
                                        TyName, DefUnit, Line, 0, 0, 0, 0, Src);
}

llvm::DIType CGDebugInfo::CreateType(const FunctionType *Ty,
                                     llvm::DICompileUnit Unit) {
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  // Add the result type at least.
  EltTys.push_back(getOrCreateType(Ty->getResultType(), Unit));
  
  // Set up remainder of arguments if there is a prototype.
  // FIXME: IF NOT, HOW IS THIS REPRESENTED?  llvm-gcc doesn't represent '...'!
  if (const FunctionTypeProto *FTP = dyn_cast<FunctionTypeProto>(Ty)) {
    for (unsigned i = 0, e = FTP->getNumArgs(); i != e; ++i)
      EltTys.push_back(getOrCreateType(FTP->getArgType(i), Unit));
  } else {
    // FIXME: Handle () case in C.  llvm-gcc doesn't do it either.
  }

  llvm::DIArray EltTypeArray =
    DebugFactory.GetOrCreateArray(&EltTys[0], EltTys.size());
  
  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_subroutine_type,
                                          Unit, "", llvm::DICompileUnit(),
                                          0, 0, 0, 0, 0,
                                          llvm::DIType(), EltTypeArray);
}

/// getOrCreateRecordType - get structure or union type.
llvm::DIType CGDebugInfo::CreateType(const RecordType *Ty,
                                     llvm::DICompileUnit Unit) {
  const RecordDecl *Decl = Ty->getDecl();
  
  unsigned Tag;
  if (Decl->isStruct())
    Tag = llvm::dwarf::DW_TAG_structure_type;
  else if (Decl->isUnion())
    Tag = llvm::dwarf::DW_TAG_union_type;
  else {
    assert(Decl->isClass() && "Unknown RecordType!");
    Tag = llvm::dwarf::DW_TAG_class_type;
  }

  SourceManager &SM = M->getContext().getSourceManager();

  // Get overall information about the record type for the debug info.
  const char *Name = Decl->getName();
  if (Name == 0) Name = "";

  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(Decl->getLocation());
  uint64_t Line = SM.getLogicalLineNumber(Decl->getLocation());
  
  
  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the struct as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the struct is recursive) and replace all
  // uses of the forward declaration with the final definition.
  llvm::DIType FwdDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Name, DefUnit, Line, 0, 0, 0, 0,
                                     llvm::DIType(), llvm::DIArray());
  
  // If this is just a forward declaration, return it.
  if (!Decl->getDefinition(M->getContext()))
    return FwdDecl;

  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  const ASTRecordLayout &RL = M->getContext().getASTRecordLayout(Decl);

  unsigned FieldNo = 0;
  for (RecordDecl::field_const_iterator I = Decl->field_begin(),
       E = Decl->field_end(); I != E; ++I, ++FieldNo) {
    FieldDecl *Field = *I;
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);
    
    const char *FieldName = Field->getName();
    if (FieldName == 0) FieldName = "";

    // Get the location for the field.
    SourceLocation FieldDefLoc = Field->getLocation();
    llvm::DICompileUnit FieldDefUnit = getOrCreateCompileUnit(FieldDefLoc);
    uint64_t FieldLine = SM.getLogicalLineNumber(FieldDefLoc);
    
    // Bit size, align and offset of the type.
    uint64_t FieldSize = M->getContext().getTypeSize(Ty);
    uint64_t FieldAlign = M->getContext().getTypeAlign(Ty);
    uint64_t FieldOffset = RL.getFieldOffset(FieldNo);    
    
    // Create a DW_TAG_member node to remember the offset of this field in the
    // struct.  FIXME: This is an absolutely insane way to capture this
    // information.  When we gut debug info, this should be fixed.
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             FieldName, FieldDefUnit,
                                             FieldLine, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
  }
  
  llvm::DIArray Elements =
    DebugFactory.GetOrCreateArray(&EltTys[0], EltTys.size());

  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);
  
  llvm::DIType RealDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Name, DefUnit, Line, Size,
                                     Align, 0, 0, llvm::DIType(), Elements);

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  FwdDecl.getGV()->replaceAllUsesWith(RealDecl.getGV());
  FwdDecl.getGV()->eraseFromParent();
  
  return RealDecl;
}

llvm::DIType CGDebugInfo::CreateType(const EnumType *Ty,
                                     llvm::DICompileUnit Unit) {
  EnumDecl *Decl = Ty->getDecl();

  llvm::SmallVector<llvm::DIDescriptor, 32> Enumerators;

  // Create DIEnumerator elements for each enumerator.
  for (EnumConstantDecl *Elt = Decl->getEnumConstantList(); Elt;
       Elt = dyn_cast_or_null<EnumConstantDecl>(Elt->getNextDeclarator())) {
    Enumerators.push_back(DebugFactory.CreateEnumerator(Elt->getName(),
                                            Elt->getInitVal().getZExtValue()));
  }
  
  // Return a CompositeType for the enum itself.
  llvm::DIArray EltArray =
    DebugFactory.GetOrCreateArray(&Enumerators[0], Enumerators.size());

  const char *EnumName = Decl->getName() ? Decl->getName() : "";
  SourceLocation DefLoc = Decl->getLocation();
  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(DefLoc);
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(DefLoc);
  
  // Size and align of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);
  
  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_enumeration_type,
                                          Unit, EnumName, DefUnit, Line,
                                          Size, Align, 0, 0,
                                          llvm::DIType(), EltArray);
}

llvm::DIType CGDebugInfo::CreateType(const TagType *Ty,
                                     llvm::DICompileUnit Unit) {
  if (const RecordType *RT = dyn_cast<RecordType>(Ty))
    return CreateType(RT, Unit);
  else if (const EnumType *ET = dyn_cast<EnumType>(Ty))
    return CreateType(ET, Unit);
  
  return llvm::DIType();
}

llvm::DIType CGDebugInfo::CreateType(const ArrayType *Ty,
                                     llvm::DICompileUnit Unit) {
  // Size and align of the whole array, not the element type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);
  
  // Add the dimensions of the array.  FIXME: This loses CV qualifiers from
  // interior arrays, do we care?  Why aren't nested arrays represented the
  // obvious/recursive way?
  llvm::SmallVector<llvm::DIDescriptor, 8> Subscripts;
  QualType EltTy(Ty, 0);
  while ((Ty = dyn_cast<ArrayType>(EltTy))) {
    uint64_t Upper = 0;
    if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(Ty))
      Upper = CAT->getSize().getZExtValue() - 1;
    // FIXME: Verify this is right for VLAs.
    Subscripts.push_back(DebugFactory.GetOrCreateSubrange(0, Upper));
    EltTy = Ty->getElementType();
  }
  
  llvm::DIArray SubscriptArray =
    DebugFactory.GetOrCreateArray(&Subscripts[0], Subscripts.size());

  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_array_type,
                                          Unit, "", llvm::DICompileUnit(),
                                          0, Size, Align, 0, 0,
                                          getOrCreateType(EltTy, Unit),
                                          SubscriptArray);
}


/// getOrCreateType - Get the type from the cache or create a new
/// one if necessary.
llvm::DIType CGDebugInfo::getOrCreateType(QualType Ty,
                                          llvm::DICompileUnit Unit) {
  if (Ty.isNull())
    return llvm::DIType();
  
  // Check to see if the compile unit already has created this type.
  llvm::DIType &Slot = TypeCache[Ty.getAsOpaquePtr()];
  if (!Slot.isNull()) return Slot;

  // Handle CVR qualifiers, which recursively handles what they refer to.
  if (Ty.getCVRQualifiers())
    return Slot = CreateCVRType(Ty, Unit);

  // Work out details of type.
  switch (Ty->getTypeClass()) {
  case Type::Complex:
  case Type::Reference:
  case Type::Vector:
  case Type::ExtVector:
  case Type::ASQual:
  case Type::ObjCInterface:
  case Type::ObjCQualifiedInterface:
  case Type::ObjCQualifiedId:
  case Type::TypeOfExp:
  case Type::TypeOfTyp:
  default:
    return llvm::DIType();

  case Type::Builtin: Slot = CreateType(cast<BuiltinType>(Ty), Unit); break;
  case Type::Pointer: Slot = CreateType(cast<PointerType>(Ty), Unit); break;
  case Type::TypeName: Slot = CreateType(cast<TypedefType>(Ty), Unit); break;
  case Type::Tagged: Slot = CreateType(cast<TagType>(Ty), Unit); break;
  case Type::FunctionProto:
  case Type::FunctionNoProto:
    Slot = CreateType(cast<FunctionType>(Ty), Unit);
    break;
    
  case Type::ConstantArray:
  case Type::VariableArray:
  case Type::IncompleteArray:
    Slot = CreateType(cast<ArrayType>(Ty), Unit);
    break;
  }
  
  return Slot;
}

/// EmitFunctionStart - Constructs the debug code for entering a function -
/// "llvm.dbg.func.start.".
void CGDebugInfo::EmitFunctionStart(const char *Name, QualType ReturnType,
                                    llvm::Function *Fn,
                                    CGBuilderTy &Builder) {
  // FIXME: Why is this using CurLoc???
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(CurLoc);
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t LineNo = SM.getLogicalLineNumber(CurLoc);
  
  llvm::DISubprogram SP =
    DebugFactory.CreateSubprogram(Unit, Name, Name, "", Unit, LineNo,
                                  getOrCreateType(ReturnType, Unit),
                                  Fn->hasInternalLinkage(), true/*definition*/);
  
  DebugFactory.InsertSubprogramStart(SP, Builder.GetInsertBlock());
                                                        
  // Push function on region stack.
  RegionStack.push_back(SP);
}


void CGDebugInfo::EmitStopPoint(llvm::Function *Fn, CGBuilderTy &Builder) {
  if (CurLoc.isInvalid() || CurLoc.isMacroID()) return;
  
  // Don't bother if things are the same as last time.
  SourceManager &SM = M->getContext().getSourceManager();
  if (CurLoc == PrevLoc 
       || (SM.getLineNumber(CurLoc) == SM.getLineNumber(PrevLoc)
           && SM.isFromSameFile(CurLoc, PrevLoc)))
    return;

  // Update last state.
  PrevLoc = CurLoc;

  // Get the appropriate compile unit.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(CurLoc);
  DebugFactory.InsertStopPoint(Unit, SM.getLogicalLineNumber(CurLoc),
                               SM.getLogicalColumnNumber(CurLoc),
                               Builder.GetInsertBlock()); 
}

/// EmitRegionStart- Constructs the debug code for entering a declarative
/// region - "llvm.dbg.region.start.".
void CGDebugInfo::EmitRegionStart(llvm::Function *Fn, CGBuilderTy &Builder) {
  llvm::DIDescriptor D;
  if (!RegionStack.empty())
    D = RegionStack.back();
  D = DebugFactory.CreateBlock(D);
  RegionStack.push_back(D);
  DebugFactory.InsertRegionStart(D, Builder.GetInsertBlock());
}

/// EmitRegionEnd - Constructs the debug code for exiting a declarative
/// region - "llvm.dbg.region.end."
void CGDebugInfo::EmitRegionEnd(llvm::Function *Fn, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Provide an region stop point.
  EmitStopPoint(Fn, Builder);
  
  DebugFactory.InsertRegionEnd(RegionStack.back(), Builder.GetInsertBlock());
  RegionStack.pop_back();
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const VarDecl *Decl, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Get location information.
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(Decl->getLocation());
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(Decl->getLocation());
  
  // Create the descriptor for the variable.
  llvm::DIVariable D = 
    DebugFactory.CreateVariable(Tag, RegionStack.back(), Decl->getName(),
                                Unit, Line,
                                getOrCreateType(Decl->getType(), Unit));
  // Insert an llvm.dbg.declare into the current block.
  DebugFactory.InsertDeclare(Storage, D, Builder.GetInsertBlock());
}

void CGDebugInfo::EmitDeclareOfAutoVariable(const VarDecl *Decl,
                                            llvm::Value *Storage,
                                            CGBuilderTy &Builder) {
  EmitDeclare(Decl, llvm::dwarf::DW_TAG_auto_variable, Storage, Builder);
}

/// EmitDeclareOfArgVariable - Emit call to llvm.dbg.declare for an argument
/// variable declaration.
void CGDebugInfo::EmitDeclareOfArgVariable(const VarDecl *Decl, llvm::Value *AI,
                                           CGBuilderTy &Builder) {
  EmitDeclare(Decl, llvm::dwarf::DW_TAG_arg_variable, AI, Builder);
}



/// EmitGlobalVariable - Emit information about a global variable.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *Var, 
                                     const VarDecl *Decl) {
  // Create global variable debug descriptor.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(Decl->getLocation());
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t LineNo = SM.getLogicalLineNumber(Decl->getLocation());
  const char *Name = Decl->getName();
  
  DebugFactory.CreateGlobalVariable(Unit, Name, Name, "", Unit, LineNo,
                                    getOrCreateType(Decl->getType(), Unit),
                                    Var->hasInternalLinkage(),
                                    true/*definition*/, Var);
}

