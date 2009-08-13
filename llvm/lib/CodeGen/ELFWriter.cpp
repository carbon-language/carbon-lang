//===-- ELFWriter.cpp - Target-independent ELF Writer code ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-independent ELF writer.  This file writes out
// the ELF file in the following order:
//
//  #1. ELF Header
//  #2. '.text' section
//  #3. '.data' section
//  #4. '.bss' section  (conceptual position in file)
//  ...
//  #X. '.shstrtab' section
//  #Y. Section Table
//
// The entries in the section table are laid out as:
//  #0. Null entry [required]
//  #1. ".text" entry - the program code
//  #2. ".data" entry - global variables with initializers.     [ if needed ]
//  #3. ".bss" entry  - global variables without initializers.  [ if needed ]
//  ...
//  #N. ".shstrtab" entry - String table for the section names.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "elfwriter"
#include "ELF.h"
#include "ELFWriter.h"
#include "ELFCodeEmitter.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/BinaryObject.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/ObjectCodeEmitter.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetELFWriterInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

char ELFWriter::ID = 0;

/// AddELFWriter - Add the ELF writer to the function pass manager
ObjectCodeEmitter *llvm::AddELFWriter(PassManagerBase &PM,
                                      raw_ostream &O,
                                      TargetMachine &TM) {
  ELFWriter *EW = new ELFWriter(O, TM);
  PM.add(EW);
  return EW->getObjectCodeEmitter();
}

//===----------------------------------------------------------------------===//
//                          ELFWriter Implementation
//===----------------------------------------------------------------------===//

ELFWriter::ELFWriter(raw_ostream &o, TargetMachine &tm)
  : MachineFunctionPass(&ID), O(o), TM(tm),
    OutContext(*new MCContext()),
    TLOF(TM.getTargetLowering()->getObjFileLowering()),
    is64Bit(TM.getTargetData()->getPointerSizeInBits() == 64),
    isLittleEndian(TM.getTargetData()->isLittleEndian()),
    ElfHdr(isLittleEndian, is64Bit) {

  TAI = TM.getTargetAsmInfo();
  TEW = TM.getELFWriterInfo();

  // Create the object code emitter object for this target.
  ElfCE = new ELFCodeEmitter(*this);

  // Inital number of sections
  NumSections = 0;
}

ELFWriter::~ELFWriter() {
  delete ElfCE;
  delete &OutContext;
}

// doInitialization - Emit the file header and all of the global variables for
// the module to the ELF file.
bool ELFWriter::doInitialization(Module &M) {
  // Initialize TargetLoweringObjectFile.
  const_cast<TargetLoweringObjectFile&>(TLOF).Initialize(OutContext, TM);
  
  Mang = new Mangler(M);

  // ELF Header
  // ----------
  // Fields e_shnum e_shstrndx are only known after all section have
  // been emitted. They locations in the ouput buffer are recorded so
  // to be patched up later.
  //
  // Note
  // ----
  // emitWord method behaves differently for ELF32 and ELF64, writing
  // 4 bytes in the former and 8 in the last for *_off and *_addr elf types

  ElfHdr.emitByte(0x7f); // e_ident[EI_MAG0]
  ElfHdr.emitByte('E');  // e_ident[EI_MAG1]
  ElfHdr.emitByte('L');  // e_ident[EI_MAG2]
  ElfHdr.emitByte('F');  // e_ident[EI_MAG3]

  ElfHdr.emitByte(TEW->getEIClass()); // e_ident[EI_CLASS]
  ElfHdr.emitByte(TEW->getEIData());  // e_ident[EI_DATA]
  ElfHdr.emitByte(EV_CURRENT);        // e_ident[EI_VERSION]
  ElfHdr.emitAlignment(16);           // e_ident[EI_NIDENT-EI_PAD]

  ElfHdr.emitWord16(ET_REL);             // e_type
  ElfHdr.emitWord16(TEW->getEMachine()); // e_machine = target
  ElfHdr.emitWord32(EV_CURRENT);         // e_version
  ElfHdr.emitWord(0);                    // e_entry, no entry point in .o file
  ElfHdr.emitWord(0);                    // e_phoff, no program header for .o
  ELFHdr_e_shoff_Offset = ElfHdr.size();
  ElfHdr.emitWord(0);                    // e_shoff = sec hdr table off in bytes
  ElfHdr.emitWord32(TEW->getEFlags());   // e_flags = whatever the target wants
  ElfHdr.emitWord16(TEW->getHdrSize());  // e_ehsize = ELF header size
  ElfHdr.emitWord16(0);                  // e_phentsize = prog header entry size
  ElfHdr.emitWord16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  ElfHdr.emitWord16(TEW->getSHdrSize());

  // e_shnum     = # of section header ents
  ELFHdr_e_shnum_Offset = ElfHdr.size();
  ElfHdr.emitWord16(0); // Placeholder

  // e_shstrndx  = Section # of '.shstrtab'
  ELFHdr_e_shstrndx_Offset = ElfHdr.size();
  ElfHdr.emitWord16(0); // Placeholder

  // Add the null section, which is required to be first in the file.
  getNullSection();

  // The first entry in the symtab is the null symbol and the second
  // is a local symbol containing the module/file name
  SymbolList.push_back(new ELFSym());
  SymbolList.push_back(ELFSym::getFileSym());

  return false;
}

// AddPendingGlobalSymbol - Add a global to be processed and to
// the global symbol lookup, use a zero index because the table
// index will be determined later.
void ELFWriter::AddPendingGlobalSymbol(const GlobalValue *GV, 
                                       bool AddToLookup /* = false */) {
  PendingGlobals.insert(GV);
  if (AddToLookup) 
    GblSymLookup[GV] = 0;
}

// AddPendingExternalSymbol - Add the external to be processed
// and to the external symbol lookup, use a zero index because
// the symbol table index will be determined later.
void ELFWriter::AddPendingExternalSymbol(const char *External) {
  PendingExternals.insert(External);
  ExtSymLookup[External] = 0;
}

ELFSection &ELFWriter::getDataSection() {
  const MCSectionELF *Data = (const MCSectionELF *)TLOF.getDataSection();
  return getSection(Data->getSectionName(), Data->getType(), 
                    Data->getFlags(), 4);
}

ELFSection &ELFWriter::getBSSSection() {
  const MCSectionELF *BSS = (const MCSectionELF *)TLOF.getBSSSection();
  return getSection(BSS->getSectionName(), BSS->getType(), BSS->getFlags(), 4);
}

// getCtorSection - Get the static constructor section
ELFSection &ELFWriter::getCtorSection() {
  const MCSectionELF *Ctor = (const MCSectionELF *)TLOF.getStaticCtorSection();
  return getSection(Ctor->getSectionName(), Ctor->getType(), Ctor->getFlags()); 
}

// getDtorSection - Get the static destructor section
ELFSection &ELFWriter::getDtorSection() {
  const MCSectionELF *Dtor = (const MCSectionELF *)TLOF.getStaticDtorSection();
  return getSection(Dtor->getSectionName(), Dtor->getType(), Dtor->getFlags());
}

// getTextSection - Get the text section for the specified function
ELFSection &ELFWriter::getTextSection(Function *F) {
  const MCSectionELF *Text = 
    (const MCSectionELF *)TLOF.SectionForGlobal(F, Mang, TM);
  return getSection(Text->getSectionName(), Text->getType(), Text->getFlags());
}

// getJumpTableSection - Get a read only section for constants when 
// emitting jump tables. TODO: add PIC support
ELFSection &ELFWriter::getJumpTableSection() {
  const MCSectionELF *JT = 
    (const MCSectionELF *)TLOF.getSectionForConstant(SectionKind::getReadOnly());
  return getSection(JT->getSectionName(), JT->getType(), JT->getFlags(),
                    TM.getTargetData()->getPointerABIAlignment());
}

// getConstantPoolSection - Get a constant pool section based on the machine 
// constant pool entry type and relocation info.
ELFSection &ELFWriter::getConstantPoolSection(MachineConstantPoolEntry &CPE) {
  SectionKind Kind;
  switch (CPE.getRelocationInfo()) {
  default: llvm_unreachable("Unknown section kind");
  case 2: Kind = SectionKind::getReadOnlyWithRel(); break;
  case 1:
    Kind = SectionKind::getReadOnlyWithRelLocal();
    break;
  case 0:
    switch (TM.getTargetData()->getTypeAllocSize(CPE.getType())) {
    case 4:  Kind = SectionKind::getMergeableConst4(); break;
    case 8:  Kind = SectionKind::getMergeableConst8(); break;
    case 16: Kind = SectionKind::getMergeableConst16(); break;
    default: Kind = SectionKind::getMergeableConst(); break;
    }
  }

  const MCSectionELF *CPSect = 
    (const MCSectionELF *)TLOF.getSectionForConstant(Kind);
  return getSection(CPSect->getSectionName(), CPSect->getType(), 
                    CPSect->getFlags(), CPE.getAlignment());
}

// getRelocSection - Return the relocation section of section 'S'. 'RelA' 
// is true if the relocation section contains entries with addends.
ELFSection &ELFWriter::getRelocSection(ELFSection &S) {
  unsigned SectionType = TEW->hasRelocationAddend() ?
                ELFSection::SHT_RELA : ELFSection::SHT_REL;

  std::string SectionName(".rel");
  if (TEW->hasRelocationAddend())
    SectionName.append("a");
  SectionName.append(S.getName());

  return getSection(SectionName, SectionType, 0, TEW->getPrefELFAlignment());
}

// getGlobalELFVisibility - Returns the ELF specific visibility type
unsigned ELFWriter::getGlobalELFVisibility(const GlobalValue *GV) {
  switch (GV->getVisibility()) {
  default:
    llvm_unreachable("unknown visibility type");
  case GlobalValue::DefaultVisibility:
    return ELFSym::STV_DEFAULT;
  case GlobalValue::HiddenVisibility:
    return ELFSym::STV_HIDDEN;
  case GlobalValue::ProtectedVisibility:
    return ELFSym::STV_PROTECTED;
  }
  return 0;
}

// getGlobalELFBinding - Returns the ELF specific binding type
unsigned ELFWriter::getGlobalELFBinding(const GlobalValue *GV) {
  if (GV->hasInternalLinkage())
    return ELFSym::STB_LOCAL;

  if (GV->isWeakForLinker() && !GV->hasCommonLinkage())
    return ELFSym::STB_WEAK;

  return ELFSym::STB_GLOBAL;
}

// getGlobalELFType - Returns the ELF specific type for a global
unsigned ELFWriter::getGlobalELFType(const GlobalValue *GV) {
  if (GV->isDeclaration())
    return ELFSym::STT_NOTYPE;

  if (isa<Function>(GV))
    return ELFSym::STT_FUNC;

  return ELFSym::STT_OBJECT;
}

// IsELFUndefSym - True if the global value must be marked as a symbol
// which points to a SHN_UNDEF section. This means that the symbol has
// no definition on the module.
static bool IsELFUndefSym(const GlobalValue *GV) {
  return GV->isDeclaration() || (isa<Function>(GV));
}

// AddToSymbolList - Update the symbol lookup and If the symbol is 
// private add it to PrivateSyms list, otherwise to SymbolList. 
void ELFWriter::AddToSymbolList(ELFSym *GblSym) {
  assert(GblSym->isGlobalValue() && "Symbol must be a global value");

  const GlobalValue *GV = GblSym->getGlobalValue(); 
  if (GV->hasPrivateLinkage()) {
    // For a private symbols, keep track of the index inside 
    // the private list since it will never go to the symbol 
    // table and won't be patched up later.
    PrivateSyms.push_back(GblSym);
    GblSymLookup[GV] = PrivateSyms.size()-1;
  } else {
    // Non private symbol are left with zero indices until 
    // they are patched up during the symbol table emition 
    // (where the indicies are created).
    SymbolList.push_back(GblSym);
    GblSymLookup[GV] = 0;
  }
}

// EmitGlobal - Choose the right section for global and emit it
void ELFWriter::EmitGlobal(const GlobalValue *GV) {

  // Check if the referenced symbol is already emitted
  if (GblSymLookup.find(GV) != GblSymLookup.end())
    return;

  // Handle ELF Bind, Visibility and Type for the current symbol
  unsigned SymBind = getGlobalELFBinding(GV);
  unsigned SymType = getGlobalELFType(GV);
  bool IsUndefSym = IsELFUndefSym(GV);

  ELFSym *GblSym = IsUndefSym ? ELFSym::getUndefGV(GV, SymBind)
    : ELFSym::getGV(GV, SymBind, SymType, getGlobalELFVisibility(GV));

  if (!IsUndefSym) {
    assert(isa<GlobalVariable>(GV) && "GV not a global variable!");
    const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);

    // Handle special llvm globals
    if (EmitSpecialLLVMGlobal(GVar))
      return;

    // Get the ELF section where this global belongs from TLOF
    const MCSectionELF *S = 
      (const MCSectionELF *)TLOF.SectionForGlobal(GV, Mang, TM);
    ELFSection &ES = 
      getSection(S->getSectionName(), S->getType(), S->getFlags());
    SectionKind Kind = S->getKind();

    // The symbol align should update the section alignment if needed
    const TargetData *TD = TM.getTargetData();
    unsigned Align = TD->getPreferredAlignment(GVar);
    unsigned Size = TD->getTypeAllocSize(GVar->getInitializer()->getType());
    GblSym->Size = Size;

    if (S->IsCommon()) { // Symbol must go to a common section
      GblSym->SectionIdx = ELFSection::SHN_COMMON;

      // A new linkonce section is created for each global in the
      // common section, the default alignment is 1 and the symbol
      // value contains its alignment.
      ES.Align = 1;
      GblSym->Value = Align;

    } else if (Kind.isBSS() || Kind.isThreadBSS()) { // Symbol goes to BSS.
      GblSym->SectionIdx = ES.SectionIdx;

      // Update the size with alignment and the next object can
      // start in the right offset in the section
      if (Align) ES.Size = (ES.Size + Align-1) & ~(Align-1);
      ES.Align = std::max(ES.Align, Align);

      // GblSym->Value should contain the virtual offset inside the section.
      // Virtual because the BSS space is not allocated on ELF objects
      GblSym->Value = ES.Size;
      ES.Size += Size;

    } else { // The symbol must go to some kind of data section
      GblSym->SectionIdx = ES.SectionIdx;

      // GblSym->Value should contain the symbol offset inside the section,
      // and all symbols should start on their required alignment boundary
      ES.Align = std::max(ES.Align, Align);
      ES.emitAlignment(Align);
      GblSym->Value = ES.size();

      // Emit the global to the data section 'ES'
      EmitGlobalConstant(GVar->getInitializer(), ES);
    }
  }

  AddToSymbolList(GblSym);
}

void ELFWriter::EmitGlobalConstantStruct(const ConstantStruct *CVS,
                                         ELFSection &GblS) {

  // Print the fields in successive locations. Pad to align if needed!
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(CVS->getType());
  const StructLayout *cvsLayout = TD->getStructLayout(CVS->getType());
  uint64_t sizeSoFar = 0;
  for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
    const Constant* field = CVS->getOperand(i);

    // Check if padding is needed and insert one or more 0s.
    uint64_t fieldSize = TD->getTypeAllocSize(field->getType());
    uint64_t padSize = ((i == e-1 ? Size : cvsLayout->getElementOffset(i+1))
                        - cvsLayout->getElementOffset(i)) - fieldSize;
    sizeSoFar += fieldSize + padSize;

    // Now print the actual field value.
    EmitGlobalConstant(field, GblS);

    // Insert padding - this may include padding to increase the size of the
    // current field up to the ABI size (if the struct is not packed) as well
    // as padding to ensure that the next field starts at the right offset.
    GblS.emitZeros(padSize);
  }
  assert(sizeSoFar == cvsLayout->getSizeInBytes() &&
         "Layout of constant struct may be incorrect!");
}

void ELFWriter::EmitGlobalConstant(const Constant *CV, ELFSection &GblS) {
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(CV->getType());

  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
      EmitGlobalConstant(CVA->getOperand(i), GblS);
    return;
  } else if (isa<ConstantAggregateZero>(CV)) {
    GblS.emitZeros(Size);
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    EmitGlobalConstantStruct(CVS, GblS);
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    APInt Val = CFP->getValueAPF().bitcastToAPInt();
    if (CFP->getType() == Type::getDoubleTy(CV->getContext()))
      GblS.emitWord64(Val.getZExtValue());
    else if (CFP->getType() == Type::getFloatTy(CV->getContext()))
      GblS.emitWord32(Val.getZExtValue());
    else if (CFP->getType() == Type::getX86_FP80Ty(CV->getContext())) {
      unsigned PadSize = 
             TD->getTypeAllocSize(Type::getX86_FP80Ty(CV->getContext()))-
             TD->getTypeStoreSize(Type::getX86_FP80Ty(CV->getContext()));
      GblS.emitWordFP80(Val.getRawData(), PadSize);
    } else if (CFP->getType() == Type::getPPC_FP128Ty(CV->getContext()))
      llvm_unreachable("PPC_FP128Ty global emission not implemented");
    return;
  } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    if (Size == 1)
      GblS.emitByte(CI->getZExtValue());
    else if (Size == 2) 
      GblS.emitWord16(CI->getZExtValue());
    else if (Size == 4)
      GblS.emitWord32(CI->getZExtValue());
    else 
      EmitGlobalConstantLargeInt(CI, GblS);
    return;
  } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(CV)) {
    const VectorType *PTy = CP->getType();
    for (unsigned I = 0, E = PTy->getNumElements(); I < E; ++I)
      EmitGlobalConstant(CP->getOperand(I), GblS);
    return;
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    // Resolve a constant expression which returns a (Constant, Offset)
    // pair. If 'Res.first' is a GlobalValue, emit a relocation with 
    // the offset 'Res.second', otherwise emit a global constant like
    // it is always done for not contant expression types.
    CstExprResTy Res = ResolveConstantExpr(CE);
    const Constant *Op = Res.first;

    if (isa<GlobalValue>(Op))
      EmitGlobalDataRelocation(cast<const GlobalValue>(Op), 
                               TD->getTypeAllocSize(Op->getType()), 
                               GblS, Res.second);
    else
      EmitGlobalConstant(Op, GblS);

    return;
  } else if (CV->getType()->getTypeID() == Type::PointerTyID) {
    // Fill the data entry with zeros or emit a relocation entry
    if (isa<ConstantPointerNull>(CV))
      GblS.emitZeros(Size);
    else 
      EmitGlobalDataRelocation(cast<const GlobalValue>(CV), 
                               Size, GblS);
    return;
  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV)) {
    // This is a constant address for a global variable or function and
    // therefore must be referenced using a relocation entry.
    EmitGlobalDataRelocation(GV, Size, GblS);
    return;
  }

  std::string msg;
  raw_string_ostream ErrorMsg(msg);
  ErrorMsg << "Constant unimp for type: " << *CV->getType();
  llvm_report_error(ErrorMsg.str());
}

// ResolveConstantExpr - Resolve the constant expression until it stop
// yielding other constant expressions.
CstExprResTy ELFWriter::ResolveConstantExpr(const Constant *CV) {
  const TargetData *TD = TM.getTargetData();
  
  // There ins't constant expression inside others anymore
  if (!isa<ConstantExpr>(CV))
    return std::make_pair(CV, 0);

  const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV);
  switch (CE->getOpcode()) {
  case Instruction::BitCast:
    return ResolveConstantExpr(CE->getOperand(0));
  
  case Instruction::GetElementPtr: {
    const Constant *ptrVal = CE->getOperand(0);
    SmallVector<Value*, 8> idxVec(CE->op_begin()+1, CE->op_end());
    int64_t Offset = TD->getIndexedOffset(ptrVal->getType(), &idxVec[0],
                                          idxVec.size());
    return std::make_pair(ptrVal, Offset);
  }
  case Instruction::IntToPtr: {
    Constant *Op = CE->getOperand(0);
    Op = ConstantExpr::getIntegerCast(Op, TD->getIntPtrType(CV->getContext()),
                                      false/*ZExt*/);
    return ResolveConstantExpr(Op);
  }
  case Instruction::PtrToInt: {
    Constant *Op = CE->getOperand(0);
    const Type *Ty = CE->getType();

    // We can emit the pointer value into this slot if the slot is an
    // integer slot greater or equal to the size of the pointer.
    if (TD->getTypeAllocSize(Ty) == TD->getTypeAllocSize(Op->getType()))
      return ResolveConstantExpr(Op);

    llvm_unreachable("Integer size less then pointer size");
  }
  case Instruction::Add:
  case Instruction::Sub: {
    // Only handle cases where there's a constant expression with GlobalValue
    // as first operand and ConstantInt as second, which are the cases we can
    // solve direclty using a relocation entry. GlobalValue=Op0, CstInt=Op1
    // 1)  Instruction::Add  => (global) + CstInt
    // 2)  Instruction::Sub  => (global) + -CstInt
    const Constant *Op0 = CE->getOperand(0); 
    const Constant *Op1 = CE->getOperand(1); 
    assert(isa<ConstantInt>(Op1) && "Op1 must be a ConstantInt");

    CstExprResTy Res = ResolveConstantExpr(Op0);
    assert(isa<GlobalValue>(Res.first) && "Op0 must be a GlobalValue");

    const APInt &RHS = cast<ConstantInt>(Op1)->getValue();
    switch (CE->getOpcode()) {
    case Instruction::Add: 
      return std::make_pair(Res.first, RHS.getSExtValue());
    case Instruction::Sub:
      return std::make_pair(Res.first, (-RHS).getSExtValue());
    }
  }
  }

  std::string msg(CE->getOpcodeName());
  raw_string_ostream ErrorMsg(msg);
  ErrorMsg << ": Unsupported ConstantExpr type";
  llvm_report_error(ErrorMsg.str());

  return std::make_pair(CV, 0); // silence warning
}

void ELFWriter::EmitGlobalDataRelocation(const GlobalValue *GV, unsigned Size,
                                         ELFSection &GblS, int64_t Offset) {
  // Create the relocation entry for the global value
  MachineRelocation MR =
    MachineRelocation::getGV(GblS.getCurrentPCOffset(),
                             TEW->getAbsoluteLabelMachineRelTy(),
                             const_cast<GlobalValue*>(GV),
                             Offset);

  // Fill the data entry with zeros
  GblS.emitZeros(Size);

  // Add the relocation entry for the current data section
  GblS.addRelocation(MR);
}

void ELFWriter::EmitGlobalConstantLargeInt(const ConstantInt *CI, 
                                           ELFSection &S) {
  const TargetData *TD = TM.getTargetData();
  unsigned BitWidth = CI->getBitWidth();
  assert(isPowerOf2_32(BitWidth) &&
         "Non-power-of-2-sized integers not handled!");

  const uint64_t *RawData = CI->getValue().getRawData();
  uint64_t Val = 0;
  for (unsigned i = 0, e = BitWidth / 64; i != e; ++i) {
    Val = (TD->isBigEndian()) ? RawData[e - i - 1] : RawData[i];
    S.emitWord64(Val);
  }
}

/// EmitSpecialLLVMGlobal - Check to see if the specified global is a
/// special global used by LLVM.  If so, emit it and return true, otherwise
/// do nothing and return false.
bool ELFWriter::EmitSpecialLLVMGlobal(const GlobalVariable *GV) {
  if (GV->getName() == "llvm.used")
    llvm_unreachable("not implemented yet");

  // Ignore debug and non-emitted data.  This handles llvm.compiler.used.
  if (GV->getSection() == "llvm.metadata" ||
      GV->hasAvailableExternallyLinkage())
    return true;
  
  if (!GV->hasAppendingLinkage()) return false;

  assert(GV->hasInitializer() && "Not a special LLVM global!");
  
  const TargetData *TD = TM.getTargetData();
  unsigned Align = TD->getPointerPrefAlignment();
  if (GV->getName() == "llvm.global_ctors") {
    ELFSection &Ctor = getCtorSection();
    Ctor.emitAlignment(Align);
    EmitXXStructorList(GV->getInitializer(), Ctor);
    return true;
  } 
  
  if (GV->getName() == "llvm.global_dtors") {
    ELFSection &Dtor = getDtorSection();
    Dtor.emitAlignment(Align);
    EmitXXStructorList(GV->getInitializer(), Dtor);
    return true;
  }
  
  return false;
}

/// EmitXXStructorList - Emit the ctor or dtor list.  This just emits out the 
/// function pointers, ignoring the init priority.
void ELFWriter::EmitXXStructorList(Constant *List, ELFSection &Xtor) {
  // Should be an array of '{ int, void ()* }' structs.  The first value is the
  // init priority, which we ignore.
  if (!isa<ConstantArray>(List)) return;
  ConstantArray *InitList = cast<ConstantArray>(List);
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))){
      if (CS->getNumOperands() != 2) return;  // Not array of 2-element structs.

      if (CS->getOperand(1)->isNullValue())
        return;  // Found a null terminator, exit printing.
      // Emit the function pointer.
      EmitGlobalConstant(CS->getOperand(1), Xtor);
    }
}

bool ELFWriter::runOnMachineFunction(MachineFunction &MF) {
  // Nothing to do here, this is all done through the ElfCE object above.
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the ELF file to 'O'.
bool ELFWriter::doFinalization(Module &M) {
  // Emit .data section placeholder
  getDataSection();

  // Emit .bss section placeholder
  getBSSSection();

  // Build and emit data, bss and "common" sections.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I);

  // Emit all pending globals
  for (PendingGblsIter I = PendingGlobals.begin(), E = PendingGlobals.end();
       I != E; ++I)
    EmitGlobal(*I);

  // Emit all pending externals
  for (PendingExtsIter I = PendingExternals.begin(), E = PendingExternals.end();
       I != E; ++I)
    SymbolList.push_back(ELFSym::getExtSym(*I));

  // Emit non-executable stack note
  if (TAI->getNonexecutableStackDirective())
    getNonExecStackSection();

  // Emit a symbol for each section created until now, skip null section
  for (unsigned i = 1, e = SectionList.size(); i < e; ++i) {
    ELFSection &ES = *SectionList[i];
    ELFSym *SectionSym = ELFSym::getSectionSym();
    SectionSym->SectionIdx = ES.SectionIdx;
    SymbolList.push_back(SectionSym);
    ES.Sym = SymbolList.back();
  }

  // Emit string table
  EmitStringTable(M.getModuleIdentifier());

  // Emit the symbol table now, if non-empty.
  EmitSymbolTable();

  // Emit the relocation sections.
  EmitRelocations();

  // Emit the sections string table.
  EmitSectionTableStringTable();

  // Dump the sections and section table to the .o file.
  OutputSectionsAndSectionTable();

  // We are done with the abstract symbols.
  SymbolList.clear();
  SectionList.clear();
  NumSections = 0;

  // Release the name mangler object.
  delete Mang; Mang = 0;
  return false;
}

// RelocateField - Patch relocatable field with 'Offset' in 'BO'
// using a 'Value' of known 'Size'
void ELFWriter::RelocateField(BinaryObject &BO, uint32_t Offset,
                              int64_t Value, unsigned Size) {
  if (Size == 32)
    BO.fixWord32(Value, Offset);
  else if (Size == 64)
    BO.fixWord64(Value, Offset);
  else
    llvm_unreachable("don't know howto patch relocatable field");
}

/// EmitRelocations - Emit relocations
void ELFWriter::EmitRelocations() {

  // True if the target uses the relocation entry to hold the addend,
  // otherwise the addend is written directly to the relocatable field.
  bool HasRelA = TEW->hasRelocationAddend();

  // Create Relocation sections for each section which needs it.
  for (unsigned i=0, e=SectionList.size(); i != e; ++i) {
    ELFSection &S = *SectionList[i];

    // This section does not have relocations
    if (!S.hasRelocations()) continue;
    ELFSection &RelSec = getRelocSection(S);

    // 'Link' - Section hdr idx of the associated symbol table
    // 'Info' - Section hdr idx of the section to which the relocation applies
    ELFSection &SymTab = getSymbolTableSection();
    RelSec.Link = SymTab.SectionIdx;
    RelSec.Info = S.SectionIdx;
    RelSec.EntSize = TEW->getRelocationEntrySize();

    // Get the relocations from Section
    std::vector<MachineRelocation> Relos = S.getRelocations();
    for (std::vector<MachineRelocation>::iterator MRI = Relos.begin(),
         MRE = Relos.end(); MRI != MRE; ++MRI) {
      MachineRelocation &MR = *MRI;

      // Relocatable field offset from the section start
      unsigned RelOffset = MR.getMachineCodeOffset();

      // Symbol index in the symbol table
      unsigned SymIdx = 0;

      // Target specific relocation field type and size
      unsigned RelType = TEW->getRelocationType(MR.getRelocationType());
      unsigned RelTySize = TEW->getRelocationTySize(RelType);
      int64_t Addend = 0;

      // There are several machine relocations types, and each one of
      // them needs a different approach to retrieve the symbol table index.
      if (MR.isGlobalValue()) {
        const GlobalValue *G = MR.getGlobalValue();
        int64_t GlobalOffset = MR.getConstantVal();
        SymIdx = GblSymLookup[G];
        if (G->hasPrivateLinkage()) {
          // If the target uses a section offset in the relocation:
          // SymIdx + Addend = section sym for global + section offset
          unsigned SectionIdx = PrivateSyms[SymIdx]->SectionIdx;
          Addend = PrivateSyms[SymIdx]->Value + GlobalOffset;
          SymIdx = SectionList[SectionIdx]->getSymbolTableIndex();
        } else {
          Addend = TEW->getDefaultAddendForRelTy(RelType, GlobalOffset);
        }
      } else if (MR.isExternalSymbol()) {
        const char *ExtSym = MR.getExternalSymbol();
        SymIdx = ExtSymLookup[ExtSym];
        Addend = TEW->getDefaultAddendForRelTy(RelType);
      } else {
        // Get the symbol index for the section symbol
        unsigned SectionIdx = MR.getConstantVal();
        SymIdx = SectionList[SectionIdx]->getSymbolTableIndex();

        // The symbol offset inside the section
        int64_t SymOffset = (int64_t)MR.getResultPointer();

        // For pc relative relocations where symbols are defined in the same
        // section they are referenced, ignore the relocation entry and patch
        // the relocatable field with the symbol offset directly.
        if (S.SectionIdx == SectionIdx && TEW->isPCRelativeRel(RelType)) {
          int64_t Value = TEW->computeRelocation(SymOffset, RelOffset, RelType);
          RelocateField(S, RelOffset, Value, RelTySize);
          continue;
        }

        Addend = TEW->getDefaultAddendForRelTy(RelType, SymOffset);
      }

      // The target without addend on the relocation symbol must be
      // patched in the relocation place itself to contain the addend
      // otherwise write zeros to make sure there is no garbage there
      RelocateField(S, RelOffset, HasRelA ? 0 : Addend, RelTySize);

      // Get the relocation entry and emit to the relocation section
      ELFRelocation Rel(RelOffset, SymIdx, RelType, HasRelA, Addend);
      EmitRelocation(RelSec, Rel, HasRelA);
    }
  }
}

/// EmitRelocation - Write relocation 'Rel' to the relocation section 'Rel'
void ELFWriter::EmitRelocation(BinaryObject &RelSec, ELFRelocation &Rel,
                               bool HasRelA) {
  RelSec.emitWord(Rel.getOffset());
  RelSec.emitWord(Rel.getInfo(is64Bit));
  if (HasRelA)
    RelSec.emitWord(Rel.getAddend());
}

/// EmitSymbol - Write symbol 'Sym' to the symbol table 'SymbolTable'
void ELFWriter::EmitSymbol(BinaryObject &SymbolTable, ELFSym &Sym) {
  if (is64Bit) {
    SymbolTable.emitWord32(Sym.NameIdx);
    SymbolTable.emitByte(Sym.Info);
    SymbolTable.emitByte(Sym.Other);
    SymbolTable.emitWord16(Sym.SectionIdx);
    SymbolTable.emitWord64(Sym.Value);
    SymbolTable.emitWord64(Sym.Size);
  } else {
    SymbolTable.emitWord32(Sym.NameIdx);
    SymbolTable.emitWord32(Sym.Value);
    SymbolTable.emitWord32(Sym.Size);
    SymbolTable.emitByte(Sym.Info);
    SymbolTable.emitByte(Sym.Other);
    SymbolTable.emitWord16(Sym.SectionIdx);
  }
}

/// EmitSectionHeader - Write section 'Section' header in 'SHdrTab'
/// Section Header Table
void ELFWriter::EmitSectionHeader(BinaryObject &SHdrTab,
                                  const ELFSection &SHdr) {
  SHdrTab.emitWord32(SHdr.NameIdx);
  SHdrTab.emitWord32(SHdr.Type);
  if (is64Bit) {
    SHdrTab.emitWord64(SHdr.Flags);
    SHdrTab.emitWord(SHdr.Addr);
    SHdrTab.emitWord(SHdr.Offset);
    SHdrTab.emitWord64(SHdr.Size);
    SHdrTab.emitWord32(SHdr.Link);
    SHdrTab.emitWord32(SHdr.Info);
    SHdrTab.emitWord64(SHdr.Align);
    SHdrTab.emitWord64(SHdr.EntSize);
  } else {
    SHdrTab.emitWord32(SHdr.Flags);
    SHdrTab.emitWord(SHdr.Addr);
    SHdrTab.emitWord(SHdr.Offset);
    SHdrTab.emitWord32(SHdr.Size);
    SHdrTab.emitWord32(SHdr.Link);
    SHdrTab.emitWord32(SHdr.Info);
    SHdrTab.emitWord32(SHdr.Align);
    SHdrTab.emitWord32(SHdr.EntSize);
  }
}

/// EmitStringTable - If the current symbol table is non-empty, emit the string
/// table for it
void ELFWriter::EmitStringTable(const std::string &ModuleName) {
  if (!SymbolList.size()) return;  // Empty symbol table.
  ELFSection &StrTab = getStringTableSection();

  // Set the zero'th symbol to a null byte, as required.
  StrTab.emitByte(0);

  // Walk on the symbol list and write symbol names into the string table.
  unsigned Index = 1;
  for (ELFSymIter I=SymbolList.begin(), E=SymbolList.end(); I != E; ++I) {
    ELFSym &Sym = *(*I);

    std::string Name;
    if (Sym.isGlobalValue())
      Name.append(Mang->getMangledName(Sym.getGlobalValue()));
    else if (Sym.isExternalSym())
      Name.append(Sym.getExternalSymbol());
    else if (Sym.isFileType())
      Name.append(ModuleName);

    if (Name.empty()) {
      Sym.NameIdx = 0;
    } else {
      Sym.NameIdx = Index;
      StrTab.emitString(Name);

      // Keep track of the number of bytes emitted to this section.
      Index += Name.size()+1;
    }
  }
  assert(Index == StrTab.size());
  StrTab.Size = Index;
}

// SortSymbols - On the symbol table local symbols must come before
// all other symbols with non-local bindings. The return value is
// the position of the first non local symbol.
unsigned ELFWriter::SortSymbols() {
  unsigned FirstNonLocalSymbol;
  std::vector<ELFSym*> LocalSyms, OtherSyms;

  for (ELFSymIter I=SymbolList.begin(), E=SymbolList.end(); I != E; ++I) {
    if ((*I)->isLocalBind())
      LocalSyms.push_back(*I);
    else
      OtherSyms.push_back(*I);
  }
  SymbolList.clear();
  FirstNonLocalSymbol = LocalSyms.size();

  for (unsigned i = 0; i < FirstNonLocalSymbol; ++i)
    SymbolList.push_back(LocalSyms[i]);

  for (ELFSymIter I=OtherSyms.begin(), E=OtherSyms.end(); I != E; ++I)
    SymbolList.push_back(*I);

  LocalSyms.clear();
  OtherSyms.clear();

  return FirstNonLocalSymbol;
}

/// EmitSymbolTable - Emit the symbol table itself.
void ELFWriter::EmitSymbolTable() {
  if (!SymbolList.size()) return;  // Empty symbol table.

  // Now that we have emitted the string table and know the offset into the
  // string table of each symbol, emit the symbol table itself.
  ELFSection &SymTab = getSymbolTableSection();
  SymTab.Align = TEW->getPrefELFAlignment();

  // Section Index of .strtab.
  SymTab.Link = getStringTableSection().SectionIdx;

  // Size of each symtab entry.
  SymTab.EntSize = TEW->getSymTabEntrySize();

  // Reorder the symbol table with local symbols first!
  unsigned FirstNonLocalSymbol = SortSymbols();

  // Emit all the symbols to the symbol table.
  for (unsigned i = 0, e = SymbolList.size(); i < e; ++i) {
    ELFSym &Sym = *SymbolList[i];

    // Emit symbol to the symbol table
    EmitSymbol(SymTab, Sym);

    // Record the symbol table index for each symbol
    if (Sym.isGlobalValue())
      GblSymLookup[Sym.getGlobalValue()] = i;
    else if (Sym.isExternalSym())
      ExtSymLookup[Sym.getExternalSymbol()] = i;

    // Keep track on the symbol index into the symbol table
    Sym.SymTabIdx = i;
  }

  // One greater than the symbol table index of the last local symbol
  SymTab.Info = FirstNonLocalSymbol;
  SymTab.Size = SymTab.size();
}

/// EmitSectionTableStringTable - This method adds and emits a section for the
/// ELF Section Table string table: the string table that holds all of the
/// section names.
void ELFWriter::EmitSectionTableStringTable() {
  // First step: add the section for the string table to the list of sections:
  ELFSection &SHStrTab = getSectionHeaderStringTableSection();

  // Now that we know which section number is the .shstrtab section, update the
  // e_shstrndx entry in the ELF header.
  ElfHdr.fixWord16(SHStrTab.SectionIdx, ELFHdr_e_shstrndx_Offset);

  // Set the NameIdx of each section in the string table and emit the bytes for
  // the string table.
  unsigned Index = 0;

  for (ELFSectionIter I=SectionList.begin(), E=SectionList.end(); I != E; ++I) {
    ELFSection &S = *(*I);
    // Set the index into the table.  Note if we have lots of entries with
    // common suffixes, we could memoize them here if we cared.
    S.NameIdx = Index;
    SHStrTab.emitString(S.getName());

    // Keep track of the number of bytes emitted to this section.
    Index += S.getName().size()+1;
  }

  // Set the size of .shstrtab now that we know what it is.
  assert(Index == SHStrTab.size());
  SHStrTab.Size = Index;
}

/// OutputSectionsAndSectionTable - Now that we have constructed the file header
/// and all of the sections, emit these to the ostream destination and emit the
/// SectionTable.
void ELFWriter::OutputSectionsAndSectionTable() {
  // Pass #1: Compute the file offset for each section.
  size_t FileOff = ElfHdr.size();   // File header first.

  // Adjust alignment of all section if needed, skip the null section.
  for (unsigned i=1, e=SectionList.size(); i < e; ++i) {
    ELFSection &ES = *SectionList[i];
    if (!ES.size()) {
      ES.Offset = FileOff;
      continue;
    }

    // Update Section size
    if (!ES.Size)
      ES.Size = ES.size();

    // Align FileOff to whatever the alignment restrictions of the section are.
    if (ES.Align)
      FileOff = (FileOff+ES.Align-1) & ~(ES.Align-1);

    ES.Offset = FileOff;
    FileOff += ES.Size;
  }

  // Align Section Header.
  unsigned TableAlign = TEW->getPrefELFAlignment();
  FileOff = (FileOff+TableAlign-1) & ~(TableAlign-1);

  // Now that we know where all of the sections will be emitted, set the e_shnum
  // entry in the ELF header.
  ElfHdr.fixWord16(NumSections, ELFHdr_e_shnum_Offset);

  // Now that we know the offset in the file of the section table, update the
  // e_shoff address in the ELF header.
  ElfHdr.fixWord(FileOff, ELFHdr_e_shoff_Offset);

  // Now that we know all of the data in the file header, emit it and all of the
  // sections!
  O.write((char *)&ElfHdr.getData()[0], ElfHdr.size());
  FileOff = ElfHdr.size();

  // Section Header Table blob
  BinaryObject SHdrTable(isLittleEndian, is64Bit);

  // Emit all of sections to the file and build the section header table.
  for (ELFSectionIter I=SectionList.begin(), E=SectionList.end(); I != E; ++I) {
    ELFSection &S = *(*I);
    DOUT << "SectionIdx: " << S.SectionIdx << ", Name: " << S.getName()
         << ", Size: " << S.Size << ", Offset: " << S.Offset
         << ", SectionData Size: " << S.size() << "\n";

    // Align FileOff to whatever the alignment restrictions of the section are.
    if (S.size()) {
      if (S.Align)  {
        for (size_t NewFileOff = (FileOff+S.Align-1) & ~(S.Align-1);
             FileOff != NewFileOff; ++FileOff)
          O << (char)0xAB;
      }
      O.write((char *)&S.getData()[0], S.Size);
      FileOff += S.Size;
    }

    EmitSectionHeader(SHdrTable, S);
  }

  // Align output for the section table.
  for (size_t NewFileOff = (FileOff+TableAlign-1) & ~(TableAlign-1);
       FileOff != NewFileOff; ++FileOff)
    O << (char)0xAB;

  // Emit the section table itself.
  O.write((char *)&SHdrTable.getData()[0], SHdrTable.size());
}
