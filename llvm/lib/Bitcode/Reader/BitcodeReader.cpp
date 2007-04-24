//===- BitcodeReader.cpp - Internal BitcodeReader implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License.  See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitcodeReader class.
//
//===----------------------------------------------------------------------===//

#include "BitcodeReader.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// ConvertToString - Convert a string from a record into an std::string, return
/// true on failure.
template<typename StrTy>
static bool ConvertToString(SmallVector<uint64_t, 64> &Record, unsigned Idx,
                            StrTy &Result) {
  if (Record.size() < Idx+1 || Record.size() < Record[Idx]+Idx+1)
    return true;
  
  for (unsigned i = 0, e = Record[Idx]; i != e; ++i)
    Result += (char)Record[Idx+i+1];
  return false;
}

static GlobalValue::LinkageTypes GetDecodedLinkage(unsigned Val) {
  switch (Val) {
  default: // Map unknown/new linkages to external
  case 0: return GlobalValue::ExternalLinkage;
  case 1: return GlobalValue::WeakLinkage;
  case 2: return GlobalValue::AppendingLinkage;
  case 3: return GlobalValue::InternalLinkage;
  case 4: return GlobalValue::LinkOnceLinkage;
  case 5: return GlobalValue::DLLImportLinkage;
  case 6: return GlobalValue::DLLExportLinkage;
  case 7: return GlobalValue::ExternalWeakLinkage;
  }
}

static GlobalValue::VisibilityTypes GetDecodedVisibility(unsigned Val) {
  switch (Val) {
  default: // Map unknown visibilities to default.
  case 0: return GlobalValue::DefaultVisibility;
  case 1: return GlobalValue::HiddenVisibility;
  }
}

namespace {
  /// @brief A class for maintaining the slot number definition
  /// as a placeholder for the actual definition for forward constants defs.
  class ConstantPlaceHolder : public ConstantExpr {
    ConstantPlaceHolder();                       // DO NOT IMPLEMENT
    void operator=(const ConstantPlaceHolder &); // DO NOT IMPLEMENT
public:
  Use Op;
  ConstantPlaceHolder(const Type *Ty)
    : ConstantExpr(Ty, Instruction::UserOp1, &Op, 1),
      Op(UndefValue::get(Type::Int32Ty), this) {
    }
  };
}

Constant *BitcodeReaderValueList::getConstantFwdRef(unsigned Idx,
                                                    const Type *Ty) {
  if (Idx >= size()) {
    // Insert a bunch of null values.
    Uses.resize(Idx+1);
    OperandList = &Uses[0];
    NumOperands = Idx+1;
  }

  if (Uses[Idx])
    return cast<Constant>(getOperand(Idx));

  // Create and return a placeholder, which will later be RAUW'd.
  Constant *C = new ConstantPlaceHolder(Ty);
  Uses[Idx].init(C, this);
  return C;
}


const Type *BitcodeReader::getTypeByID(unsigned ID, bool isTypeTable) {
  // If the TypeID is in range, return it.
  if (ID < TypeList.size())
    return TypeList[ID].get();
  if (!isTypeTable) return 0;
  
  // The type table allows forward references.  Push as many Opaque types as
  // needed to get up to ID.
  while (TypeList.size() <= ID)
    TypeList.push_back(OpaqueType::get());
  return TypeList.back().get();
}


bool BitcodeReader::ParseTypeTable(BitstreamReader &Stream) {
  if (Stream.EnterSubBlock())
    return Error("Malformed block record");
  
  if (!TypeList.empty())
    return Error("Multiple TYPE_BLOCKs found!");

  SmallVector<uint64_t, 64> Record;
  unsigned NumRecords = 0;

  // Read all the records for this type table.
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (NumRecords != TypeList.size())
        return Error("Invalid type forward reference in TYPE_BLOCK");
      return Stream.ReadBlockEnd();
    }
    
    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }
    
    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    Record.clear();
    const Type *ResultTy = 0;
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: unknown type.
      ResultTy = 0;
      break;
    case bitc::TYPE_CODE_NUMENTRY: // TYPE_CODE_NUMENTRY: [numentries]
      // TYPE_CODE_NUMENTRY contains a count of the number of types in the
      // type list.  This allows us to reserve space.
      if (Record.size() < 1)
        return Error("Invalid TYPE_CODE_NUMENTRY record");
      TypeList.reserve(Record[0]);
      continue;
    case bitc::TYPE_CODE_META:      // TYPE_CODE_META: [metacode]...
      // No metadata supported yet.
      if (Record.size() < 1)
        return Error("Invalid TYPE_CODE_META record");
      continue;
      
    case bitc::TYPE_CODE_VOID:      // VOID
      ResultTy = Type::VoidTy;
      break;
    case bitc::TYPE_CODE_FLOAT:     // FLOAT
      ResultTy = Type::FloatTy;
      break;
    case bitc::TYPE_CODE_DOUBLE:    // DOUBLE
      ResultTy = Type::DoubleTy;
      break;
    case bitc::TYPE_CODE_LABEL:     // LABEL
      ResultTy = Type::LabelTy;
      break;
    case bitc::TYPE_CODE_OPAQUE:    // OPAQUE
      ResultTy = 0;
      break;
    case bitc::TYPE_CODE_INTEGER:   // INTEGER: [width]
      if (Record.size() < 1)
        return Error("Invalid Integer type record");
      
      ResultTy = IntegerType::get(Record[0]);
      break;
    case bitc::TYPE_CODE_POINTER:   // POINTER: [pointee type]
      if (Record.size() < 1)
        return Error("Invalid POINTER type record");
      ResultTy = PointerType::get(getTypeByID(Record[0], true));
      break;
    case bitc::TYPE_CODE_FUNCTION: {
      // FUNCTION: [vararg, retty, #pararms, paramty N]
      if (Record.size() < 3 || Record.size() < Record[2]+3)
        return Error("Invalid FUNCTION type record");
      std::vector<const Type*> ArgTys;
      for (unsigned i = 0, e = Record[2]; i != e; ++i)
        ArgTys.push_back(getTypeByID(Record[3+i], true));
      
      // FIXME: PARAM TYS.
      ResultTy = FunctionType::get(getTypeByID(Record[1], true), ArgTys,
                                   Record[0]);
      break;
    }
    case bitc::TYPE_CODE_STRUCT: {  // STRUCT: [ispacked, #elts, eltty x N]
      if (Record.size() < 2 || Record.size() < Record[1]+2)
        return Error("Invalid STRUCT type record");
      std::vector<const Type*> EltTys;
      for (unsigned i = 0, e = Record[1]; i != e; ++i)
        EltTys.push_back(getTypeByID(Record[2+i], true));
      ResultTy = StructType::get(EltTys, Record[0]);
      break;
    }
    case bitc::TYPE_CODE_ARRAY:     // ARRAY: [numelts, eltty]
      if (Record.size() < 2)
        return Error("Invalid ARRAY type record");
      ResultTy = ArrayType::get(getTypeByID(Record[1], true), Record[0]);
      break;
    case bitc::TYPE_CODE_VECTOR:    // VECTOR: [numelts, eltty]
      if (Record.size() < 2)
        return Error("Invalid VECTOR type record");
      ResultTy = VectorType::get(getTypeByID(Record[1], true), Record[0]);
      break;
    }
    
    if (NumRecords == TypeList.size()) {
      // If this is a new type slot, just append it.
      TypeList.push_back(ResultTy ? ResultTy : OpaqueType::get());
      ++NumRecords;
    } else if (ResultTy == 0) {
      // Otherwise, this was forward referenced, so an opaque type was created,
      // but the result type is actually just an opaque.  Leave the one we
      // created previously.
      ++NumRecords;
    } else {
      // Otherwise, this was forward referenced, so an opaque type was created.
      // Resolve the opaque type to the real type now.
      assert(NumRecords < TypeList.size() && "Typelist imbalance");
      const OpaqueType *OldTy = cast<OpaqueType>(TypeList[NumRecords++].get());
     
      // Don't directly push the new type on the Tab. Instead we want to replace
      // the opaque type we previously inserted with the new concrete value. The
      // refinement from the abstract (opaque) type to the new type causes all
      // uses of the abstract type to use the concrete type (NewTy). This will
      // also cause the opaque type to be deleted.
      const_cast<OpaqueType*>(OldTy)->refineAbstractTypeTo(ResultTy);
      
      // This should have replaced the old opaque type with the new type in the
      // value table... or with a preexisting type that was already in the
      // system.  Let's just make sure it did.
      assert(TypeList[NumRecords-1].get() != OldTy &&
             "refineAbstractType didn't work!");
    }
  }
}


bool BitcodeReader::ParseTypeSymbolTable(BitstreamReader &Stream) {
  if (Stream.EnterSubBlock())
    return Error("Malformed block record");
  
  SmallVector<uint64_t, 64> Record;
  
  // Read all the records for this type table.
  std::string TypeName;
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK)
      return Stream.ReadBlockEnd();
    
    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }
    
    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: unknown type.
      break;
    case bitc::TST_CODE_ENTRY:    // TST_ENTRY: [typeid, namelen, namechar x N]
      if (ConvertToString(Record, 1, TypeName))
        return Error("Invalid TST_ENTRY record");
      unsigned TypeID = Record[0];
      if (TypeID >= TypeList.size())
        return Error("Invalid Type ID in TST_ENTRY record");

      TheModule->addTypeName(TypeName, TypeList[TypeID].get());
      TypeName.clear();
      break;
    }
  }
}

bool BitcodeReader::ParseValueSymbolTable(BitstreamReader &Stream) {
  if (Stream.EnterSubBlock())
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;
  
  // Read all the records for this value table.
  SmallString<128> ValueName;
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK)
      return Stream.ReadBlockEnd();
    
    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }
    
    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: unknown type.
      break;
    case bitc::TST_CODE_ENTRY:    // VST_ENTRY: [valueid, namelen, namechar x N]
      if (ConvertToString(Record, 1, ValueName))
        return Error("Invalid TST_ENTRY record");
      unsigned ValueID = Record[0];
      if (ValueID >= ValueList.size())
        return Error("Invalid Value ID in VST_ENTRY record");
      Value *V = ValueList[ValueID];
      
      V->setName(&ValueName[0], ValueName.size());
      ValueName.clear();
      break;
    }
  }
}

/// DecodeSignRotatedValue - Decode a signed value stored with the sign bit in
/// the LSB for dense VBR encoding.
static uint64_t DecodeSignRotatedValue(uint64_t V) {
  if ((V & 1) == 0)
    return V >> 1;
  if (V != 1) 
    return -(V >> 1);
  // There is no such thing as -0 with integers.  "-0" really means MININT.
  return 1ULL << 63;
}

bool BitcodeReader::ParseConstants(BitstreamReader &Stream) {
  if (Stream.EnterSubBlock())
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;
  
  // Read all the records for this value table.
  const Type *CurTy = Type::Int32Ty;
  unsigned NextCstNo = ValueList.size();
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      // If there are global var inits to process, do so now.
      if (!GlobalInits.empty()) {
        while (!GlobalInits.empty()) {
          unsigned ValID = GlobalInits.back().second;
          if (ValID >= ValueList.size())
            return Error("Invalid value ID for global var init!");
          if (Constant *C = dyn_cast<Constant>(ValueList[ValID]))
            GlobalInits.back().first->setInitializer(C);
          else
            return Error("Global variable initializer is not a constant!");
          GlobalInits.pop_back(); 
        }
      }
      
      if (NextCstNo != ValueList.size())
        return Error("Invalid constant reference!");
      
      return Stream.ReadBlockEnd();
    }
    
    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }
    
    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    Record.clear();
    Value *V = 0;
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: unknown constant
    case bitc::CST_CODE_UNDEF:     // UNDEF
      V = UndefValue::get(CurTy);
      break;
    case bitc::CST_CODE_SETTYPE:   // SETTYPE: [typeid]
      if (Record.empty())
        return Error("Malformed CST_SETTYPE record");
      if (Record[0] >= TypeList.size())
        return Error("Invalid Type ID in CST_SETTYPE record");
      CurTy = TypeList[Record[0]];
      continue;  // Skip the ValueList manipulation.
    case bitc::CST_CODE_NULL:      // NULL
      V = Constant::getNullValue(CurTy);
      break;
    case bitc::CST_CODE_INTEGER:   // INTEGER: [intval]
      if (!isa<IntegerType>(CurTy) || Record.empty())
        return Error("Invalid CST_INTEGER record");
      V = ConstantInt::get(CurTy, DecodeSignRotatedValue(Record[0]));
      break;
    case bitc::CST_CODE_WIDE_INTEGER: {// WIDE_INTEGER: [n, n x intval]
      if (!isa<IntegerType>(CurTy) || Record.empty() ||
          Record.size() < Record[0]+1)
        return Error("Invalid WIDE_INTEGER record");
      
      unsigned NumWords = Record[0];
      uint64_t *Data = new uint64_t[NumWords];
      for (unsigned i = 0; i != NumWords; ++i)
        Data[i] = DecodeSignRotatedValue(Record[i+1]);
      V = ConstantInt::get(APInt(cast<IntegerType>(CurTy)->getBitWidth(),
                                 NumWords, Data));
      break;
    }
    case bitc::CST_CODE_FLOAT:     // FLOAT: [fpval]
      if (Record.empty())
        return Error("Invalid FLOAT record");
      if (CurTy == Type::FloatTy)
        V = ConstantFP::get(CurTy, BitsToFloat(Record[0]));
      else if (CurTy == Type::DoubleTy)
        V = ConstantFP::get(CurTy, BitsToDouble(Record[0]));
      else
        V = UndefValue::get(CurTy);
      break;
      
    case bitc::CST_CODE_AGGREGATE: {// AGGREGATE: [n, n x value number]
      if (Record.empty() || Record.size() < Record[0]+1)
        return Error("Invalid CST_AGGREGATE record");
      
      unsigned Size = Record[0];
      std::vector<Constant*> Elts;
      
      if (const StructType *STy = dyn_cast<StructType>(CurTy)) {
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i+1],
                                                     STy->getElementType(i)));
        V = ConstantStruct::get(STy, Elts);
      } else if (const ArrayType *ATy = dyn_cast<ArrayType>(CurTy)) {
        const Type *EltTy = ATy->getElementType();
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i+1], EltTy));
        V = ConstantArray::get(ATy, Elts);
      } else if (const VectorType *VTy = dyn_cast<VectorType>(CurTy)) {
        const Type *EltTy = VTy->getElementType();
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i+1], EltTy));
        V = ConstantVector::get(Elts);
      } else {
        V = UndefValue::get(CurTy);
      }
    }
    }
    
    if (NextCstNo == ValueList.size())
      ValueList.push_back(V);
    else if (ValueList[NextCstNo] == 0)
      ValueList.initVal(NextCstNo, V);
    else {
      // If there was a forward reference to this constant, 
      Value *OldV = ValueList[NextCstNo];
      ValueList.setOperand(NextCstNo, V);
      OldV->replaceAllUsesWith(V);
      delete OldV;
    }
    
    ++NextCstNo;
  }
}

bool BitcodeReader::ParseModule(BitstreamReader &Stream,
                                const std::string &ModuleID) {
  // Reject multiple MODULE_BLOCK's in a single bitstream.
  if (TheModule)
    return Error("Multiple MODULE_BLOCKs in same stream");
  
  if (Stream.EnterSubBlock())
    return Error("Malformed block record");

  // Otherwise, create the module.
  TheModule = new Module(ModuleID);
  
  SmallVector<uint64_t, 64> Record;
  std::vector<std::string> SectionTable;

  // Read all the records for this module.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (!GlobalInits.empty())
        return Error("Malformed global initializer set");
      return Stream.ReadBlockEnd();
    }
    
    if (Code == bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      default:  // Skip unknown content.
        if (Stream.SkipBlock())
          return Error("Malformed block record");
        break;
      case bitc::TYPE_BLOCK_ID:
        if (ParseTypeTable(Stream))
          return true;
        break;
      case bitc::TYPE_SYMTAB_BLOCK_ID:
        if (ParseTypeSymbolTable(Stream))
          return true;
        break;
      case bitc::VALUE_SYMTAB_BLOCK_ID:
        if (ParseValueSymbolTable(Stream))
          return true;
        break;
      case bitc::CONSTANTS_BLOCK_ID:
        if (ParseConstants(Stream))
          return true;
        break;
      }
      continue;
    }
    
    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    switch (Stream.ReadRecord(Code, Record)) {
    default: break;  // Default behavior, ignore unknown content.
    case bitc::MODULE_CODE_VERSION:  // VERSION: [version#]
      if (Record.size() < 1)
        return Error("Malformed MODULE_CODE_VERSION");
      // Only version #0 is supported so far.
      if (Record[0] != 0)
        return Error("Unknown bitstream version!");
      break;
    case bitc::MODULE_CODE_TRIPLE: {  // TRIPLE: [strlen, strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_TRIPLE record");
      TheModule->setTargetTriple(S);
      break;
    }
    case bitc::MODULE_CODE_DATALAYOUT: {  // DATALAYOUT: [strlen, strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_DATALAYOUT record");
      TheModule->setDataLayout(S);
      break;
    }
    case bitc::MODULE_CODE_ASM: {  // ASM: [strlen, strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_ASM record");
      TheModule->setModuleInlineAsm(S);
      break;
    }
    case bitc::MODULE_CODE_DEPLIB: {  // DEPLIB: [strlen, strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_DEPLIB record");
      TheModule->addLibrary(S);
      break;
    }
    case bitc::MODULE_CODE_SECTIONNAME: {  // SECTIONNAME: [strlen, strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_SECTIONNAME record");
      SectionTable.push_back(S);
      break;
    }
    // GLOBALVAR: [type, isconst, initid, 
    //             linkage, alignment, section, visibility, threadlocal]
    case bitc::MODULE_CODE_GLOBALVAR: {
      if (Record.size() < 6)
        return Error("Invalid MODULE_CODE_GLOBALVAR record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!isa<PointerType>(Ty))
        return Error("Global not a pointer type!");
      Ty = cast<PointerType>(Ty)->getElementType();
      
      bool isConstant = Record[1];
      GlobalValue::LinkageTypes Linkage = GetDecodedLinkage(Record[3]);
      unsigned Alignment = (1 << Record[4]) >> 1;
      std::string Section;
      if (Record[5]) {
        if (Record[5]-1 >= SectionTable.size())
          return Error("Invalid section ID");
        Section = SectionTable[Record[5]-1];
      }
      GlobalValue::VisibilityTypes Visibility = GlobalValue::DefaultVisibility;
      if (Record.size() >= 6) Visibility = GetDecodedVisibility(Record[6]);
      bool isThreadLocal = false;
      if (Record.size() >= 7) isThreadLocal = Record[7];

      GlobalVariable *NewGV =
        new GlobalVariable(Ty, isConstant, Linkage, 0, "", TheModule);
      NewGV->setAlignment(Alignment);
      if (!Section.empty())
        NewGV->setSection(Section);
      NewGV->setVisibility(Visibility);
      NewGV->setThreadLocal(isThreadLocal);
      
      ValueList.push_back(NewGV);
      
      // Remember which value to use for the global initializer.
      if (unsigned InitID = Record[2])
        GlobalInits.push_back(std::make_pair(NewGV, InitID-1));
      break;
    }
    // FUNCTION:  [type, callingconv, isproto, linkage, alignment, section,
    //             visibility]
    case bitc::MODULE_CODE_FUNCTION: {
      if (Record.size() < 7)
        return Error("Invalid MODULE_CODE_FUNCTION record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!isa<PointerType>(Ty))
        return Error("Function not a pointer type!");
      const FunctionType *FTy =
        dyn_cast<FunctionType>(cast<PointerType>(Ty)->getElementType());
      if (!FTy)
        return Error("Function not a pointer to function type!");

      Function *Func = new Function(FTy, GlobalValue::ExternalLinkage,
                                    "", TheModule);

      Func->setCallingConv(Record[1]);
      Func->setLinkage(GetDecodedLinkage(Record[3]));
      Func->setAlignment((1 << Record[4]) >> 1);
      if (Record[5]) {
        if (Record[5]-1 >= SectionTable.size())
          return Error("Invalid section ID");
        Func->setSection(SectionTable[Record[5]-1]);
      }
      Func->setVisibility(GetDecodedVisibility(Record[6]));
      
      ValueList.push_back(Func);
      // TODO: remember initializer/global pair for later substitution.
      break;
    }
    }
    Record.clear();
  }
  
  return Error("Premature end of bitstream");
}


bool BitcodeReader::ParseBitcode(unsigned char *Buf, unsigned Length,
                                 const std::string &ModuleID) {
  TheModule = 0;
  
  if (Length & 3)
    return Error("Bitcode stream should be a multiple of 4 bytes in length");
  
  BitstreamReader Stream(Buf, Buf+Length);
  
  // Sniff for the signature.
  if (Stream.Read(8) != 'B' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(4) != 0x0 ||
      Stream.Read(4) != 0xC ||
      Stream.Read(4) != 0xE ||
      Stream.Read(4) != 0xD)
    return Error("Invalid bitcode signature");
  
  // We expect a number of well-defined blocks, though we don't necessarily
  // need to understand them all.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    
    if (Code != bitc::ENTER_SUBBLOCK)
      return Error("Invalid record at top-level");
    
    unsigned BlockID = Stream.ReadSubBlockID();
    
    // We only know the MODULE subblock ID.
    if (BlockID == bitc::MODULE_BLOCK_ID) {
      if (ParseModule(Stream, ModuleID))
        return true;
    } else if (Stream.SkipBlock()) {
      return Error("Malformed block record");
    }
  }
  
  return false;
}
