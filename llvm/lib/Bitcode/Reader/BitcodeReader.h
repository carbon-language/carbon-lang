//===- BitcodeReader.h - Internal BitcodeReader impl ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitcodeReader class.
//
//===----------------------------------------------------------------------===//

#ifndef BITCODE_READER_H
#define BITCODE_READER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueHandle.h"
#include <system_error>
#include <vector>

namespace llvm {
  class Comdat;
  class MemoryBuffer;
  class LLVMContext;

//===----------------------------------------------------------------------===//
//                          BitcodeReaderValueList Class
//===----------------------------------------------------------------------===//

class BitcodeReaderValueList {
  std::vector<WeakVH> ValuePtrs;

  /// ResolveConstants - As we resolve forward-referenced constants, we add
  /// information about them to this vector.  This allows us to resolve them in
  /// bulk instead of resolving each reference at a time.  See the code in
  /// ResolveConstantForwardRefs for more information about this.
  ///
  /// The key of this vector is the placeholder constant, the value is the slot
  /// number that holds the resolved value.
  typedef std::vector<std::pair<Constant*, unsigned> > ResolveConstantsTy;
  ResolveConstantsTy ResolveConstants;
  LLVMContext &Context;
public:
  BitcodeReaderValueList(LLVMContext &C) : Context(C) {}
  ~BitcodeReaderValueList() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
  }

  // vector compatibility methods
  unsigned size() const { return ValuePtrs.size(); }
  void resize(unsigned N) { ValuePtrs.resize(N); }
  void push_back(Value *V) {
    ValuePtrs.push_back(V);
  }

  void clear() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
    ValuePtrs.clear();
  }

  Value *operator[](unsigned i) const {
    assert(i < ValuePtrs.size());
    return ValuePtrs[i];
  }

  Value *back() const { return ValuePtrs.back(); }
    void pop_back() { ValuePtrs.pop_back(); }
  bool empty() const { return ValuePtrs.empty(); }
  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    ValuePtrs.resize(N);
  }

  Constant *getConstantFwdRef(unsigned Idx, Type *Ty);
  Value *getValueFwdRef(unsigned Idx, Type *Ty);

  void AssignValue(Value *V, unsigned Idx);

  /// ResolveConstantForwardRefs - Once all constants are read, this method bulk
  /// resolves any forward references.
  void ResolveConstantForwardRefs();
};


//===----------------------------------------------------------------------===//
//                          BitcodeReaderMDValueList Class
//===----------------------------------------------------------------------===//

class BitcodeReaderMDValueList {
  std::vector<WeakVH> MDValuePtrs;

  LLVMContext &Context;
public:
  BitcodeReaderMDValueList(LLVMContext& C) : Context(C) {}

  // vector compatibility methods
  unsigned size() const       { return MDValuePtrs.size(); }
  void resize(unsigned N)     { MDValuePtrs.resize(N); }
  void push_back(Value *V)    { MDValuePtrs.push_back(V);  }
  void clear()                { MDValuePtrs.clear();  }
  Value *back() const         { return MDValuePtrs.back(); }
  void pop_back()             { MDValuePtrs.pop_back(); }
  bool empty() const          { return MDValuePtrs.empty(); }

  Value *operator[](unsigned i) const {
    assert(i < MDValuePtrs.size());
    return MDValuePtrs[i];
  }

  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    MDValuePtrs.resize(N);
  }

  Value *getValueFwdRef(unsigned Idx);
  void AssignValue(Value *V, unsigned Idx);
};

class BitcodeReader : public GVMaterializer {
  LLVMContext &Context;
  Module *TheModule;
  std::unique_ptr<MemoryBuffer> Buffer;
  std::unique_ptr<BitstreamReader> StreamFile;
  BitstreamCursor Stream;
  DataStreamer *LazyStreamer;
  uint64_t NextUnreadBit;
  bool SeenValueSymbolTable;

  std::vector<Type*> TypeList;
  BitcodeReaderValueList ValueList;
  BitcodeReaderMDValueList MDValueList;
  std::vector<Comdat *> ComdatList;
  SmallVector<Instruction *, 64> InstructionList;
  SmallVector<SmallVector<uint64_t, 64>, 64> UseListRecords;

  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInits;
  std::vector<std::pair<GlobalAlias*, unsigned> > AliasInits;
  std::vector<std::pair<Function*, unsigned> > FunctionPrefixes;

  SmallVector<Instruction*, 64> InstsWithTBAATag;

  /// MAttributes - The set of attributes by index.  Index zero in the
  /// file is for null, and is thus not represented here.  As such all indices
  /// are off by one.
  std::vector<AttributeSet> MAttributes;

  /// \brief The set of attribute groups.
  std::map<unsigned, AttributeSet> MAttributeGroups;

  /// FunctionBBs - While parsing a function body, this is a list of the basic
  /// blocks for the function.
  std::vector<BasicBlock*> FunctionBBs;

  // When reading the module header, this list is populated with functions that
  // have bodies later in the file.
  std::vector<Function*> FunctionsWithBodies;

  // When intrinsic functions are encountered which require upgrading they are
  // stored here with their replacement function.
  typedef std::vector<std::pair<Function*, Function*> > UpgradedIntrinsicMap;
  UpgradedIntrinsicMap UpgradedIntrinsics;

  // Map the bitcode's custom MDKind ID to the Module's MDKind ID.
  DenseMap<unsigned, unsigned> MDKindMap;

  // Several operations happen after the module header has been read, but
  // before function bodies are processed. This keeps track of whether
  // we've done this yet.
  bool SeenFirstFunctionBody;

  /// DeferredFunctionInfo - When function bodies are initially scanned, this
  /// map contains info about where to find deferred function body in the
  /// stream.
  DenseMap<Function*, uint64_t> DeferredFunctionInfo;

  /// BlockAddrFwdRefs - These are blockaddr references to basic blocks.  These
  /// are resolved lazily when functions are loaded.
  typedef std::pair<unsigned, GlobalVariable*> BlockAddrRefTy;
  DenseMap<Function*, std::vector<BlockAddrRefTy> > BlockAddrFwdRefs;

  /// UseRelativeIDs - Indicates that we are using a new encoding for
  /// instruction operands where most operands in the current
  /// FUNCTION_BLOCK are encoded relative to the instruction number,
  /// for a more compact encoding.  Some instruction operands are not
  /// relative to the instruction ID: basic block numbers, and types.
  /// Once the old style function blocks have been phased out, we would
  /// not need this flag.
  bool UseRelativeIDs;

  static const std::error_category &BitcodeErrorCategory();

public:
  enum ErrorType {
    BitcodeStreamInvalidSize,
    ConflictingMETADATA_KINDRecords,
    CouldNotFindFunctionInStream,
    ExpectedConstant,
    InsufficientFunctionProtos,
    InvalidBitcodeSignature,
    InvalidBitcodeWrapperHeader,
    InvalidConstantReference,
    InvalidID, // A read identifier is not found in the table it should be in.
    InvalidInstructionWithNoBB,
    InvalidRecord, // A read record doesn't have the expected size or structure
    InvalidTypeForValue, // Type read OK, but is invalid for its use
    InvalidTYPETable,
    InvalidType, // We were unable to read a type
    MalformedBlock, // We are unable to advance in the stream.
    MalformedGlobalInitializerSet,
    InvalidMultipleBlocks, // We found multiple blocks of a kind that should
                           // have only one
    NeverResolvedValueFoundInFunction,
    InvalidValue // Invalid version, inst number, attr number, etc
  };

  std::error_code Error(ErrorType E) {
    return std::error_code(E, BitcodeErrorCategory());
  }

  explicit BitcodeReader(MemoryBuffer *buffer, LLVMContext &C)
      : Context(C), TheModule(nullptr), Buffer(buffer), LazyStreamer(nullptr),
        NextUnreadBit(0), SeenValueSymbolTable(false), ValueList(C),
        MDValueList(C), SeenFirstFunctionBody(false), UseRelativeIDs(false) {}
  explicit BitcodeReader(DataStreamer *streamer, LLVMContext &C)
      : Context(C), TheModule(nullptr), Buffer(nullptr), LazyStreamer(streamer),
        NextUnreadBit(0), SeenValueSymbolTable(false), ValueList(C),
        MDValueList(C), SeenFirstFunctionBody(false), UseRelativeIDs(false) {}
  ~BitcodeReader() { FreeState(); }

  void materializeForwardReferencedFunctions();

  void FreeState();

  void releaseBuffer() override;

  bool isMaterializable(const GlobalValue *GV) const override;
  bool isDematerializable(const GlobalValue *GV) const override;
  std::error_code Materialize(GlobalValue *GV) override;
  std::error_code MaterializeModule(Module *M) override;
  void Dematerialize(GlobalValue *GV) override;

  /// @brief Main interface to parsing a bitcode buffer.
  /// @returns true if an error occurred.
  std::error_code ParseBitcodeInto(Module *M);

  /// @brief Cheap mechanism to just extract module triple
  /// @returns true if an error occurred.
  ErrorOr<std::string> parseTriple();

  static uint64_t decodeSignRotatedValue(uint64_t V);

private:
  Type *getTypeByID(unsigned ID);
  Value *getFnValueByID(unsigned ID, Type *Ty) {
    if (Ty && Ty->isMetadataTy())
      return MDValueList.getValueFwdRef(ID);
    return ValueList.getValueFwdRef(ID, Ty);
  }
  BasicBlock *getBasicBlock(unsigned ID) const {
    if (ID >= FunctionBBs.size()) return nullptr; // Invalid ID
    return FunctionBBs[ID];
  }
  AttributeSet getAttributes(unsigned i) const {
    if (i-1 < MAttributes.size())
      return MAttributes[i-1];
    return AttributeSet();
  }

  /// getValueTypePair - Read a value/type pair out of the specified record from
  /// slot 'Slot'.  Increment Slot past the number of slots used in the record.
  /// Return true on failure.
  bool getValueTypePair(SmallVectorImpl<uint64_t> &Record, unsigned &Slot,
                        unsigned InstNum, Value *&ResVal) {
    if (Slot == Record.size()) return true;
    unsigned ValNo = (unsigned)Record[Slot++];
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    if (ValNo < InstNum) {
      // If this is not a forward reference, just return the value we already
      // have.
      ResVal = getFnValueByID(ValNo, nullptr);
      return ResVal == nullptr;
    } else if (Slot == Record.size()) {
      return true;
    }

    unsigned TypeNo = (unsigned)Record[Slot++];
    ResVal = getFnValueByID(ValNo, getTypeByID(TypeNo));
    return ResVal == nullptr;
  }

  /// popValue - Read a value out of the specified record from slot 'Slot'.
  /// Increment Slot past the number of slots used by the value in the record.
  /// Return true if there is an error.
  bool popValue(SmallVectorImpl<uint64_t> &Record, unsigned &Slot,
                unsigned InstNum, Type *Ty, Value *&ResVal) {
    if (getValue(Record, Slot, InstNum, Ty, ResVal))
      return true;
    // All values currently take a single record slot.
    ++Slot;
    return false;
  }

  /// getValue -- Like popValue, but does not increment the Slot number.
  bool getValue(SmallVectorImpl<uint64_t> &Record, unsigned Slot,
                unsigned InstNum, Type *Ty, Value *&ResVal) {
    ResVal = getValue(Record, Slot, InstNum, Ty);
    return ResVal == nullptr;
  }

  /// getValue -- Version of getValue that returns ResVal directly,
  /// or 0 if there is an error.
  Value *getValue(SmallVectorImpl<uint64_t> &Record, unsigned Slot,
                  unsigned InstNum, Type *Ty) {
    if (Slot == Record.size()) return nullptr;
    unsigned ValNo = (unsigned)Record[Slot];
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    return getFnValueByID(ValNo, Ty);
  }

  /// getValueSigned -- Like getValue, but decodes signed VBRs.
  Value *getValueSigned(SmallVectorImpl<uint64_t> &Record, unsigned Slot,
                        unsigned InstNum, Type *Ty) {
    if (Slot == Record.size()) return nullptr;
    unsigned ValNo = (unsigned)decodeSignRotatedValue(Record[Slot]);
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    return getFnValueByID(ValNo, Ty);
  }

  std::error_code ParseAttrKind(uint64_t Code, Attribute::AttrKind *Kind);
  std::error_code ParseModule(bool Resume);
  std::error_code ParseAttributeBlock();
  std::error_code ParseAttributeGroupBlock();
  std::error_code ParseTypeTable();
  std::error_code ParseTypeTableBody();

  std::error_code ParseValueSymbolTable();
  std::error_code ParseConstants();
  std::error_code RememberAndSkipFunctionBody();
  std::error_code ParseFunctionBody(Function *F);
  std::error_code GlobalCleanup();
  std::error_code ResolveGlobalAndAliasInits();
  std::error_code ParseMetadata();
  std::error_code ParseMetadataAttachment();
  ErrorOr<std::string> parseModuleTriple();
  std::error_code ParseUseLists();
  std::error_code InitStream();
  std::error_code InitStreamFromBuffer();
  std::error_code InitLazyStream();
  std::error_code FindFunctionInStream(
      Function *F,
      DenseMap<Function *, uint64_t>::iterator DeferredFunctionInfoIterator);
};

} // End llvm namespace

#endif
