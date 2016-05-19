//=-- Profilesummary.cpp - Profile summary support --------------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for converting profile summary data from/to
// metadata.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ProfileSummary.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

const char *ProfileSummary::KindStr[2] = {"InstrProf", "SampleProfile"};

// Return an MDTuple with two elements. The first element is a string Key and
// the second is a uint64_t Value.
static Metadata *getKeyValMD(LLVMContext &Context, const char *Key,
                             uint64_t Val) {
  Type *Int64Ty = Type::getInt64Ty(Context);
  Metadata *Ops[2] = {MDString::get(Context, Key),
                      ConstantAsMetadata::get(ConstantInt::get(Int64Ty, Val))};
  return MDTuple::get(Context, Ops);
}

// Return an MDTuple with two elements. The first element is a string Key and
// the second is a string Value.
static Metadata *getKeyValMD(LLVMContext &Context, const char *Key,
                             const char *Val) {
  Metadata *Ops[2] = {MDString::get(Context, Key), MDString::get(Context, Val)};
  return MDTuple::get(Context, Ops);
}

// This returns an MDTuple representing the detiled summary. The tuple has two
// elements: a string "DetailedSummary" and an MDTuple representing the value
// of the detailed summary. Each element of this tuple is again an MDTuple whose
// elements are the (Cutoff, MinCount, NumCounts) triplet of the
// DetailedSummaryEntry.
Metadata *ProfileSummary::getDetailedSummaryMD(LLVMContext &Context) {
  std::vector<Metadata *> Entries;
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int64Ty = Type::getInt64Ty(Context);
  for (auto &Entry : DetailedSummary) {
    Metadata *EntryMD[3] = {
        ConstantAsMetadata::get(ConstantInt::get(Int32Ty, Entry.Cutoff)),
        ConstantAsMetadata::get(ConstantInt::get(Int64Ty, Entry.MinCount)),
        ConstantAsMetadata::get(ConstantInt::get(Int32Ty, Entry.NumCounts))};
    Entries.push_back(MDTuple::get(Context, EntryMD));
  }
  Metadata *Ops[2] = {MDString::get(Context, "DetailedSummary"),
                      MDTuple::get(Context, Entries)};
  return MDTuple::get(Context, Ops);
}

// This returns an MDTuple representing this ProfileSummary object. The first
// entry of this tuple is another MDTuple of two elements: a string
// "ProfileFormat" and a string representing the format ("InstrProf" or
// "SampleProfile"). The rest of the elements of the outer MDTuple are specific
// to the kind of profile summary as returned by getFormatSpecificMD.
Metadata *ProfileSummary::getMD(LLVMContext &Context) {
  std::vector<Metadata *> Components;
  Components.push_back(getKeyValMD(Context, "ProfileFormat", getKindStr()));
  std::vector<Metadata *> Res = getFormatSpecificMD(Context);
  Components.insert(Components.end(), Res.begin(), Res.end());
  return MDTuple::get(Context, Components);
}

// Returns a vector of MDTuples specific to InstrProfSummary. The first six
// elements of this vector are (Key, Val) pairs of the six scalar fields of
// InstrProfSummary (TotalCount, MaxBlockCount, MaxInternalBlockCount,
// MaxFunctionCount, NumBlocks, NumFunctions). The last element of this vector
// is an MDTuple returned by getDetailedSummaryMD.
std::vector<Metadata *>
InstrProfSummary::getFormatSpecificMD(LLVMContext &Context) {
  std::vector<Metadata *> Components;

  Components.push_back(getKeyValMD(Context, "TotalCount", getTotalCount()));
  Components.push_back(
      getKeyValMD(Context, "MaxBlockCount", getMaxBlockCount()));
  Components.push_back(getKeyValMD(Context, "MaxInternalBlockCount",
                                   getMaxInternalBlockCount()));
  Components.push_back(
      getKeyValMD(Context, "MaxFunctionCount", getMaxFunctionCount()));
  Components.push_back(getKeyValMD(Context, "NumBlocks", getNumBlocks()));
  Components.push_back(getKeyValMD(Context, "NumFunctions", getNumFunctions()));

  Components.push_back(getDetailedSummaryMD(Context));
  return Components;
}

std::vector<Metadata *>
SampleProfileSummary::getFormatSpecificMD(LLVMContext &Context) {
  std::vector<Metadata *> Components;

  Components.push_back(getKeyValMD(Context, "TotalSamples", getTotalSamples()));
  Components.push_back(
      getKeyValMD(Context, "MaxSamplesPerLine", getMaxSamplesPerLine()));
  Components.push_back(
      getKeyValMD(Context, "MaxFunctionCount", getMaxFunctionCount()));
  Components.push_back(
      getKeyValMD(Context, "NumLinesWithSamples", getNumLinesWithSamples()));
  Components.push_back(getKeyValMD(Context, "NumFunctions", NumFunctions));

  Components.push_back(getDetailedSummaryMD(Context));
  return Components;
}

// Parse an MDTuple representing (Key, Val) pair.
static bool getVal(MDTuple *MD, const char *Key, uint64_t &Val) {
  if (!MD)
    return false;
  if (MD->getNumOperands() != 2)
    return false;
  MDString *KeyMD = dyn_cast<MDString>(MD->getOperand(0));
  ConstantAsMetadata *ValMD = dyn_cast<ConstantAsMetadata>(MD->getOperand(1));
  if (!KeyMD || !ValMD)
    return false;
  if (!KeyMD->getString().equals(Key))
    return false;
  Val = cast<ConstantInt>(ValMD->getValue())->getZExtValue();
  return true;
}

// Check if an MDTuple represents a (Key, Val) pair.
static bool isKeyValuePair(MDTuple *MD, const char *Key, const char *Val) {
  if (!MD || MD->getNumOperands() != 2)
    return false;
  MDString *KeyMD = dyn_cast<MDString>(MD->getOperand(0));
  MDString *ValMD = dyn_cast<MDString>(MD->getOperand(1));
  if (!KeyMD || !ValMD)
    return false;
  if (!KeyMD->getString().equals(Key) || !ValMD->getString().equals(Val))
    return false;
  return true;
}

// Parse an MDTuple representing detailed summary.
static bool getSummaryFromMD(MDTuple *MD, SummaryEntryVector &Summary) {
  if (!MD || MD->getNumOperands() != 2)
    return false;
  MDString *KeyMD = dyn_cast<MDString>(MD->getOperand(0));
  if (!KeyMD || !KeyMD->getString().equals("DetailedSummary"))
    return false;
  MDTuple *EntriesMD = dyn_cast<MDTuple>(MD->getOperand(1));
  if (!EntriesMD)
    return false;
  for (auto &&MDOp : EntriesMD->operands()) {
    MDTuple *EntryMD = dyn_cast<MDTuple>(MDOp);
    if (!EntryMD || EntryMD->getNumOperands() != 3)
      return false;
    ConstantAsMetadata *Op0 =
        dyn_cast<ConstantAsMetadata>(EntryMD->getOperand(0));
    ConstantAsMetadata *Op1 =
        dyn_cast<ConstantAsMetadata>(EntryMD->getOperand(1));
    ConstantAsMetadata *Op2 =
        dyn_cast<ConstantAsMetadata>(EntryMD->getOperand(2));

    if (!Op0 || !Op1 || !Op2)
      return false;
    Summary.emplace_back(cast<ConstantInt>(Op0->getValue())->getZExtValue(),
                         cast<ConstantInt>(Op1->getValue())->getZExtValue(),
                         cast<ConstantInt>(Op2->getValue())->getZExtValue());
  }
  return true;
}

// Parse an MDTuple representing an InstrProfSummary object.
static ProfileSummary *getInstrProfSummaryFromMD(MDTuple *Tuple) {
  uint64_t NumBlocks, TotalCount, NumFunctions, MaxFunctionCount, MaxBlockCount,
      MaxInternalBlockCount;
  SummaryEntryVector Summary;

  if (Tuple->getNumOperands() != 8)
    return nullptr;

  // Skip operand 0 which has been already parsed in the caller
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(1)), "TotalCount",
              TotalCount))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(2)), "MaxBlockCount",
              MaxBlockCount))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(3)), "MaxInternalBlockCount",
              MaxInternalBlockCount))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(4)), "MaxFunctionCount",
              MaxFunctionCount))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(5)), "NumBlocks", NumBlocks))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(6)), "NumFunctions",
              NumFunctions))
    return nullptr;
  if (!getSummaryFromMD(dyn_cast<MDTuple>(Tuple->getOperand(7)), Summary))
    return nullptr;
  return new InstrProfSummary(TotalCount, MaxBlockCount, MaxInternalBlockCount,
                              MaxFunctionCount, NumBlocks, NumFunctions,
                              Summary);
}

// Parse an MDTuple representing a SampleProfileSummary object.
static ProfileSummary *getSampleProfileSummaryFromMD(MDTuple *Tuple) {
  uint64_t TotalSamples, MaxSamplesPerLine, MaxFunctionCount,
      NumLinesWithSamples, NumFunctions;
  SummaryEntryVector Summary;

  if (Tuple->getNumOperands() != 7)
    return nullptr;

  // Skip operand 0 which has been already parsed in the caller
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(1)), "TotalSamples",
              TotalSamples))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(2)), "MaxSamplesPerLine",
              MaxSamplesPerLine))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(3)), "MaxFunctionCount",
              MaxFunctionCount))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(4)), "NumLinesWithSamples",
              NumLinesWithSamples))
    return nullptr;
  if (!getVal(dyn_cast<MDTuple>(Tuple->getOperand(5)), "NumFunctions",
              NumFunctions))
    return nullptr;
  if (!getSummaryFromMD(dyn_cast<MDTuple>(Tuple->getOperand(6)), Summary))
    return nullptr;
  return new SampleProfileSummary(TotalSamples, MaxSamplesPerLine,
                                  MaxFunctionCount, NumLinesWithSamples,
                                  NumFunctions, Summary);
}

ProfileSummary *ProfileSummary::getFromMD(Metadata *MD) {
  if (!isa<MDTuple>(MD))
    return nullptr;
  MDTuple *Tuple = cast<MDTuple>(MD);
  auto &FormatMD = Tuple->getOperand(0);
  if (isKeyValuePair(dyn_cast_or_null<MDTuple>(FormatMD), "ProfileFormat",
                     "SampleProfile"))
    return getSampleProfileSummaryFromMD(Tuple);
  else if (isKeyValuePair(dyn_cast_or_null<MDTuple>(FormatMD), "ProfileFormat",
                          "InstrProf"))
    return getInstrProfSummaryFromMD(Tuple);
  else
    return nullptr;
}
