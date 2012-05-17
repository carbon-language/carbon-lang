//===- IntrinsicEmitter.cpp - Generate intrinsic information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits information about intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "IntrinsicEmitter.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// IntrinsicEmitter Implementation
//===----------------------------------------------------------------------===//

void IntrinsicEmitter::run(raw_ostream &OS) {
  EmitSourceFileHeader("Intrinsic Function Source Fragment", OS);
  
  std::vector<CodeGenIntrinsic> Ints = LoadIntrinsics(Records, TargetOnly);
  
  if (TargetOnly && !Ints.empty())
    TargetPrefix = Ints[0].TargetPrefix;

  EmitPrefix(OS);

  // Emit the enum information.
  EmitEnumInfo(Ints, OS);

  // Emit the intrinsic ID -> name table.
  EmitIntrinsicToNameTable(Ints, OS);

  // Emit the intrinsic ID -> overload table.
  EmitIntrinsicToOverloadTable(Ints, OS);

  // Emit the function name recognizer.
  EmitFnNameRecognizer(Ints, OS);
  
  // Emit the intrinsic verifier.
  EmitVerifier(Ints, OS);
  
  // Emit the intrinsic declaration generator.
  EmitGenerator(Ints, OS);
  
  // Emit the intrinsic parameter attributes.
  EmitAttributes(Ints, OS);

  // Emit intrinsic alias analysis mod/ref behavior.
  EmitModRefBehavior(Ints, OS);

  // Emit code to translate GCC builtins into LLVM intrinsics.
  EmitIntrinsicToGCCBuiltinMap(Ints, OS);

  EmitSuffix(OS);
}

void IntrinsicEmitter::EmitPrefix(raw_ostream &OS) {
  OS << "// VisualStudio defines setjmp as _setjmp\n"
        "#if defined(_MSC_VER) && defined(setjmp) && \\\n"
        "                         !defined(setjmp_undefined_for_msvc)\n"
        "#  pragma push_macro(\"setjmp\")\n"
        "#  undef setjmp\n"
        "#  define setjmp_undefined_for_msvc\n"
        "#endif\n\n";
}

void IntrinsicEmitter::EmitSuffix(raw_ostream &OS) {
  OS << "#if defined(_MSC_VER) && defined(setjmp_undefined_for_msvc)\n"
        "// let's return it to _setjmp state\n"
        "#  pragma pop_macro(\"setjmp\")\n"
        "#  undef setjmp_undefined_for_msvc\n"
        "#endif\n\n";
}

void IntrinsicEmitter::EmitEnumInfo(const std::vector<CodeGenIntrinsic> &Ints,
                                    raw_ostream &OS) {
  OS << "// Enum values for Intrinsics.h\n";
  OS << "#ifdef GET_INTRINSIC_ENUM_VALUES\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "    " << Ints[i].EnumName;
    OS << ((i != e-1) ? ", " : "  ");
    OS << std::string(40-Ints[i].EnumName.size(), ' ') 
      << "// " << Ints[i].Name << "\n";
  }
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitFnNameRecognizer(const std::vector<CodeGenIntrinsic> &Ints, 
                     raw_ostream &OS) {
  // Build a 'first character of function name' -> intrinsic # mapping.
  std::map<char, std::vector<unsigned> > IntMapping;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    IntMapping[Ints[i].Name[5]].push_back(i);
  
  OS << "// Function name -> enum value recognizer code.\n";
  OS << "#ifdef GET_FUNCTION_RECOGNIZER\n";
  OS << "  StringRef NameR(Name+6, Len-6);   // Skip over 'llvm.'\n";
  OS << "  switch (Name[5]) {                  // Dispatch on first letter.\n";
  OS << "  default: break;\n";
  // Emit the intrinsic matching stuff by first letter.
  for (std::map<char, std::vector<unsigned> >::iterator I = IntMapping.begin(),
       E = IntMapping.end(); I != E; ++I) {
    OS << "  case '" << I->first << "':\n";
    std::vector<unsigned> &IntList = I->second;

    // Emit all the overloaded intrinsics first, build a table of the
    // non-overloaded ones.
    std::vector<StringMatcher::StringPair> MatchTable;
    
    for (unsigned i = 0, e = IntList.size(); i != e; ++i) {
      unsigned IntNo = IntList[i];
      std::string Result = "return " + TargetPrefix + "Intrinsic::" +
        Ints[IntNo].EnumName + ";";

      if (!Ints[IntNo].isOverloaded) {
        MatchTable.push_back(std::make_pair(Ints[IntNo].Name.substr(6),Result));
        continue;
      }

      // For overloaded intrinsics, only the prefix needs to match
      std::string TheStr = Ints[IntNo].Name.substr(6);
      TheStr += '.';  // Require "bswap." instead of bswap.
      OS << "    if (NameR.startswith(\"" << TheStr << "\")) "
         << Result << '\n';
    }
    
    // Emit the matcher logic for the fixed length strings.
    StringMatcher("NameR", MatchTable, OS).Emit(1);
    OS << "    break;  // end of '" << I->first << "' case.\n";
  }
  
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitIntrinsicToNameTable(const std::vector<CodeGenIntrinsic> &Ints, 
                         raw_ostream &OS) {
  OS << "// Intrinsic ID to name table\n";
  OS << "#ifdef GET_INTRINSIC_NAME_TABLE\n";
  OS << "  // Note that entry #0 is the invalid intrinsic!\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    OS << "  \"" << Ints[i].Name << "\",\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitIntrinsicToOverloadTable(const std::vector<CodeGenIntrinsic> &Ints, 
                         raw_ostream &OS) {
  OS << "// Intrinsic ID to overload bitset\n";
  OS << "#ifdef GET_INTRINSIC_OVERLOAD_TABLE\n";
  OS << "static const uint8_t OTable[] = {\n";
  OS << "  0";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    // Add one to the index so we emit a null bit for the invalid #0 intrinsic.
    if ((i+1)%8 == 0)
      OS << ",\n  0";
    if (Ints[i].isOverloaded)
      OS << " | (1<<" << (i+1)%8 << ')';
  }
  OS << "\n};\n\n";
  // OTable contains a true bit at the position if the intrinsic is overloaded.
  OS << "return (OTable[id/8] & (1 << (id%8))) != 0;\n";
  OS << "#endif\n\n";
}

/// RecordListComparator - Provide a deterministic comparator for lists of
/// records.
namespace {
  typedef std::pair<std::vector<Record*>, std::vector<Record*> > RecPair;
  struct RecordListComparator {
    bool operator()(const RecPair &LHS,
                    const RecPair &RHS) const {
      unsigned i = 0;
      const std::vector<Record*> *LHSVec = &LHS.first;
      const std::vector<Record*> *RHSVec = &RHS.first;
      unsigned RHSSize = RHSVec->size();
      unsigned LHSSize = LHSVec->size();

      for (; i != LHSSize; ++i) {
        if (i == RHSSize) return false;  // RHS is shorter than LHS.
        if ((*LHSVec)[i] != (*RHSVec)[i])
          return (*LHSVec)[i]->getName() < (*RHSVec)[i]->getName();
      }

      if (i != RHSSize) return true;

      i = 0;
      LHSVec = &LHS.second;
      RHSVec = &RHS.second;
      RHSSize = RHSVec->size();
      LHSSize = LHSVec->size();

      for (i = 0; i != LHSSize; ++i) {
        if (i == RHSSize) return false;  // RHS is shorter than LHS.
        if ((*LHSVec)[i] != (*RHSVec)[i])
          return (*LHSVec)[i]->getName() < (*RHSVec)[i]->getName();
      }

      return i != RHSSize;
    }
  };
}

void IntrinsicEmitter::EmitVerifier(const std::vector<CodeGenIntrinsic> &Ints, 
                                    raw_ostream &OS) {
  OS << "// Verifier::visitIntrinsicFunctionCall code.\n";
  OS << "#ifdef GET_INTRINSIC_VERIFIER\n";
  OS << "  switch (ID) {\n";
  OS << "  default: llvm_unreachable(\"Invalid intrinsic!\");\n";
  
  // This checking can emit a lot of very common code.  To reduce the amount of
  // code that we emit, batch up cases that have identical types.  This avoids
  // problems where GCC can run out of memory compiling Verifier.cpp.
  typedef std::map<RecPair, std::vector<unsigned>, RecordListComparator> MapTy;
  MapTy UniqueArgInfos;
  
  // Compute the unique argument type info.
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    UniqueArgInfos[make_pair(Ints[i].IS.RetTypeDefs,
                             Ints[i].IS.ParamTypeDefs)].push_back(i);

  // Loop through the array, emitting one comparison for each batch.
  for (MapTy::iterator I = UniqueArgInfos.begin(),
       E = UniqueArgInfos.end(); I != E; ++I) {
    for (unsigned i = 0, e = I->second.size(); i != e; ++i)
      OS << "  case Intrinsic::" << Ints[I->second[i]].EnumName << ":\t\t// "
         << Ints[I->second[i]].Name << "\n";
    
    const RecPair &ArgTypes = I->first;
    const std::vector<Record*> &RetTys = ArgTypes.first;
    const std::vector<Record*> &ParamTys = ArgTypes.second;
    std::vector<unsigned> OverloadedTypeIndices;

    OS << "    VerifyIntrinsicPrototype(ID, IF, " << RetTys.size() << ", "
       << ParamTys.size();

    // Emit return types.
    for (unsigned j = 0, je = RetTys.size(); j != je; ++j) {
      Record *ArgType = RetTys[j];
      OS << ", ";

      if (ArgType->isSubClassOf("LLVMMatchType")) {
        unsigned Number = ArgType->getValueAsInt("Number");
        assert(Number < OverloadedTypeIndices.size() &&
               "Invalid matching number!");
        Number = OverloadedTypeIndices[Number];
        if (ArgType->isSubClassOf("LLVMExtendedElementVectorType"))
          OS << "~(ExtendedElementVectorType | " << Number << ")";
        else if (ArgType->isSubClassOf("LLVMTruncatedElementVectorType"))
          OS << "~(TruncatedElementVectorType | " << Number << ")";
        else
          OS << "~" << Number;
      } else {
        MVT::SimpleValueType VT = getValueType(ArgType->getValueAsDef("VT"));
        OS << getEnumName(VT);

        if (EVT(VT).isOverloaded())
          OverloadedTypeIndices.push_back(j);

        if (VT == MVT::isVoid && j != 0 && j != je - 1)
          throw "Var arg type not last argument";
      }
    }

    // Emit the parameter types.
    for (unsigned j = 0, je = ParamTys.size(); j != je; ++j) {
      Record *ArgType = ParamTys[j];
      OS << ", ";

      if (ArgType->isSubClassOf("LLVMMatchType")) {
        unsigned Number = ArgType->getValueAsInt("Number");
        assert(Number < OverloadedTypeIndices.size() &&
               "Invalid matching number!");
        Number = OverloadedTypeIndices[Number];
        if (ArgType->isSubClassOf("LLVMExtendedElementVectorType"))
          OS << "~(ExtendedElementVectorType | " << Number << ")";
        else if (ArgType->isSubClassOf("LLVMTruncatedElementVectorType"))
          OS << "~(TruncatedElementVectorType | " << Number << ")";
        else
          OS << "~" << Number;
      } else {
        MVT::SimpleValueType VT = getValueType(ArgType->getValueAsDef("VT"));
        OS << getEnumName(VT);

        if (EVT(VT).isOverloaded())
          OverloadedTypeIndices.push_back(j + RetTys.size());

        if (VT == MVT::isVoid && j != 0 && j != je - 1)
          throw "Var arg type not last argument";
      }
    }
      
    OS << ");\n";
    OS << "    break;\n";
  }
  OS << "  }\n";
  OS << "#endif\n\n";
}

static void EmitTypeForValueType(raw_ostream &OS, MVT::SimpleValueType VT) {
  if (EVT(VT).isInteger()) {
    unsigned BitWidth = EVT(VT).getSizeInBits();
    OS << "IntegerType::get(Context, " << BitWidth << ")";
  } else if (VT == MVT::Other) {
    // MVT::OtherVT is used to mean the empty struct type here.
    OS << "StructType::get(Context)";
  } else if (VT == MVT::f16) {
    OS << "Type::getHalfTy(Context)";
  } else if (VT == MVT::f32) {
    OS << "Type::getFloatTy(Context)";
  } else if (VT == MVT::f64) {
    OS << "Type::getDoubleTy(Context)";
  } else if (VT == MVT::f128) {
    OS << "Type::getFP128Ty(Context)";
  } else if (VT == MVT::ppcf128) {
    OS << "Type::getPPC_FP128Ty(Context)";
  } else if (VT == MVT::isVoid) {
    OS << "Type::getVoidTy(Context)";
  } else if (VT == MVT::Metadata) {
    OS << "Type::getMetadataTy(Context)";
  } else if (VT == MVT::x86mmx) {
    OS << "Type::getX86_MMXTy(Context)";
  } else {
    assert(false && "Unsupported ValueType!");
  }
}

static void EmitTypeGenerate(raw_ostream &OS, const Record *ArgType,
                             unsigned &ArgNo);

static void EmitTypeGenerate(raw_ostream &OS,
                             const std::vector<Record*> &ArgTypes,
                             unsigned &ArgNo) {
  if (ArgTypes.empty())
    return EmitTypeForValueType(OS, MVT::isVoid);
  
  if (ArgTypes.size() == 1)
    return EmitTypeGenerate(OS, ArgTypes.front(), ArgNo);
  
  OS << "StructType::get(";
  
  for (std::vector<Record*>::const_iterator
       I = ArgTypes.begin(), E = ArgTypes.end(); I != E; ++I) {
    EmitTypeGenerate(OS, *I, ArgNo);
    OS << ", ";
  }
  
  OS << " NULL)";
}

static void EmitTypeGenerate(raw_ostream &OS, const Record *ArgType,
                             unsigned &ArgNo) {
  MVT::SimpleValueType VT = getValueType(ArgType->getValueAsDef("VT"));
  
  if (ArgType->isSubClassOf("LLVMMatchType")) {
    unsigned Number = ArgType->getValueAsInt("Number");
    assert(Number < ArgNo && "Invalid matching number!");
    if (ArgType->isSubClassOf("LLVMExtendedElementVectorType"))
      OS << "VectorType::getExtendedElementVectorType"
      << "(cast<VectorType>(Tys[" << Number << "]))";
    else if (ArgType->isSubClassOf("LLVMTruncatedElementVectorType"))
      OS << "VectorType::getTruncatedElementVectorType"
      << "(cast<VectorType>(Tys[" << Number << "]))";
    else
      OS << "Tys[" << Number << "]";
  } else if (VT == MVT::iAny || VT == MVT::fAny || VT == MVT::vAny ||
             VT == MVT::iPTRAny) {
    // NOTE: The ArgNo variable here is not the absolute argument number, it is
    // the index of the "arbitrary" type in the Tys array passed to the
    // Intrinsic::getDeclaration function. Consequently, we only want to
    // increment it when we actually hit an overloaded type. Getting this wrong
    // leads to very subtle bugs!
    OS << "Tys[" << ArgNo++ << "]";
  } else if (EVT(VT).isVector()) {
    EVT VVT = VT;
    OS << "VectorType::get(";
    EmitTypeForValueType(OS, VVT.getVectorElementType().getSimpleVT().SimpleTy);
    OS << ", " << VVT.getVectorNumElements() << ")";
  } else if (VT == MVT::iPTR) {
    OS << "PointerType::getUnqual(";
    EmitTypeGenerate(OS, ArgType->getValueAsDef("ElTy"), ArgNo);
    OS << ")";
  } else if (VT == MVT::isVoid) {
    assert(ArgNo == 0);
    OS << "Type::getVoidTy(Context)";
  } else {
    EmitTypeForValueType(OS, VT);
  }
}


// NOTE: This must be kept in synch with the version emitted to the .gen file!
enum IIT_Info {
  // Common values should be encoded with 0-15.
  IIT_Done = 0,
  IIT_I1   = 1,
  IIT_I8   = 2,
  IIT_I16  = 3,
  IIT_I32  = 4,
  IIT_I64  = 5,
  IIT_F32  = 6,
  IIT_F64  = 7,
  IIT_V2   = 8,
  IIT_V4   = 9,
  IIT_V8   = 10,
  IIT_V16  = 11,
  IIT_V32  = 12,
  IIT_MMX  = 13,
  IIT_PTR  = 14,
  IIT_ARG  = 15,
  
  // Values from 16+ are only encodable with the inefficient encoding.
  IIT_METADATA = 16,
  IIT_EMPTYSTRUCT = 17,
  IIT_STRUCT2 = 18,
  IIT_STRUCT3 = 19,
  IIT_STRUCT4 = 20,
  IIT_STRUCT5 = 21,
  IIT_EXTEND_VEC_ARG = 22,
  IIT_TRUNC_VEC_ARG = 23
};


static void EncodeFixedValueType(MVT::SimpleValueType VT,
                                 SmallVectorImpl<unsigned> &Sig) {
  if (EVT(VT).isInteger()) {
    unsigned BitWidth = EVT(VT).getSizeInBits();
    switch (BitWidth) {
    default: throw "unhandled integer type width in intrinsic!";
    case 1: return Sig.push_back(IIT_I1);
    case 8: return Sig.push_back(IIT_I8);
    case 16: return Sig.push_back(IIT_I16);
    case 32: return Sig.push_back(IIT_I32);
    case 64: return Sig.push_back(IIT_I64);
    }
  }
  
  switch (VT) {
  default: throw "unhandled MVT in intrinsic!";
  case MVT::f32: return Sig.push_back(IIT_F32);
  case MVT::f64: return Sig.push_back(IIT_F64);
  case MVT::Metadata: return Sig.push_back(IIT_METADATA);
  case MVT::x86mmx: return Sig.push_back(IIT_MMX);
  // MVT::OtherVT is used to mean the empty struct type here.
  case MVT::Other: return Sig.push_back(IIT_EMPTYSTRUCT);
  }
}

#ifdef _MSC_VER
#pragma optimize("",off) // MSVC 2010 optimizer can't deal with this function.
#endif 

static void EncodeFixedType(Record *R, unsigned &NextArgNo,
                            SmallVectorImpl<unsigned> &Sig) {
  
  if (R->isSubClassOf("LLVMMatchType")) {
    unsigned Number = R->getValueAsInt("Number");
    assert(Number < NextArgNo && "Invalid matching number!");
    if (R->isSubClassOf("LLVMExtendedElementVectorType"))
      Sig.push_back(IIT_EXTEND_VEC_ARG);
    else if (R->isSubClassOf("LLVMTruncatedElementVectorType"))
      Sig.push_back(IIT_TRUNC_VEC_ARG);
    else
      Sig.push_back(IIT_ARG);
    return Sig.push_back(Number);
  }
  
  MVT::SimpleValueType VT = getValueType(R->getValueAsDef("VT"));

  // If this is an "any" valuetype, then the type is the type of the next
  // type in the list specified to getIntrinsic().  
  if (VT == MVT::iAny || VT == MVT::fAny || VT == MVT::vAny ||
      VT == MVT::iPTRAny) {
    Sig.push_back(IIT_ARG);
    return Sig.push_back(NextArgNo++);
  }
  
  if (EVT(VT).isVector()) {
    EVT VVT = VT;
    switch (VVT.getVectorNumElements()) {
    default: throw "unhandled vector type width in intrinsic!";
    case 2: Sig.push_back(IIT_V2); break;
    case 4: Sig.push_back(IIT_V4); break;
    case 8: Sig.push_back(IIT_V8); break;
    case 16: Sig.push_back(IIT_V16); break;
    case 32: Sig.push_back(IIT_V32); break;
    }
    
    return EncodeFixedValueType(VVT.getVectorElementType().
                                getSimpleVT().SimpleTy, Sig);
  }
  
  if (VT == MVT::iPTR) {
    Sig.push_back(IIT_PTR);
    return EncodeFixedType(R->getValueAsDef("ElTy"), NextArgNo, Sig);
  }
  
  EncodeFixedValueType(VT, Sig);
}

#ifdef _MSC_VER
#pragma optimize("",on)
#endif

/// ComputeFixedEncoding - If we can encode the type signature for this
/// intrinsic into 32 bits, return it.  If not, return ~0U.
static unsigned ComputeFixedEncoding(const CodeGenIntrinsic &Int) {
  unsigned NextArgNo = 0;
  
  SmallVector<unsigned, 8> TypeSig;
  if (Int.IS.RetVTs.empty())
    TypeSig.push_back(IIT_Done);
  else if (Int.IS.RetVTs.size() == 1 &&
           Int.IS.RetVTs[0] == MVT::isVoid)
    TypeSig.push_back(IIT_Done);
  else {
    switch (Int.IS.RetVTs.size()) {
    case 1: break;
    case 2: TypeSig.push_back(IIT_STRUCT2); break;
    case 3: TypeSig.push_back(IIT_STRUCT3); break;
    case 4: TypeSig.push_back(IIT_STRUCT4); break;
    case 5: TypeSig.push_back(IIT_STRUCT5); break;
    default: assert(0 && "Unhandled case in struct");
    }

    for (unsigned i = 0, e = Int.IS.RetVTs.size(); i != e; ++i)
      EncodeFixedType(Int.IS.RetTypeDefs[i], NextArgNo, TypeSig);
  }
  
  for (unsigned i = 0, e = Int.IS.ParamTypeDefs.size(); i != e; ++i)
    EncodeFixedType(Int.IS.ParamTypeDefs[i], NextArgNo, TypeSig);
  
  // Can only encode 8 nibbles into a 32-bit word.
  if (TypeSig.size() > 8) return ~0U;
  
  unsigned Result = 0;
  for (unsigned i = 0, e = TypeSig.size(); i != e; ++i) {
    // If we had an unencodable argument, bail out.
    if (TypeSig[i] > 15)
      return ~0U;
    Result = (Result << 4) | TypeSig[e-i-1];
  }
  
  return Result;
}

void IntrinsicEmitter::EmitGenerator(const std::vector<CodeGenIntrinsic> &Ints, 
                                     raw_ostream &OS) {
  OS << "// Global intrinsic function declaration type table.\n";
  OS << "#ifdef GET_INTRINSTIC_GENERATOR_GLOBAL\n";
  // NOTE: These enums must be kept in sync with the ones above!
  OS << "enum IIT_Info {\n";
  OS << "  IIT_Done = 0,\n";
  OS << "  IIT_I1   = 1,\n";
  OS << "  IIT_I8   = 2,\n";
  OS << "  IIT_I16  = 3,\n";
  OS << "  IIT_I32  = 4,\n";
  OS << "  IIT_I64  = 5,\n";
  OS << "  IIT_F32  = 6,\n";
  OS << "  IIT_F64  = 7,\n";
  OS << "  IIT_V2   = 8,\n";
  OS << "  IIT_V4   = 9,\n";
  OS << "  IIT_V8   = 10,\n";
  OS << "  IIT_V16  = 11,\n";
  OS << "  IIT_V32  = 12,\n";
  OS << "  IIT_MMX  = 13,\n";
  OS << "  IIT_PTR  = 14,\n";
  OS << "  IIT_ARG  = 15,\n";
  OS << "  IIT_METADATA = 16,\n";
  OS << "  IIT_EMPTYSTRUCT = 17,\n";
  OS << "  IIT_STRUCT2 = 18,\n";
  OS << "  IIT_STRUCT3 = 19,\n";
  OS << "  IIT_STRUCT4 = 20,\n";
  OS << "  IIT_STRUCT5 = 21,\n";
  OS << "  IIT_EXTEND_VEC_ARG = 22,\n";
  OS << "  IIT_TRUNC_VEC_ARG = 23\n";
  OS << "};\n\n";

  
  // Similar to GET_INTRINSIC_VERIFIER, batch up cases that have identical
  // types.
  typedef std::map<RecPair, std::vector<unsigned>, RecordListComparator> MapTy;
  MapTy UniqueArgInfos;

  // If we can compute a 32-bit fixed encoding for this intrinsic, do so and
  // capture it in this vector, otherwise store a ~0U.
  std::vector<unsigned> FixedEncodings;
  
  // Compute the unique argument type info.
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    FixedEncodings.push_back(ComputeFixedEncoding(Ints[i]));
    
    // If we didn't compute a compact encoding, emit a long-form variant.
    if (FixedEncodings.back() == ~0U)
      UniqueArgInfos[make_pair(Ints[i].IS.RetTypeDefs,
                               Ints[i].IS.ParamTypeDefs)].push_back(i);
  }
  
  OS << "static const unsigned IIT_Table[] = {\n  ";
  
  for (unsigned i = 0, e = FixedEncodings.size(); i != e; ++i) {
    if ((i & 7) == 7)
      OS << "\n  ";
    if (FixedEncodings[i] == ~0U) 
      OS << "~0U, ";
    else
      OS << "0x" << utohexstr(FixedEncodings[i]) << ", ";
  }
  
  OS << "0\n};\n\n#endif\n\n";  // End of GET_INTRINSTIC_GENERATOR_GLOBAL
  
  OS << "// Code for generating Intrinsic function declarations.\n";
  OS << "#ifdef GET_INTRINSIC_GENERATOR\n";
  OS << "  switch (id) {\n";
  OS << "  default: llvm_unreachable(\"Invalid intrinsic!\");\n";

  // Loop through the array, emitting one generator for each batch.
  std::string IntrinsicStr = TargetPrefix + "Intrinsic::";
  
  for (MapTy::iterator I = UniqueArgInfos.begin(),
       E = UniqueArgInfos.end(); I != E; ++I) {
    for (unsigned i = 0, e = I->second.size(); i != e; ++i)
      OS << "  case " << IntrinsicStr << Ints[I->second[i]].EnumName 
         << ":\t\t// " << Ints[I->second[i]].Name << "\n";
    
    const RecPair &ArgTypes = I->first;
    const std::vector<Record*> &RetTys = ArgTypes.first;
    const std::vector<Record*> &ParamTys = ArgTypes.second;

    unsigned N = ParamTys.size();
    unsigned ArgNo = 0;
    OS << "    ResultTy = ";
    EmitTypeGenerate(OS, RetTys, ArgNo);
    OS << ";\n";
    
    for (unsigned j = 0; j != N; ++j) {
      OS << "    ArgTys.push_back(";
      EmitTypeGenerate(OS, ParamTys[j], ArgNo);
      OS << ");\n";
    }

    OS << "    break;\n";
  }

  OS << "  }\n";
  OS << "#endif\n\n";
}

namespace {
  enum ModRefKind {
    MRK_none,
    MRK_readonly,
    MRK_readnone
  };

  ModRefKind getModRefKind(const CodeGenIntrinsic &intrinsic) {
    switch (intrinsic.ModRef) {
    case CodeGenIntrinsic::NoMem:
      return MRK_readnone;
    case CodeGenIntrinsic::ReadArgMem:
    case CodeGenIntrinsic::ReadMem:
      return MRK_readonly;
    case CodeGenIntrinsic::ReadWriteArgMem:
    case CodeGenIntrinsic::ReadWriteMem:
      return MRK_none;
    }
    llvm_unreachable("bad mod-ref kind");
  }

  struct AttributeComparator {
    bool operator()(const CodeGenIntrinsic *L, const CodeGenIntrinsic *R) const {
      // Sort throwing intrinsics after non-throwing intrinsics.
      if (L->canThrow != R->canThrow)
        return R->canThrow;

      // Try to order by readonly/readnone attribute.
      ModRefKind LK = getModRefKind(*L);
      ModRefKind RK = getModRefKind(*R);
      if (LK != RK) return (LK > RK);

      // Order by argument attributes.
      // This is reliable because each side is already sorted internally.
      return (L->ArgumentAttributes < R->ArgumentAttributes);
    }
  };
}

/// EmitAttributes - This emits the Intrinsic::getAttributes method.
void IntrinsicEmitter::
EmitAttributes(const std::vector<CodeGenIntrinsic> &Ints, raw_ostream &OS) {
  OS << "// Add parameter attributes that are not common to all intrinsics.\n";
  OS << "#ifdef GET_INTRINSIC_ATTRIBUTES\n";
  if (TargetOnly)
    OS << "static AttrListPtr getAttributes(" << TargetPrefix 
       << "Intrinsic::ID id) {\n";
  else
    OS << "AttrListPtr Intrinsic::getAttributes(ID id) {\n";

  // Compute the maximum number of attribute arguments and the map
  typedef std::map<const CodeGenIntrinsic*, unsigned,
                   AttributeComparator> UniqAttrMapTy;
  UniqAttrMapTy UniqAttributes;
  unsigned maxArgAttrs = 0;
  unsigned AttrNum = 0;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    const CodeGenIntrinsic &intrinsic = Ints[i];
    maxArgAttrs =
      std::max(maxArgAttrs, unsigned(intrinsic.ArgumentAttributes.size()));
    unsigned &N = UniqAttributes[&intrinsic];
    if (N) continue;
    assert(AttrNum < 256 && "Too many unique attributes for table!");
    N = ++AttrNum;
  }

  // Emit an array of AttributeWithIndex.  Most intrinsics will have
  // at least one entry, for the function itself (index ~1), which is
  // usually nounwind.
  OS << "  static const uint8_t IntrinsicsToAttributesMap[] = {\n";

  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    const CodeGenIntrinsic &intrinsic = Ints[i];

    OS << "    " << UniqAttributes[&intrinsic] << ", // "
       << intrinsic.Name << "\n";
  }
  OS << "  };\n\n";

  OS << "  AttributeWithIndex AWI[" << maxArgAttrs+1 << "];\n";
  OS << "  unsigned NumAttrs = 0;\n";
  OS << "  if (id != 0) {\n";
  OS << "    switch(IntrinsicsToAttributesMap[id - ";
  if (TargetOnly)
    OS << "Intrinsic::num_intrinsics";
  else
    OS << "1";
  OS << "]) {\n";
  OS << "    default: llvm_unreachable(\"Invalid attribute number\");\n";
  for (UniqAttrMapTy::const_iterator I = UniqAttributes.begin(),
       E = UniqAttributes.end(); I != E; ++I) {
    OS << "    case " << I->second << ":\n";

    const CodeGenIntrinsic &intrinsic = *(I->first);

    // Keep track of the number of attributes we're writing out.
    unsigned numAttrs = 0;

    // The argument attributes are alreadys sorted by argument index.
    for (unsigned ai = 0, ae = intrinsic.ArgumentAttributes.size(); ai != ae;) {
      unsigned argNo = intrinsic.ArgumentAttributes[ai].first;

      OS << "      AWI[" << numAttrs++ << "] = AttributeWithIndex::get("
         << argNo+1 << ", ";

      bool moreThanOne = false;

      do {
        if (moreThanOne) OS << '|';

        switch (intrinsic.ArgumentAttributes[ai].second) {
        case CodeGenIntrinsic::NoCapture:
          OS << "Attribute::NoCapture";
          break;
        }

        ++ai;
        moreThanOne = true;
      } while (ai != ae && intrinsic.ArgumentAttributes[ai].first == argNo);

      OS << ");\n";
    }

    ModRefKind modRef = getModRefKind(intrinsic);

    if (!intrinsic.canThrow || modRef) {
      OS << "      AWI[" << numAttrs++ << "] = AttributeWithIndex::get(~0, ";
      if (!intrinsic.canThrow) {
        OS << "Attribute::NoUnwind";
        if (modRef) OS << '|';
      }
      switch (modRef) {
      case MRK_none: break;
      case MRK_readonly: OS << "Attribute::ReadOnly"; break;
      case MRK_readnone: OS << "Attribute::ReadNone"; break;
      }
      OS << ");\n";
    }

    if (numAttrs) {
      OS << "      NumAttrs = " << numAttrs << ";\n";
      OS << "      break;\n";
    } else {
      OS << "      return AttrListPtr();\n";
    }
  }
  
  OS << "    }\n";
  OS << "  }\n";
  OS << "  return AttrListPtr::get(AWI, NumAttrs);\n";
  OS << "}\n";
  OS << "#endif // GET_INTRINSIC_ATTRIBUTES\n\n";
}

/// EmitModRefBehavior - Determine intrinsic alias analysis mod/ref behavior.
void IntrinsicEmitter::
EmitModRefBehavior(const std::vector<CodeGenIntrinsic> &Ints, raw_ostream &OS){
  OS << "// Determine intrinsic alias analysis mod/ref behavior.\n"
     << "#ifdef GET_INTRINSIC_MODREF_BEHAVIOR\n"
     << "assert(iid <= Intrinsic::" << Ints.back().EnumName << " && "
     << "\"Unknown intrinsic.\");\n\n";

  OS << "static const uint8_t IntrinsicModRefBehavior[] = {\n"
     << "  /* invalid */ UnknownModRefBehavior,\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "  /* " << TargetPrefix << Ints[i].EnumName << " */ ";
    switch (Ints[i].ModRef) {
    case CodeGenIntrinsic::NoMem:
      OS << "DoesNotAccessMemory,\n";
      break;
    case CodeGenIntrinsic::ReadArgMem:
      OS << "OnlyReadsArgumentPointees,\n";
      break;
    case CodeGenIntrinsic::ReadMem:
      OS << "OnlyReadsMemory,\n";
      break;
    case CodeGenIntrinsic::ReadWriteArgMem:
      OS << "OnlyAccessesArgumentPointees,\n";
      break;
    case CodeGenIntrinsic::ReadWriteMem:
      OS << "UnknownModRefBehavior,\n";
      break;
    }
  }
  OS << "};\n\n"
     << "return static_cast<ModRefBehavior>(IntrinsicModRefBehavior[iid]);\n"
     << "#endif // GET_INTRINSIC_MODREF_BEHAVIOR\n\n";
}

/// EmitTargetBuiltins - All of the builtins in the specified map are for the
/// same target, and we already checked it.
static void EmitTargetBuiltins(const std::map<std::string, std::string> &BIM,
                               const std::string &TargetPrefix,
                               raw_ostream &OS) {
  
  std::vector<StringMatcher::StringPair> Results;
  
  for (std::map<std::string, std::string>::const_iterator I = BIM.begin(),
       E = BIM.end(); I != E; ++I) {
    std::string ResultCode =
    "return " + TargetPrefix + "Intrinsic::" + I->second + ";";
    Results.push_back(StringMatcher::StringPair(I->first, ResultCode));
  }

  StringMatcher("BuiltinName", Results, OS).Emit();
}

        
void IntrinsicEmitter::
EmitIntrinsicToGCCBuiltinMap(const std::vector<CodeGenIntrinsic> &Ints, 
                             raw_ostream &OS) {
  typedef std::map<std::string, std::map<std::string, std::string> > BIMTy;
  BIMTy BuiltinMap;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (!Ints[i].GCCBuiltinName.empty()) {
      // Get the map for this target prefix.
      std::map<std::string, std::string> &BIM =BuiltinMap[Ints[i].TargetPrefix];
      
      if (!BIM.insert(std::make_pair(Ints[i].GCCBuiltinName,
                                     Ints[i].EnumName)).second)
        throw "Intrinsic '" + Ints[i].TheDef->getName() +
              "': duplicate GCC builtin name!";
    }
  }
  
  OS << "// Get the LLVM intrinsic that corresponds to a GCC builtin.\n";
  OS << "// This is used by the C front-end.  The GCC builtin name is passed\n";
  OS << "// in as BuiltinName, and a target prefix (e.g. 'ppc') is passed\n";
  OS << "// in as TargetPrefix.  The result is assigned to 'IntrinsicID'.\n";
  OS << "#ifdef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN\n";
  
  if (TargetOnly) {
    OS << "static " << TargetPrefix << "Intrinsic::ID "
       << "getIntrinsicForGCCBuiltin(const char "
       << "*TargetPrefixStr, const char *BuiltinNameStr) {\n";
  } else {
    OS << "Intrinsic::ID Intrinsic::getIntrinsicForGCCBuiltin(const char "
       << "*TargetPrefixStr, const char *BuiltinNameStr) {\n";
  }
  
  OS << "  StringRef BuiltinName(BuiltinNameStr);\n";
  OS << "  StringRef TargetPrefix(TargetPrefixStr);\n\n";
  
  // Note: this could emit significantly better code if we cared.
  for (BIMTy::iterator I = BuiltinMap.begin(), E = BuiltinMap.end();I != E;++I){
    OS << "  ";
    if (!I->first.empty())
      OS << "if (TargetPrefix == \"" << I->first << "\") ";
    else
      OS << "/* Target Independent Builtins */ ";
    OS << "{\n";

    // Emit the comparisons for this target prefix.
    EmitTargetBuiltins(I->second, TargetPrefix, OS);
    OS << "  }\n";
  }
  OS << "  return ";
  if (!TargetPrefix.empty())
    OS << "(" << TargetPrefix << "Intrinsic::ID)";
  OS << "Intrinsic::not_intrinsic;\n";
  OS << "}\n";
  OS << "#endif\n\n";
}
