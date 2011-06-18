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
#include "Record.h"
#include "StringMatcher.h"
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

  // Emit a list of intrinsics with corresponding GCC builtins.
  EmitGCCBuiltinList(Ints, OS);

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
  OS << "// Intrinsic ID to overload table\n";
  OS << "#ifdef GET_INTRINSIC_OVERLOAD_TABLE\n";
  OS << "  // Note that entry #0 is the invalid intrinsic!\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "  ";
    if (Ints[i].isOverloaded)
      OS << "true";
    else
      OS << "false";
    OS << ",\n";
  }
  OS << "#endif\n\n";
}

static void EmitTypeForValueType(raw_ostream &OS, MVT::SimpleValueType VT) {
  if (EVT(VT).isInteger()) {
    unsigned BitWidth = EVT(VT).getSizeInBits();
    OS << "IntegerType::get(Context, " << BitWidth << ")";
  } else if (VT == MVT::Other) {
    // MVT::OtherVT is used to mean the empty struct type here.
    OS << "StructType::get(Context)";
  } else if (VT == MVT::f32) {
    OS << "Type::getFloatTy(Context)";
  } else if (VT == MVT::f64) {
    OS << "Type::getDoubleTy(Context)";
  } else if (VT == MVT::f80) {
    OS << "Type::getX86_FP80Ty(Context)";
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
         << "(dyn_cast<VectorType>(Tys[" << Number << "]))";
    else if (ArgType->isSubClassOf("LLVMTruncatedElementVectorType"))
      OS << "VectorType::getTruncatedElementVectorType"
         << "(dyn_cast<VectorType>(Tys[" << Number << "]))";
    else
      OS << "Tys[" << Number << "]";
  } else if (VT == MVT::iAny || VT == MVT::fAny || VT == MVT::vAny) {
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
  } else if (VT == MVT::iPTRAny) {
    // Make sure the user has passed us an argument type to overload. If not,
    // treat it as an ordinary (not overloaded) intrinsic.
    OS << "(" << ArgNo << " < numTys) ? Tys[" << ArgNo 
    << "] : PointerType::getUnqual(";
    EmitTypeGenerate(OS, ArgType->getValueAsDef("ElTy"), ArgNo);
    OS << ")";
    ++ArgNo;
  } else if (VT == MVT::isVoid) {
    if (ArgNo == 0)
      OS << "Type::getVoidTy(Context)";
    else
      // MVT::isVoid is used to mean varargs here.
      OS << "...";
  } else {
    EmitTypeForValueType(OS, VT);
  }
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
  OS << "  default: assert(0 && \"Invalid intrinsic!\");\n";
  
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

void IntrinsicEmitter::EmitGenerator(const std::vector<CodeGenIntrinsic> &Ints, 
                                     raw_ostream &OS) {
  OS << "// Code for generating Intrinsic function declarations.\n";
  OS << "#ifdef GET_INTRINSIC_GENERATOR\n";
  OS << "  switch (id) {\n";
  OS << "  default: assert(0 && \"Invalid intrinsic!\");\n";
  
  // Similar to GET_INTRINSIC_VERIFIER, batch up cases that have identical
  // types.
  typedef std::map<RecPair, std::vector<unsigned>, RecordListComparator> MapTy;
  MapTy UniqueArgInfos;
  
  // Compute the unique argument type info.
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    UniqueArgInfos[make_pair(Ints[i].IS.RetTypeDefs,
                             Ints[i].IS.ParamTypeDefs)].push_back(i);

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

    if (N > 1 &&
        getValueType(ParamTys[N - 1]->getValueAsDef("VT")) == MVT::isVoid) {
      OS << "    IsVarArg = true;\n";
      --N;
    }

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
    assert(0 && "bad mod-ref kind");
    return MRK_none;
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

  // Compute the maximum number of attribute arguments.
  std::vector<const CodeGenIntrinsic*> sortedIntrinsics(Ints.size());
  unsigned maxArgAttrs = 0;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    const CodeGenIntrinsic &intrinsic = Ints[i];
    sortedIntrinsics[i] = &intrinsic;
    maxArgAttrs =
      std::max(maxArgAttrs, unsigned(intrinsic.ArgumentAttributes.size()));
  }

  // Emit an array of AttributeWithIndex.  Most intrinsics will have
  // at least one entry, for the function itself (index ~1), which is
  // usually nounwind.
  OS << "  AttributeWithIndex AWI[" << maxArgAttrs+1 << "];\n";
  OS << "  unsigned NumAttrs = 0;\n";
  OS << "  switch (id) {\n";
  OS << "    default: break;\n";

  AttributeComparator precedes;

  std::stable_sort(sortedIntrinsics.begin(), sortedIntrinsics.end(), precedes);

  for (unsigned i = 0, e = sortedIntrinsics.size(); i != e; ++i) {
    const CodeGenIntrinsic &intrinsic = *sortedIntrinsics[i];
    OS << "  case " << TargetPrefix << "Intrinsic::"
       << intrinsic.EnumName << ":\n";

    // Fill out the case if this is the last case for this range of
    // intrinsics.
    if (i + 1 != e && !precedes(&intrinsic, sortedIntrinsics[i + 1]))
      continue;

    // Keep track of the number of attributes we're writing out.
    unsigned numAttrs = 0;

    // The argument attributes are alreadys sorted by argument index.
    for (unsigned ai = 0, ae = intrinsic.ArgumentAttributes.size(); ai != ae;) {
      unsigned argNo = intrinsic.ArgumentAttributes[ai].first;
      
      OS << "    AWI[" << numAttrs++ << "] = AttributeWithIndex::get("
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
      OS << "    AWI[" << numAttrs++ << "] = AttributeWithIndex::get(~0, ";
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
      OS << "    NumAttrs = " << numAttrs << ";\n";
      OS << "    break;\n";
    } else {
      OS << "    return AttrListPtr();\n";
    }
  }
  
  OS << "  }\n";
  OS << "  return AttrListPtr::get(AWI, NumAttrs);\n";
  OS << "}\n";
  OS << "#endif // GET_INTRINSIC_ATTRIBUTES\n\n";
}

/// EmitModRefBehavior - Determine intrinsic alias analysis mod/ref behavior.
void IntrinsicEmitter::
EmitModRefBehavior(const std::vector<CodeGenIntrinsic> &Ints, raw_ostream &OS){
  OS << "// Determine intrinsic alias analysis mod/ref behavior.\n";
  OS << "#ifdef GET_INTRINSIC_MODREF_BEHAVIOR\n";
  OS << "switch (iid) {\n";
  OS << "default:\n    return UnknownModRefBehavior;\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (Ints[i].ModRef == CodeGenIntrinsic::ReadWriteMem)
      continue;
    OS << "case " << TargetPrefix << "Intrinsic::" << Ints[i].EnumName
      << ":\n";
    switch (Ints[i].ModRef) {
    default:
      assert(false && "Unknown Mod/Ref type!");
    case CodeGenIntrinsic::NoMem:
      OS << "  return DoesNotAccessMemory;\n";
      break;
    case CodeGenIntrinsic::ReadArgMem:
      OS << "  return OnlyReadsArgumentPointees;\n";
      break;
    case CodeGenIntrinsic::ReadMem:
      OS << "  return OnlyReadsMemory;\n";
      break;
    case CodeGenIntrinsic::ReadWriteArgMem:
      OS << "  return OnlyAccessesArgumentPointees;\n";
      break;
    }
  }
  OS << "}\n";
  OS << "#endif // GET_INTRINSIC_MODREF_BEHAVIOR\n\n";
}

void IntrinsicEmitter::
EmitGCCBuiltinList(const std::vector<CodeGenIntrinsic> &Ints, raw_ostream &OS){
  OS << "// Get the GCC builtin that corresponds to an LLVM intrinsic.\n";
  OS << "#ifdef GET_GCC_BUILTIN_NAME\n";
  OS << "  switch (F->getIntrinsicID()) {\n";
  OS << "  default: BuiltinName = \"\"; break;\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (!Ints[i].GCCBuiltinName.empty()) {
      OS << "  case Intrinsic::" << Ints[i].EnumName << ": BuiltinName = \""
         << Ints[i].GCCBuiltinName << "\"; break;\n";
    }
  }
  OS << "  }\n";
  OS << "#endif\n\n";
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
