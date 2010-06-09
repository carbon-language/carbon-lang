//===- NeonEmitter.cpp - Generate arm_neon.h for use with clang -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting arm_neon.h, which includes
// a declaration and definition of each function specified by the ARM NEON 
// compiler interface.  See ARM document DUI0348B.
//
//===----------------------------------------------------------------------===//

#include "NeonEmitter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <string>

using namespace llvm;

static void ParseTypes(Record *r, std::string &s,
                       SmallVectorImpl<StringRef> &TV) {
  const char *data = s.data();
  int len = 0;
  
  for (unsigned i = 0, e = s.size(); i != e; ++i, ++len) {
    if (data[len] == 'P' || data[len] == 'Q' || data[len] == 'U')
      continue;
    
    switch (data[len]) {
      case 'c':
      case 's':
      case 'i':
      case 'l':
      case 'h':
      case 'f':
        break;
      default:
        throw TGError(r->getLoc(),
                      "Unexpected letter: " + std::string(data + len, 1));
        break;
    }
    TV.push_back(StringRef(data, len + 1));
    data += len + 1;
    len = -1;
  }
}

static char Widen(const char t) {
  switch (t) {
    case 'c':
      return 's';
    case 's':
      return 'i';
    case 'i':
      return 'l';
    default: throw "unhandled type in widen!";
  }
  return '\0';
}

static char Narrow(const char t) {
  switch (t) {
    case 's':
      return 'c';
    case 'i':
      return 's';
    case 'l':
      return 'i';
    case 'f':
      return 'h';
    default: throw "unhandled type in widen!";
  }
  return '\0';
}

static char ClassifyType(StringRef ty, bool &quad, bool &poly, bool &usgn) {
  unsigned off = 0;
  
  // remember quad.
  if (ty[off] == 'Q') {
    quad = true;
    ++off;
  }
  
  // remember poly.
  if (ty[off] == 'P') {
    poly = true;
    ++off;
  }
  
  // remember unsigned.
  if (ty[off] == 'U') {
    usgn = true;
    ++off;
  }
  
  // base type to get the type string for.
  return ty[off];
}

static char ModType(const char mod, char type, bool &quad, bool &poly,
                    bool &usgn, bool &scal, bool &cnst, bool &pntr) {
  switch (mod) {
    case 't':
      if (poly) {
        poly = false;
        usgn = true;
      }
      break;
    case 'u':
      usgn = true;
    case 'x':
      poly = false;
      if (type == 'f')
        type = 'i';
      break;
    case 'f':
      if (type == 'h')
        quad = true;
      type = 'f';
      usgn = false;
      break;
    case 'w':
      type = Widen(type);
      quad = true;
      break;
    case 'n':
      type = Widen(type);
      break;
    case 'l':
      type = 'l';
      scal = true;
      usgn = true;
      break;
    case 's':
      scal = true;
      break;
    case 'k':
      quad = true;
      break;
    case 'c':
      cnst = true;
    case 'p':
      usgn = false;
      poly = false;
      pntr = true;
      scal = true;
      break;
    case 'h':
      type = Narrow(type);
      if (type == 'h')
        quad = false;
      break;
    case 'e':
      type = Narrow(type);
      usgn = true;
      break;
    default:
      break;
  }
  return type;
}

static std::string TypeString(const char mod, StringRef typestr,
                              bool ret = false) {
  bool quad = false;
  bool poly = false;
  bool usgn = false;
  bool scal = false;
  bool cnst = false;
  bool pntr = false;
  
  if (mod == 'v')
    return "void";
  if (mod == 'i')
    return "int";
  
  // base type to get the type string for.
  char type = ClassifyType(typestr, quad, poly, usgn);
  
  // Based on the modifying character, change the type and width if necessary.
  type = ModType(mod, type, quad, poly, usgn, scal, cnst, pntr);
  
  SmallString<128> s;
  
  if (ret)
    s += "__neon_";
  
  if (usgn)
    s.push_back('u');
  
  switch (type) {
    case 'c':
      s += poly ? "poly8" : "int8";
      if (scal)
        break;
      s += quad ? "x16" : "x8";
      break;
    case 's':
      s += poly ? "poly16" : "int16";
      if (scal)
        break;
      s += quad ? "x8" : "x4";
      break;
    case 'i':
      s += "int32";
      if (scal)
        break;
      s += quad ? "x4" : "x2";
      break;
    case 'l':
      s += "int64";
      if (scal)
        break;
      s += quad ? "x2" : "x1";
      break;
    case 'h':
      s += "float16";
      if (scal)
        break;
      s += quad ? "x8" : "x4";
      break;
    case 'f':
      s += "float32";
      if (scal)
        break;
      s += quad ? "x4" : "x2";
      break;
    default:
      throw "unhandled type!";
      break;
  }

  if (mod == '2')
    s += "x2";
  if (mod == '3')
    s += "x3";
  if (mod == '4')
    s += "x4";
  
  // Append _t, finishing the type string typedef type.
  s += "_t";
  
  if (cnst)
    s += " const";
  
  if (pntr)
    s += " *";
  
  return s.str();
}

static std::string BuiltinTypeString(const char mod, StringRef typestr,
                                     ClassKind ck, bool ret) {
  bool quad = false;
  bool poly = false;
  bool usgn = false;
  bool scal = false;
  bool cnst = false;
  bool pntr = false;
  
  if (mod == 'v')
    return "v";
  if (mod == 'i')
    return "i";
  
  // base type to get the type string for.
  char type = ClassifyType(typestr, quad, poly, usgn);
  
  // Based on the modifying character, change the type and width if necessary.
  type = ModType(mod, type, quad, poly, usgn, scal, cnst, pntr);

  if (pntr)
    type = 'v';
  
  if (type == 'h') {
    type = 's';
    usgn = true;
  }
  usgn = usgn | poly | ((ck == ClassI || ck == ClassW) && scal && type != 'f');

  if (scal) {
    SmallString<128> s;

    if (usgn)
      s.push_back('U');
    
    if (type == 'l')
      s += "LLi";
    else
      s.push_back(type);
 
    if (cnst)
      s.push_back('C');
    if (pntr)
      s.push_back('*');
    return s.str();
  }

  // Since the return value must be one type, return a vector type of the
  // appropriate width which we will bitcast.
  if (ret) {
    if (mod == '2')
      return quad ? "V32c" : "V16c";
    if (mod == '3')
      return quad ? "V48c" : "V24c";
    if (mod == '4')
      return quad ? "V64c" : "V32c";
    if (mod == 'f' || (ck != ClassB && type == 'f'))
      return quad ? "V4f" : "V2f";
    if (ck != ClassB && type == 's')
      return quad ? "V8s" : "V4s";
    if (ck != ClassB && type == 'i')
      return quad ? "V4i" : "V2i";
    if (ck != ClassB && type == 'l')
      return quad ? "V2LLi" : "V1LLi";
    
    return quad ? "V16c" : "V8c";
  }    

  // Non-return array types are passed as individual vectors.
  if (mod == '2')
    return quad ? "V16cV16c" : "V8cV8c";
  if (mod == '3')
    return quad ? "V16cV16cV16c" : "V8cV8cV8c";
  if (mod == '4')
    return quad ? "V16cV16cV16cV16c" : "V8cV8cV8cV8c";

  if (mod == 'f' || (ck != ClassB && type == 'f'))
    return quad ? "V4f" : "V2f";
  if (ck != ClassB && type == 's')
    return quad ? "V8s" : "V4s";
  if (ck != ClassB && type == 'i')
    return quad ? "V4i" : "V2i";
  if (ck != ClassB && type == 'l')
    return quad ? "V2LLi" : "V1LLi";
  
  return quad ? "V16c" : "V8c";
}

// Turn "vst2_lane" into "vst2q_lane_f32", etc.
static std::string MangleName(const std::string &name, StringRef typestr,
                              ClassKind ck) {
  if (name == "vcvt_f32_f16")
    return name;
  
  bool quad = false;
  bool poly = false;
  bool usgn = false;
  char type = ClassifyType(typestr, quad, poly, usgn);

  std::string s = name;
  
  switch (type) {
  case 'c':
    switch (ck) {
    case ClassS: s += poly ? "_p8" : usgn ? "_u8" : "_s8"; break;
    case ClassI: s += "_i8"; break;
    case ClassW: s += "_8"; break;
    default: break;
    }
    break;
  case 's':
    switch (ck) {
    case ClassS: s += poly ? "_p16" : usgn ? "_u16" : "_s16"; break;
    case ClassI: s += "_i16"; break;
    case ClassW: s += "_16"; break;
    default: break;
    }
    break;
  case 'i':
    switch (ck) {
    case ClassS: s += usgn ? "_u32" : "_s32"; break;
    case ClassI: s += "_i32"; break;
    case ClassW: s += "_32"; break;
    default: break;
    }
    break;
  case 'l':
    switch (ck) {
    case ClassS: s += usgn ? "_u64" : "_s64"; break;
    case ClassI: s += "_i64"; break;
    case ClassW: s += "_64"; break;
    default: break;
    }
    break;
  case 'h':
    switch (ck) {
    case ClassS:
    case ClassI: s += "_f16"; break;
    case ClassW: s += "_16"; break;
    default: break;
    }
    break;
  case 'f':
    switch (ck) {
    case ClassS:
    case ClassI: s += "_f32"; break;
    case ClassW: s += "_32"; break;
    default: break;
    }
    break;
  default:
    throw "unhandled type!";
    break;
  }
  if (ck == ClassB)
    s += "_v";
    
  // Insert a 'q' before the first '_' character so that it ends up before 
  // _lane or _n on vector-scalar operations.
  if (quad) {
    size_t pos = s.find('_');
    s = s.insert(pos, "q");
  }
  return s;
}

// Generate the string "(argtype a, argtype b, ...)"
static std::string GenArgs(const std::string &proto, StringRef typestr) {
  bool define = proto.find('i') != std::string::npos;
  char arg = 'a';
  
  std::string s;
  s += "(";
  
  for (unsigned i = 1, e = proto.size(); i != e; ++i, ++arg) {
    if (!define) {
      s += TypeString(proto[i], typestr);
      s.push_back(' ');
    }
    s.push_back(arg);
    if ((i + 1) < e)
      s += ", ";
  }
  
  s += ")";
  return s;
}

// Generate the definition for this intrinsic, e.g. "a + b" for OpAdd.
// If structTypes is true, the NEON types are structs of vector types rather
// than vector types, and the call becomes "a.val + b.val"
static std::string GenOpString(OpKind op, const std::string &proto,
                               StringRef typestr, bool structTypes = true) {
  std::string ts = TypeString(proto[0], typestr);
  std::string s = ts + " r; r";

  bool dummy, quad = false;
  char type = ClassifyType(typestr, quad, dummy, dummy);
  unsigned nElts = 0;
  switch (type) {
    case 'c': nElts = 8; break;
    case 's': nElts = 4; break;
    case 'i': nElts = 2; break;
    case 'l': nElts = 1; break;
    case 'h': nElts = 4; break;
    case 'f': nElts = 2; break;
  }
  nElts <<= quad;
  
  if (structTypes)
    s += ".val";
  
  s += " = ";

  std::string a, b, c;
  if (proto.size() > 1)
    a = (structTypes && proto[1] != 'l' && proto[1] != 's') ? "a.val" : "a";
  b = structTypes ? "b.val" : "b";
  c = structTypes ? "c.val" : "c";
  
  switch(op) {
  case OpAdd:
    s += a + " + " + b;
    break;
  case OpSub:
    s += a + " - " + b;
    break;
  case OpMul:
    s += a + " * " + b;
    break;
  case OpMla:
    s += a + " + ( " + b + " * " + c + " )";
    break;
  case OpMls:
    s += a + " - ( " + b + " * " + c + " )";
    break;
  case OpEq:
    s += "(__neon_" + ts + ")(" + a + " == " + b + ")";
    break;
  case OpGe:
    s += "(__neon_" + ts + ")(" + a + " >= " + b + ")";
    break;
  case OpLe:
    s += "(__neon_" + ts + ")(" + a + " <= " + b + ")";
    break;
  case OpGt:
    s += "(__neon_" + ts + ")(" + a + " > " + b + ")";
    break;
  case OpLt:
    s += "(__neon_" + ts + ")(" + a + " < " + b + ")";
    break;
  case OpNeg:
    s += " -" + a;
    break;
  case OpNot:
    s += " ~" + a;
    break;
  case OpAnd:
    s += a + " & " + b;
    break;
  case OpOr:
    s += a + " | " + b;
    break;
  case OpXor:
    s += a + " ^ " + b;
    break;
  case OpAndNot:
    s += a + " & ~" + b;
    break;
  case OpOrNot:
    s += a + " | ~" + b;
    break;
  case OpCast:
    s += "(__neon_" + ts + ")" + a;
    break;
  case OpConcat:
    s += "__builtin_shufflevector((__neon_int64x1_t)" + a;
    s += ", (__neon_int64x1_t)" + b + ", 0, 1)";
    break;
  case OpHi:
    s += "(__neon_int64x1_t)(((__neon_int64x2_t)" + a + ")[1])";
    break;
  case OpLo:
    s += "(__neon_int64x1_t)(((__neon_int64x2_t)" + a + ")[0])";
    break;
  case OpDup:
    s += "(__neon_" + ts + "){ ";
    for (unsigned i = 0; i != nElts; ++i) {
      s += a;
      if ((i + 1) < nElts)
        s += ", ";
    }
    s += " }";
    break;
  default:
    throw "unknown OpKind!";
    break;
  }
  s += "; return r;";
  return s;
}

static unsigned GetNeonEnum(const std::string &proto, StringRef typestr) {
  unsigned mod = proto[0];
  unsigned ret = 0;

  if (mod == 'v' || mod == 'f')
    mod = proto[1];

  bool quad = false;
  bool poly = false;
  bool usgn = false;
  bool scal = false;
  bool cnst = false;
  bool pntr = false;
  
  // base type to get the type string for.
  char type = ClassifyType(typestr, quad, poly, usgn);
  
  // Based on the modifying character, change the type and width if necessary.
  type = ModType(mod, type, quad, poly, usgn, scal, cnst, pntr);
  
  if (usgn)
    ret |= 0x08;
  if (quad)
    ret |= 0x10;
  
  switch (type) {
    case 'c': 
      ret |= poly ? 5 : 0;
      break;
    case 's':
      ret |= poly ? 6 : 1;
      break;
    case 'i':
      ret |= 2;
      break;
    case 'l':
      ret |= 3;
      break;
    case 'h':
      ret |= 7;
      break;
    case 'f':
      ret |= 4;
      break;
    default:
      throw "unhandled type!";
      break;
  }
  return ret;
}

// Generate the definition for this intrinsic, e.g. __builtin_neon_cls(a)
// If structTypes is true, the NEON types are structs of vector types rather
// than vector types, and the call becomes __builtin_neon_cls(a.val)
static std::string GenBuiltin(const std::string &name, const std::string &proto,
                              StringRef typestr, ClassKind ck,
                              bool structTypes = true) {
  char arg = 'a';
  std::string s;

  bool unioning = (proto[0] == '2' || proto[0] == '3' || proto[0] == '4');
  bool define = proto.find('i') != std::string::npos;

  // If all types are the same size, bitcasting the args will take care 
  // of arg checking.  The actual signedness etc. will be taken care of with
  // special enums.
  if (proto.find('s') == std::string::npos)
    ck = ClassB;

  if (proto[0] != 'v') {
    std::string ts = TypeString(proto[0], typestr);
    
    if (define) {
      if (proto[0] != 's')
        s += "(" + ts + "){(__neon_" + ts + ")";
    } else {
      if (unioning) {
        s += "union { ";
        s += TypeString(proto[0], typestr, true) + " val; ";
        s += TypeString(proto[0], typestr, false) + " s; ";
        s += "} r;";
      } else {
        s += ts;
      }
      
      s += " r; r";
      if (structTypes && proto[0] != 's' && proto[0] != 'i' && proto[0] != 'l')
        s += ".val";
      
      s += " = ";
    }
  }    
  
  s += "__builtin_neon_";
  s += MangleName(name, typestr, ck);
  s += "(";
  
  for (unsigned i = 1, e = proto.size(); i != e; ++i, ++arg) {
    // Handle multiple-vector values specially, emitting each subvector as an
    // argument to the __builtin.
    if (structTypes && (proto[i] == '2' || proto[i] == '3' || proto[i] == '4')){
      for (unsigned vi = 0, ve = proto[i] - '0'; vi != ve; ++vi) {
        s.push_back(arg);
        s += ".val[" + utostr(vi) + "]";
        if ((vi + 1) < ve)
          s += ", ";
      }
      if ((i + 1) < e)
        s += ", ";

      continue;
    }
    
    // Parenthesize the args from the macro.
    if (define)
      s.push_back('(');
    s.push_back(arg);
    if (define)
      s.push_back(')');
    
    if (structTypes && proto[i] != 's' && proto[i] != 'i' && proto[i] != 'l' &&
        proto[i] != 'p' && proto[i] != 'c') {
      s += ".val";
    }
    if ((i + 1) < e)
      s += ", ";
  }
  
  // Extra constant integer to hold type class enum for this function, e.g. s8
  if (ck == ClassB)
    s += ", " + utostr(GetNeonEnum(proto, typestr));
  
  if (define)
    s += ")";
  else
    s += ");";

  if (proto[0] != 'v') {
    if (define) {
      if (proto[0] != 's')
        s += "}";
    } else {
      if (unioning)
        s += " return r.s;";
      else
        s += " return r;";
    }
  }
  return s;
}

static std::string GenBuiltinDef(const std::string &name, 
                                 const std::string &proto,
                                 StringRef typestr, ClassKind ck) {
  std::string s("BUILTIN(__builtin_neon_");

  // If all types are the same size, bitcasting the args will take care 
  // of arg checking.  The actual signedness etc. will be taken care of with
  // special enums.
  if (proto.find('s') == std::string::npos)
    ck = ClassB;
  
  s += MangleName(name, typestr, ck);
  s += ", \"";
  
  for (unsigned i = 0, e = proto.size(); i != e; ++i)
    s += BuiltinTypeString(proto[i], typestr, ck, i == 0);

  // Extra constant integer to hold type class enum for this function, e.g. s8
  if (ck == ClassB)
    s += "i";
  
  s += "\", \"n\")";
  return s;
}

void NeonEmitter::run(raw_ostream &OS) {
  EmitSourceFileHeader("ARM NEON Header", OS);
  
  // FIXME: emit license into file?
  
  OS << "#ifndef __ARM_NEON_H\n";
  OS << "#define __ARM_NEON_H\n\n";
  
  OS << "#ifndef __ARM_NEON__\n";
  OS << "#error \"NEON support not enabled\"\n";
  OS << "#endif\n\n";

  OS << "#include <stdint.h>\n\n";

  // Emit NEON-specific scalar typedefs.
  // FIXME: probably need to do something better for polynomial types.
  // FIXME: is this the correct thing to do for float16?
  OS << "typedef float float32_t;\n";
  OS << "typedef uint8_t poly8_t;\n";
  OS << "typedef uint16_t poly16_t;\n";
  OS << "typedef uint16_t float16_t;\n";

  // Emit Neon vector typedefs.
  std::string TypedefTypes("cQcsQsiQilQlUcQUcUsQUsUiQUiUlQUlhQhfQfPcQPcPsQPs");
  SmallVector<StringRef, 24> TDTypeVec;
  ParseTypes(0, TypedefTypes, TDTypeVec);

  // Emit vector typedefs.
  for (unsigned v = 1; v != 5; ++v) {
    for (unsigned i = 0, e = TDTypeVec.size(); i != e; ++i) {
      bool dummy, quad = false;
      (void) ClassifyType(TDTypeVec[i], quad, dummy, dummy);
      OS << "typedef __attribute__(( __vector_size__(";
      
      OS << utostr(8*v*(quad ? 2 : 1)) << ") )) ";
      if (!quad)
        OS << " ";
      
      OS << TypeString('s', TDTypeVec[i]);
      OS << " __neon_";
      
      char t = (v == 1) ? 'd' : '0' + v;
      OS << TypeString(t, TDTypeVec[i]) << ";\n";
    }
  }
  OS << "\n";

  // Emit struct typedefs.
  for (unsigned vi = 1; vi != 5; ++vi) {
    for (unsigned i = 0, e = TDTypeVec.size(); i != e; ++i) {
      std::string ts = TypeString('d', TDTypeVec[i]);
      std::string vs = (vi > 1) ? TypeString('0' + vi, TDTypeVec[i]) : ts;
      OS << "typedef struct __" << vs << " {\n";
      OS << "  __neon_" << ts << " val";
      if (vi > 1)
        OS << "[" << utostr(vi) << "]";
      OS << ";\n} " << vs << ";\n\n";
    }
  }
  
  OS << "#define __ai static __attribute__((__always_inline__))\n\n";

  std::vector<Record*> RV = Records.getAllDerivedDefinitions("Inst");
  
  // Unique the return+pattern types, and assign them.
  for (unsigned i = 0, e = RV.size(); i != e; ++i) {
    Record *R = RV[i];
    std::string name = LowercaseString(R->getName());
    std::string Proto = R->getValueAsString("Prototype");
    std::string Types = R->getValueAsString("Types");
    
    SmallVector<StringRef, 16> TypeVec;
    ParseTypes(R, Types, TypeVec);
    
    OpKind k = OpMap[R->getValueAsDef("Operand")->getName()];
    
    bool define = Proto.find('i') != std::string::npos;
    
    for (unsigned ti = 0, te = TypeVec.size(); ti != te; ++ti) {
      assert(!Proto.empty() && "");
      
      // static always inline + return type
      if (define)
        OS << "#define";
      else
        OS << "__ai " << TypeString(Proto[0], TypeVec[ti]);
      
      // Function name with type suffix
      OS << " " << MangleName(name, TypeVec[ti], ClassS);
      
      // Function arguments
      OS << GenArgs(Proto, TypeVec[ti]);
      
      // Definition.
      if (define)
        OS << " ";
      else
        OS << " { ";
      
      if (k != OpNone) {
        OS << GenOpString(k, Proto, TypeVec[ti]);
      } else {
        if (R->getSuperClasses().size() < 2)
          throw TGError(R->getLoc(), "Builtin has no class kind");
        
        ClassKind ck = ClassMap[R->getSuperClasses()[1]];

        if (ck == ClassNone)
          throw TGError(R->getLoc(), "Builtin has no class kind");
        OS << GenBuiltin(name, Proto, TypeVec[ti], ck);
      }
      if (!define)
        OS << " }";
      OS << "\n";
    }
    OS << "\n";
  }
  OS << "#undef __ai\n\n";
  OS << "#endif /* __ARM_NEON_H */\n";
}

void NeonEmitter::runHeader(raw_ostream &OS) {
  std::vector<Record*> RV = Records.getAllDerivedDefinitions("Inst");

  StringMap<OpKind> EmittedMap;
  
  for (unsigned i = 0, e = RV.size(); i != e; ++i) {
    Record *R = RV[i];

    OpKind k = OpMap[R->getValueAsDef("Operand")->getName()];
    if (k != OpNone)
      continue;
    
    std::string name = LowercaseString(R->getName());
    std::string Proto = R->getValueAsString("Prototype");
    std::string Types = R->getValueAsString("Types");

    SmallVector<StringRef, 16> TypeVec;
    ParseTypes(R, Types, TypeVec);

    if (R->getSuperClasses().size() < 2)
      throw TGError(R->getLoc(), "Builtin has no class kind");
    
    ClassKind ck = ClassMap[R->getSuperClasses()[1]];
    
    for (unsigned ti = 0, te = TypeVec.size(); ti != te; ++ti) {
      std::string bd = GenBuiltinDef(name, Proto, TypeVec[ti], ck);
      if (EmittedMap.count(bd))
        continue;
      
      EmittedMap[bd] = OpNone;
      OS << bd << "\n";
    }
  }
}
