//===-- UpgradeParser.y - Upgrade parser for llvm assmbly -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the bison parser for LLVM 1.9 assembly language.
//
//===----------------------------------------------------------------------===//

%{
#include "ParserInternals.h"
#include <llvm/ADT/StringExtras.h>
#include <algorithm>
#include <map>
#include <utility>
#include <iostream>
#include <cassert>

#define YYERROR_VERBOSE 1
#define YYINCLUDED_STDLIB_H
#define YYDEBUG 1
#define UPGRADE_SETCOND_OPS 0

int yylex();                       // declaration" of xxx warnings.
int yyparse();
extern int yydebug;

static std::string CurFilename;
static std::ostream *O = 0;
std::istream* LexInput = 0;
unsigned SizeOfPointer = 32;
static uint64_t unique = 1;

typedef std::vector<TypeInfo> TypeVector;
static TypeVector EnumeratedTypes;
typedef std::map<std::string,TypeInfo> TypeMap;
static TypeMap NamedTypes;

void destroy(ValueList* VL) {
  while (!VL->empty()) {
    ValueInfo& VI = VL->back();
    VI.destroy();
    VL->pop_back();
  }
  delete VL;
}

void UpgradeAssembly(const std::string &infile, std::istream& in, 
                     std::ostream &out, bool debug)
{
  Upgradelineno = 1; 
  CurFilename = infile;
  LexInput = &in;
  yydebug = debug;
  O = &out;

  if (yyparse()) {
    std::cerr << "Parse failed.\n";
    exit(1);
  }
}

static void ResolveType(TypeInfo& Ty) {
  if (Ty.oldTy == UnresolvedTy) {
    TypeMap::iterator I = NamedTypes.find(*Ty.newTy);
    if (I != NamedTypes.end()) {
      Ty.oldTy = I->second.oldTy;
      Ty.elemTy = I->second.elemTy;
    } else {
      std::string msg("Can't resolve type: ");
      msg += *Ty.newTy;
      yyerror(msg.c_str());
    }
  } else if (Ty.oldTy == NumericTy) {
    unsigned ref = atoi(&((Ty.newTy->c_str())[1])); // Skip the '\\'
    if (ref < EnumeratedTypes.size()) {
      Ty.oldTy = EnumeratedTypes[ref].oldTy;
      Ty.elemTy = EnumeratedTypes[ref].elemTy;
    } else {
      std::string msg("Can't resolve type: ");
      msg += *Ty.newTy;
      yyerror(msg.c_str());
    }
  }
  // otherwise its already resolved.
}

static const char* getCastOpcode(
  std::string& Source, const TypeInfo& SrcTy, const TypeInfo& DstTy) 
{
  unsigned SrcBits = SrcTy.getBitWidth();
  unsigned DstBits = DstTy.getBitWidth();
  const char* opcode = "bitcast";
  // Run through the possibilities ...
  if (DstTy.isIntegral()) {                        // Casting to integral
    if (SrcTy.isIntegral()) {                      // Casting from integral
      if (DstBits < SrcBits)
        opcode = "trunc";
      else if (DstBits > SrcBits) {                // its an extension
        if (SrcTy.isSigned())
          opcode ="sext";                          // signed -> SEXT
        else
          opcode = "zext";                         // unsigned -> ZEXT
      } else {
        opcode = "bitcast";                        // Same size, No-op cast
      }
    } else if (SrcTy.isFloatingPoint()) {          // Casting from floating pt
      if (DstTy.isSigned()) 
        opcode = "fptosi";                         // FP -> sint
      else
        opcode = "fptoui";                         // FP -> uint 
    } else if (SrcTy.isPacked()) {
      assert(DstBits == SrcTy.getBitWidth() &&
               "Casting packed to integer of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(SrcTy.isPointer() &&
             "Casting from a value that is not first-class type");
      opcode = "ptrtoint";                         // ptr -> int
    }
  } else if (DstTy.isFloatingPoint()) {           // Casting to floating pt
    if (SrcTy.isIntegral()) {                     // Casting from integral
      if (SrcTy.isSigned())
        opcode = "sitofp";                         // sint -> FP
      else
        opcode = "uitofp";                         // uint -> FP
    } else if (SrcTy.isFloatingPoint()) {         // Casting from floating pt
      if (DstBits < SrcBits) {
        opcode = "fptrunc";                        // FP -> smaller FP
      } else if (DstBits > SrcBits) {
        opcode = "fpext";                          // FP -> larger FP
      } else  {
        opcode ="bitcast";                         // same size, no-op cast
      }
    } else if (SrcTy.isPacked()) {
      assert(DstBits == SrcTy.getBitWidth() &&
             "Casting packed to floating point of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(0 && "Casting pointer or non-first class to float");
    }
  } else if (DstTy.isPacked()) {
    if (SrcTy.isPacked()) {
      assert(DstTy.getBitWidth() == SrcTy.getBitWidth() &&
             "Casting packed to packed of different widths");
      opcode = "bitcast";                          // packed -> packed
    } else if (DstTy.getBitWidth() == SrcBits) {
      opcode = "bitcast";                          // float/int -> packed
    } else {
      assert(!"Illegal cast to packed (wrong type or size)");
    }
  } else if (DstTy.isPointer()) {
    if (SrcTy.isPointer()) {
      opcode = "bitcast";                          // ptr -> ptr
    } else if (SrcTy.isIntegral()) {
      opcode = "inttoptr";                         // int -> ptr
    } else {
      assert(!"Casting invalid type to pointer");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }
  return opcode;
}

static std::string getCastUpgrade(
  const std::string& Src, TypeInfo& SrcTy, TypeInfo& DstTy, bool isConst)
{
  std::string Result;
  std::string Source = Src;
  if (SrcTy.isFloatingPoint() && DstTy.isPointer()) {
    // fp -> ptr cast is no longer supported but we must upgrade this
    // by doing a double cast: fp -> int -> ptr
    if (isConst)
      Source = "ulong fptoui(" + Source + " to ulong)";
    else {
      *O << "    %cast_upgrade" << unique << " = fptoui " << Source 
         << " to ulong\n";
      Source = "ulong %cast_upgrade" + llvm::utostr(unique);
    }
    // Update the SrcTy for the getCastOpcode call below
    SrcTy.destroy();
    SrcTy.newTy = new std::string("ulong");
    SrcTy.oldTy = ULongTy;
  } else if (DstTy.oldTy == BoolTy) {
    // cast ptr %x to  bool was previously defined as setne ptr %x, null
    // The ptrtoint semantic is to truncate, not compare so we must retain
    // the original intent by replace the cast with a setne
    const char* comparator = SrcTy.isPointer() ? ", null" : 
      (SrcTy.isFloatingPoint() ? ", 0.0" : ", 0");
    if (isConst) 
      Result = "setne (" + Source + comparator + ")";
    else
      Result = "setne " + Source + comparator;
    return Result; // skip cast processing below
  }
  ResolveType(SrcTy);
  ResolveType(DstTy);
  std::string Opcode(getCastOpcode(Source, SrcTy, DstTy));
  if (isConst)
    Result += Opcode + "( " + Source + " to " + *DstTy.newTy + ")";
  else
    Result += Opcode + " " + Source + " to " + *DstTy.newTy;
  return Result;
}

const char* getDivRemOpcode(const std::string& opcode, const TypeInfo& TI) {
  const char* op = opcode.c_str();
  TypeInfo Ty = TI;
  ResolveType(Ty);
  if (Ty.isPacked())
    Ty.oldTy = Ty.getElementType();
  if (opcode == "div")
    if (Ty.isFloatingPoint())
      op = "fdiv";
    else if (Ty.isUnsigned())
      op = "udiv";
    else if (Ty.isSigned())
      op = "sdiv";
    else
      yyerror("Invalid type for div instruction");
  else if (opcode == "rem")
    if (Ty.isFloatingPoint())
      op = "frem";
    else if (Ty.isUnsigned())
      op = "urem";
    else if (Ty.isSigned())
      op = "srem";
    else
      yyerror("Invalid type for rem instruction");
  return op;
}

std::string 
getCompareOp(const std::string& setcc, const TypeInfo& TI) {
  assert(setcc.length() == 5);
  char cc1 = setcc[3];
  char cc2 = setcc[4];
  assert(cc1 == 'e' || cc1 == 'n' || cc1 == 'l' || cc1 == 'g');
  assert(cc2 == 'q' || cc2 == 'e' || cc2 == 'e' || cc2 == 't');
  std::string result("xcmp xxx");
  result[6] = cc1;
  result[7] = cc2;
  if (TI.isFloatingPoint()) {
    result[0] = 'f';
    result[5] = 'o'; // FIXME: Always map to ordered comparison ?
  } else if (TI.isIntegral() || TI.isPointer()) {
    result[0] = 'i';
    if ((cc1 == 'e' && cc2 == 'q') || (cc1 == 'n' && cc2 == 'e'))
      result.erase(5,1);
    else if (TI.isSigned())
      result[5] = 's';
    else if (TI.isUnsigned() || TI.isPointer())
      result[5] = 'u';
    else
      yyerror("Invalid integral type for setcc");
  }
  return result;
}

%}

%file-prefix="UpgradeParser"

%union {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
}

%token <Type>   VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token <Type>   FLOAT DOUBLE LABEL 
%token <String> OPAQUE ESINT64VAL EUINT64VAL SINTVAL UINTVAL FPVAL
%token <String> NULL_TOK UNDEF ZEROINITIALIZER TRUETOK FALSETOK
%token <String> TYPE VAR_ID LABELSTR STRINGCONSTANT
%token <String> IMPLEMENTATION BEGINTOK ENDTOK
%token <String> DECLARE GLOBAL CONSTANT SECTION VOLATILE
%token <String> TO DOTDOTDOT CONST INTERNAL LINKONCE WEAK 
%token <String> DLLIMPORT DLLEXPORT EXTERN_WEAK APPENDING
%token <String> NOT EXTERNAL TARGET TRIPLE ENDIAN POINTERSIZE LITTLE BIG
%token <String> ALIGN UNINITIALIZED
%token <String> DEPLIBS CALL TAIL ASM_TOK MODULE SIDEEFFECT
%token <String> CC_TOK CCC_TOK CSRETCC_TOK FASTCC_TOK COLDCC_TOK
%token <String> X86_STDCALLCC_TOK X86_FASTCALLCC_TOK
%token <String> DATALAYOUT
%token <String> RET BR SWITCH INVOKE EXCEPT UNWIND UNREACHABLE
%token <String> ADD SUB MUL DIV UDIV SDIV FDIV REM UREM SREM FREM AND OR XOR
%token <String> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comparators
%token <String> ICMP FCMP EQ NE SLT SGT SLE SGE OEQ ONE OLT OGT OLE OGE 
%token <String> ORD UNO UEQ UNE ULT UGT ULE UGE
%token <String> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR
%token <String> PHI_TOK SELECT SHL SHR ASHR LSHR VAARG
%token <String> EXTRACTELEMENT INSERTELEMENT SHUFFLEVECTOR
%token <String> CAST TRUNC ZEXT SEXT FPTRUNC FPEXT FPTOUI FPTOSI UITOFP SITOFP 
%token <String> PTRTOINT INTTOPTR BITCAST

%type <String> OptAssign OptLinkage OptCallingConv OptAlign OptCAlign 
%type <String> SectionString OptSection GlobalVarAttributes GlobalVarAttribute
%type <String> ArgTypeListI ConstExpr DefinitionList
%type <String> ConstPool TargetDefinition LibrariesDefinition LibList OptName
%type <String> ArgVal ArgListH ArgList FunctionHeaderH BEGIN FunctionHeader END
%type <String> Function FunctionProto BasicBlock TypeListI
%type <String> InstructionList BBTerminatorInst JumpTable Inst PHIList
%type <String> OptTailCall InstVal OptVolatile Unwind
%type <String> MemoryInst SymbolicValueRef OptSideEffect GlobalType
%type <String> FnDeclareLinkage BasicBlockList BigOrLittle AsmBlock
%type <String> Name ConstValueRef ConstVector External
%type <String> ShiftOps SetCondOps LogicalOps ArithmeticOps CastOps CompareOps
%type <String> Predicates

%type <ValList> ValueRefList ValueRefListE IndexList

%type <Type> IntType SIntType UIntType FPType TypesV Types 
%type <Type> PrimType UpRTypesV UpRTypes

%type <String> IntVal EInt64Val 
%type <Const>  ConstVal

%type <Value> ValueRef ResolvedVal 

%start Module

%%

// Handle constant integer size restriction and conversion...
IntVal : SINTVAL | UINTVAL ;
EInt64Val : ESINT64VAL | EUINT64VAL;

// Operations that are notably excluded from this list include:
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
ArithmeticOps: ADD | SUB | MUL | DIV | UDIV | SDIV | FDIV 
             | REM | UREM | SREM | FREM;
LogicalOps   : AND | OR | XOR;
SetCondOps   : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE;
CompareOps   : ICMP | FCMP;
Predicates   : EQ | NE | SLT | SGT | SLE | SGE | ULT | UGT | ULE | UGE
             | OEQ | ONE | OLT | OGT | OLE | OGE | ORD | UNO | UEQ | UNE  ;
ShiftOps     : SHL | SHR | ASHR | LSHR;
CastOps      : TRUNC | ZEXT | SEXT | FPTRUNC | FPEXT | FPTOUI | FPTOSI | 
               UITOFP | SITOFP | PTRTOINT | INTTOPTR | BITCAST | CAST
             ;

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType :  LONG |  INT |  SHORT | SBYTE;
UIntType : ULONG | UINT | USHORT | UBYTE;
IntType  : SIntType | UIntType;
FPType   : FLOAT | DOUBLE;

// OptAssign - Value producing statements have an optional assignment component
OptAssign : Name '=' {
    $$ = $1;
  }
  | /*empty*/ {
    $$ = new std::string(""); 
  };

OptLinkage 
  : INTERNAL | LINKONCE | WEAK | APPENDING | DLLIMPORT | DLLEXPORT 
  | EXTERN_WEAK 
  | /*empty*/   { $$ = new std::string(""); } ;

OptCallingConv 
  : CCC_TOK | CSRETCC_TOK | FASTCC_TOK | COLDCC_TOK | X86_STDCALLCC_TOK 
  | X86_FASTCALLCC_TOK 
  | CC_TOK EUINT64VAL { 
    *$1 += *$2; 
    delete $2;
    $$ = $1; 
    }
  | /*empty*/ { $$ = new std::string(""); } ;

// OptAlign/OptCAlign - An optional alignment, and an optional alignment with
// a comma before it.
OptAlign 
  : /*empty*/        { $$ = new std::string(); }
  | ALIGN EUINT64VAL { *$1 += " " + *$2; delete $2; $$ = $1; };
         ;
OptCAlign 
  : /*empty*/            { $$ = new std::string(); } 
  | ',' ALIGN EUINT64VAL { 
    $2->insert(0, ", "); 
    *$2 += " " + *$3;
    delete $3;
    $$ = $2;
  };

SectionString 
  : SECTION STRINGCONSTANT { 
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  };

OptSection : /*empty*/     { $$ = new std::string(); } 
           | SectionString;

GlobalVarAttributes 
    : /* empty */ { $$ = new std::string(); } 
    | ',' GlobalVarAttribute GlobalVarAttributes  {
      $2->insert(0, ", ");
      if (!$3->empty())
        *$2 += " " + *$3;
      delete $3;
      $$ = $2;
    };

GlobalVarAttribute 
    : SectionString 
    | ALIGN EUINT64VAL {
      *$1 += " " + *$2;
      delete $2;
      $$ = $1;
    };

//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (function returning void for example).  To have
// access to it, a user must explicitly use TypesV.
//

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    : Types    | VOID ;
UpRTypesV : UpRTypes | VOID ; 
Types     : UpRTypes ;

// Derived types are added later...
//
PrimType : BOOL | SBYTE | UBYTE | SHORT  | USHORT | INT   | UINT ;
PrimType : LONG | ULONG | FLOAT | DOUBLE | LABEL;
UpRTypes 
  : OPAQUE { 
    $$.newTy = $1; 
    $$.oldTy = OpaqueTy; 
  } 
  | SymbolicValueRef { 
    $$.newTy = $1;
    $$.oldTy = UnresolvedTy;
  }
  | PrimType { 
    $$ = $1; 
  }
  | '\\' EUINT64VAL {                   // Type UpReference
    $2->insert(0, "\\");
    $$.newTy = $2;
    $$.oldTy = NumericTy;
  }
  | UpRTypesV '(' ArgTypeListI ')' {           // Function derived type?
    *$1.newTy += "( " + *$3 + " )";
    delete $3;
    $$.newTy = $1.newTy;
    $$.oldTy = FunctionTy;
  }
  | '[' EUINT64VAL 'x' UpRTypes ']' {          // Sized array type?
    $2->insert(0,"[ ");
    *$2 += " x " + *$4.newTy + " ]";
    delete $4.newTy;
    $$.newTy = $2;
    $$.oldTy = ArrayTy;
    $$.elemTy = $4.oldTy;
  }
  | '<' EUINT64VAL 'x' UpRTypes '>' {          // Packed array type?
    $2->insert(0,"< ");
    *$2 += " x " + *$4.newTy + " >";
    delete $4.newTy;
    $$.newTy = $2;
    $$.oldTy = PackedTy;
    $$.elemTy = $4.oldTy;
  }
  | '{' TypeListI '}' {                        // Structure type?
    $2->insert(0, "{ ");
    *$2 += " }";
    $$.newTy = $2;
    $$.oldTy = StructTy;
  }
  | '{' '}' {                                  // Empty structure type?
    $$.newTy = new std::string("{}");
    $$.oldTy = StructTy;
  }
  | UpRTypes '*' {                             // Pointer type?
    *$1.newTy += '*';
    $$.elemTy = $1.oldTy;
    $1.oldTy = PointerTy;
    $$ = $1;
  };

// TypeList - Used for struct declarations and as a basis for function type 
// declaration type lists
//
TypeListI 
  : UpRTypes {
    $$ = $1.newTy;
  }
  | TypeListI ',' UpRTypes {
    *$1 += ", " + *$3.newTy;
    delete $3.newTy;
    $$ = $1;
  };

// ArgTypeList - List of types for a function type declaration...
ArgTypeListI 
  : TypeListI 
  | TypeListI ',' DOTDOTDOT {
    *$1 += ", ...";
    delete $3;
    $$ = $1;
  }
  | DOTDOTDOT {
    $$ = $1;
  }
  | /*empty*/ {
    $$ = new std::string();
  };

// ConstVal - The various declarations that go into the constant pool.  This
// production is used ONLY to represent constants that show up AFTER a 'const',
// 'constant' or 'global' token at global scope.  Constants that can be inlined
// into other expressions (such as integers and constexprs) are handled by the
// ResolvedVal, ValueRef and ConstValueRef productions.
//
ConstVal: Types '[' ConstVector ']' { // Nonempty unsized arr
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " [ " + *$3 + " ]";
    delete $3;
  }
  | Types '[' ']' {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += "[ ]";
  }
  | Types 'c' STRINGCONSTANT {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " c" + *$3;
    delete $3;
  }
  | Types '<' ConstVector '>' { // Nonempty unsized arr
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " < " + *$3 + " >";
    delete $3;
  }
  | Types '{' ConstVector '}' {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " { " + *$3 + " }";
    delete $3;
  }
  | Types '{' '}' {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " {}";
  }
  | Types NULL_TOK {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst +=  " " + *$2;
    delete $2;
  }
  | Types UNDEF {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | Types SymbolicValueRef {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | Types ConstExpr {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | Types ZEROINITIALIZER {
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | SIntType EInt64Val {      // integral constants
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | UIntType EUINT64VAL {            // integral constants
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | BOOL TRUETOK {                      // Boolean constants
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | BOOL FALSETOK {                     // Boolean constants
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | FPType FPVAL {                   // Float & Double constants
    $$.type = $1;
    $$.cnst = new std::string(*$1.newTy);
    *$$.cnst += " " + *$2;
    delete $2;
  };


ConstExpr: CastOps '(' ConstVal TO Types ')' {
    std::string source = *$3.cnst;
    TypeInfo DstTy = $5;
    ResolveType(DstTy);
    if (*$1 == "cast") {
      // Call getCastUpgrade to upgrade the old cast
      $$ = new std::string(getCastUpgrade(source, $3.type, $5, true));
    } else {
      // Nothing to upgrade, just create the cast constant expr
      $$ = new std::string(*$1);
      *$$ += "( " + source + " to " + *$5.newTy + ")";
    }
    delete $1; $3.destroy(); delete $4; $5.destroy();
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
    *$1 += "(" + *$3.cnst;
    for (unsigned i = 0; i < $4->size(); ++i) {
      ValueInfo& VI = (*$4)[i];
      *$1 += ", " + *VI.val;
      VI.destroy();
    }
    *$1 += ")";
    $$ = $1;
    $3.destroy();
    delete $4;
  }
  | SELECT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + "," + *$7.cnst + ")";
    $3.destroy(); $5.destroy(); $7.destroy();
    $$ = $1;
  }
  | ArithmeticOps '(' ConstVal ',' ConstVal ')' {
    const char* op = getDivRemOpcode(*$1, $3.type); 
    $$ = new std::string(op);
    *$$ += "(" + *$3.cnst + "," + *$5.cnst + ")";
    delete $1; $3.destroy(); $5.destroy();
  }
  | LogicalOps '(' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | SetCondOps '(' ConstVal ',' ConstVal ')' {
#if UPGRADE_SETCOND_OPS
    *$1 = getCompareOp(*$1, $3.type);
#endif
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | CompareOps Predicates '(' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$2 + "," + *$4.cnst + "," + *$6.cnst + ")";
    delete $2; $4.destroy(); $6.destroy();
    $$ = $1;
  }
  | ShiftOps '(' ConstVal ',' ConstVal ')' {
    const char* shiftop = $1->c_str();
    if (*$1 == "shr")
      shiftop = ($3.type.isUnsigned()) ? "lshr" : "ashr";
    $$ = new std::string(shiftop);
    *$$ += "(" + *$3.cnst + "," + *$5.cnst + ")";
    delete $1; $3.destroy(); $5.destroy();
  }
  | EXTRACTELEMENT '(' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | INSERTELEMENT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + "," + *$7.cnst + ")";
    $3.destroy(); $5.destroy(); $7.destroy();
    $$ = $1;
  }
  | SHUFFLEVECTOR '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + "," + *$7.cnst + ")";
    $3.destroy(); $5.destroy(); $7.destroy();
    $$ = $1;
  };


// ConstVector - A list of comma separated constants.

ConstVector 
  : ConstVector ',' ConstVal {
    *$1 += ", " + *$3.cnst;
    $3.destroy();
    $$ = $1;
  }
  | ConstVal { $$ = new std::string(*$1.cnst); $1.destroy(); }
  ;


// GlobalType - Match either GLOBAL or CONSTANT for global declarations...
GlobalType : GLOBAL | CONSTANT ;


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module : DefinitionList {
};

// DefinitionList - Top level definitions
//
DefinitionList : DefinitionList Function {
    $$ = 0;
  } 
  | DefinitionList FunctionProto {
    *O << *$2 << "\n";
    delete $2;
    $$ = 0;
  }
  | DefinitionList MODULE ASM_TOK AsmBlock {
    *O << "module asm " << " " << *$4 << "\n";
    $$ = 0;
  }  
  | DefinitionList IMPLEMENTATION {
    *O << "implementation\n";
    $$ = 0;
  }
  | ConstPool { $$ = 0; }

External : EXTERNAL | UNINITIALIZED { $$ = $1; *$$ = "external"; }

// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign TYPE TypesV {
    EnumeratedTypes.push_back($4);
    if (!$2->empty()) {
      NamedTypes[*$2].newTy = new std::string(*$4.newTy);
      NamedTypes[*$2].oldTy = $4.oldTy;
      NamedTypes[*$2].elemTy = $4.elemTy;
      *O << *$2 << " = ";
    }
    *O << "type " << *$4.newTy << "\n";
    delete $2; delete $3; $4.destroy();
    $$ = 0;
  }
  | ConstPool FunctionProto {       // Function prototypes can be in const pool
    *O << *$2 << "\n";
    delete $2;
    $$ = 0;
  }
  | ConstPool MODULE ASM_TOK AsmBlock {  // Asm blocks can be in the const pool
    *O << *$2 << " " << *$3 << " " << *$4 << "\n";
    delete $2; delete $3; delete $4; 
    $$ = 0;
  }
  | ConstPool OptAssign OptLinkage GlobalType ConstVal  GlobalVarAttributes {
    if (!$2->empty())
      *O << *$2 << " = ";
    *O << *$3 << " " << *$4 << " " << *$5.cnst << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; $5.destroy(); delete $6; 
    $$ = 0;
  }
  | ConstPool OptAssign External GlobalType Types  GlobalVarAttributes {
    if (!$2->empty())
      *O << *$2 << " = ";
    *O <<  *$3 << " " << *$4 << " " << *$5.newTy << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; $5.destroy(); delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign DLLIMPORT GlobalType Types  GlobalVarAttributes {
    if (!$2->empty())
      *O << *$2 << " = ";
    *O << *$3 << " " << *$4 << " " << *$5.newTy << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; $5.destroy(); delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign EXTERN_WEAK GlobalType Types  GlobalVarAttributes {
    if (!$2->empty())
      *O << *$2 << " = ";
    *O << *$3 << " " << *$4 << " " << *$5.newTy << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; $5.destroy(); delete $6;
    $$ = 0;
  }
  | ConstPool TARGET TargetDefinition { 
    *O << *$2 << " " << *$3 << "\n";
    delete $2; delete $3;
    $$ = 0;
  }
  | ConstPool DEPLIBS '=' LibrariesDefinition {
    *O << *$2 << " = " << *$4 << "\n";
    delete $2; delete $4;
    $$ = 0;
  }
  | /* empty: end of list */ { 
    $$ = 0;
  };


AsmBlock : STRINGCONSTANT ;

BigOrLittle : BIG | LITTLE 

TargetDefinition 
  : ENDIAN '=' BigOrLittle {
    *$1 += " = " + *$3;
    delete $3;
    $$ = $1;
  }
  | POINTERSIZE '=' EUINT64VAL {
    *$1 += " = " + *$3;
    if (*$3 == "64")
      SizeOfPointer = 64;
    delete $3;
    $$ = $1;
  }
  | TRIPLE '=' STRINGCONSTANT {
    *$1 += " = " + *$3;
    delete $3;
    $$ = $1;
  }
  | DATALAYOUT '=' STRINGCONSTANT {
    *$1 += " = " + *$3;
    delete $3;
    $$ = $1;
  };

LibrariesDefinition 
  : '[' LibList ']' {
    $2->insert(0, "[ ");
    *$2 += " ]";
    $$ = $2;
  };

LibList 
  : LibList ',' STRINGCONSTANT {
    *$1 += ", " + *$3;
    delete $3;
    $$ = $1;
  }
  | STRINGCONSTANT 
  | /* empty: end of list */ {
    $$ = new std::string();
  };

//===----------------------------------------------------------------------===//
//                       Rules to match Function Headers
//===----------------------------------------------------------------------===//

Name : VAR_ID | STRINGCONSTANT;
OptName : Name | /*empty*/ { $$ = new std::string(); };

ArgVal : Types OptName {
  $$ = $1.newTy;
  if (!$2->empty())
    *$$ += " " + *$2;
  delete $2;
};

ArgListH : ArgListH ',' ArgVal {
    *$1 += ", " + *$3;
    delete $3;
  }
  | ArgVal {
    $$ = $1;
  };

ArgList : ArgListH {
    $$ = $1;
  }
  | ArgListH ',' DOTDOTDOT {
    *$1 += ", ...";
    $$ = $1;
    delete $3;
  }
  | DOTDOTDOT {
    $$ = $1;
  }
  | /* empty */ { $$ = new std::string(); };

FunctionHeaderH : OptCallingConv TypesV Name '(' ArgList ')' 
                  OptSection OptAlign {
    if (!$1->empty()) {
      *$1 += " ";
    }
    *$1 += *$2.newTy + " " + *$3 + "(" + *$5 + ")";
    if (!$7->empty()) {
      *$1 += " " + *$7;
    }
    if (!$8->empty()) {
      *$1 += " " + *$8;
    }
    $2.destroy();
    delete $3;
    delete $5;
    delete $7;
    delete $8;
    $$ = $1;
  };

BEGIN : BEGINTOK { $$ = new std::string("{"); delete $1; }
  | '{' { $$ = new std::string ("{"); }

FunctionHeader : OptLinkage FunctionHeaderH BEGIN {
  if (!$1->empty()) {
    *O << *$1 << " ";
  }
  *O << *$2 << " " << *$3 << "\n";
  delete $1; delete $2; delete $3;
  $$ = 0;
};

END : ENDTOK { $$ = new std::string("}"); delete $1; }
    | '}' { $$ = new std::string("}"); };

Function : FunctionHeader BasicBlockList END {
  if ($2)
    *O << *$2;
  *O << '\n' << *$3 << "\n";
  $$ = 0;
};

FnDeclareLinkage
  : /*default*/ { $$ = new std::string(); }
  | DLLIMPORT    
  | EXTERN_WEAK 
  ;
  
FunctionProto 
  : DECLARE FnDeclareLinkage FunctionHeaderH { 
    if (!$2->empty())
      *$1 += " " + *$2;
    *$1 += " " + *$3;
    delete $2;
    delete $3;
    $$ = $1;
  };

//===----------------------------------------------------------------------===//
//                        Rules to match Basic Blocks
//===----------------------------------------------------------------------===//

OptSideEffect : /* empty */ { $$ = new std::string(); }
  | SIDEEFFECT;

ConstValueRef 
  : ESINT64VAL | EUINT64VAL | FPVAL | TRUETOK | FALSETOK | NULL_TOK | UNDEF
  | ZEROINITIALIZER 
  | '<' ConstVector '>' { 
    $2->insert(0, "<");
    *$2 += ">";
    $$ = $2;
  }
  | ConstExpr 
  | ASM_TOK OptSideEffect STRINGCONSTANT ',' STRINGCONSTANT {
    if (!$2->empty()) {
      *$1 += " " + *$2;
    }
    *$1 += " " + *$3 + ", " + *$5;
    delete $2; delete $3; delete $5;
    $$ = $1;
  };

SymbolicValueRef : IntVal | Name ;

// ValueRef - A reference to a definition... either constant or symbolic
ValueRef 
  : SymbolicValueRef {
    $$.val = $1;
    $$.constant = false;
    $$.type.newTy = 0;
    $$.type.oldTy = UnresolvedTy;
  }
  | ConstValueRef {
    $$.val = $1;
    $$.constant = true;
    $$.type.newTy = 0;
    $$.type.oldTy = UnresolvedTy;
  }
  ;

// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : Types ValueRef {
    $$ = $2;
    $$.type = $1;
    $$.val->insert(0, *$1.newTy + " ");
  };

BasicBlockList : BasicBlockList BasicBlock {
    $$ = 0;
  }
  | BasicBlock { // Do not allow functions with 0 basic blocks   
    $$ = 0;
  };


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock : InstructionList BBTerminatorInst  {
    $$ = 0;
  };

InstructionList : InstructionList Inst {
    *O << "    " << *$2 << "\n";
    delete $2;
    $$ = 0;
  }
  | /* empty */ {
    $$ = 0;
  }
  | LABELSTR {
    *O << *$1 << "\n";
    delete $1;
    $$ = 0;
  };

Unwind : UNWIND | EXCEPT { $$ = $1; *$$ = "unwind"; }

BBTerminatorInst : RET ResolvedVal {              // Return with a result...
    *O << "    " << *$1 << " " << *$2.val << "\n";
    delete $1; $2.destroy();
    $$ = 0;
  }
  | RET VOID {                                       // Return with no result...
    *O << "    " << *$1 << " " << *$2.newTy << "\n";
    delete $1; $2.destroy();
    $$ = 0;
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3.val << "\n";
    delete $1; $2.destroy(); $3.destroy();
    $$ = 0;
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3.val << ", " 
       << *$5.newTy << " " << *$6.val << ", " << *$8.newTy << " " 
       << *$9.val << "\n";
    delete $1; $2.destroy(); $3.destroy(); $5.destroy(); $6.destroy(); 
    $8.destroy(); $9.destroy();
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3.val << ", " 
       << *$5.newTy << " " << *$6.val << " [" << *$8 << " ]\n";
    delete $1; $2.destroy(); $3.destroy(); $5.destroy(); $6.destroy(); 
    delete $8;
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3.val << ", " 
       << *$5.newTy << " " << *$6.val << "[]\n";
    delete $1; $2.destroy(); $3.destroy(); $5.destroy(); $6.destroy();
    $$ = 0;
  }
  | OptAssign INVOKE OptCallingConv TypesV ValueRef '(' ValueRefListE ')'
    TO LABEL ValueRef Unwind LABEL ValueRef {
    *O << "    ";
    if (!$1->empty())
      *O << *$1 << " = ";
    *O << *$2 << " " << *$3 << " " << *$4.newTy << " " << *$5.val << " (";
    for (unsigned i = 0; i < $7->size(); ++i) {
      ValueInfo& VI = (*$7)[i];
      *O << *VI.val;
      if (i+1 < $7->size())
        *O << ", ";
      VI.destroy();
    }
    *O << ") " << *$9 << " " << *$10.newTy << " " << *$11.val << " " 
       << *$12 << " " << *$13.newTy << " " << *$14.val << "\n";
    delete $1; delete $2; delete $3; $4.destroy(); $5.destroy(); delete $7; 
    delete $9; $10.destroy(); $11.destroy(); delete $12; $13.destroy(); 
    $14.destroy(); 
    $$ = 0;
  }
  | Unwind {
    *O << "    " << *$1 << "\n";
    delete $1;
    $$ = 0;
  }
  | UNREACHABLE {
    *O << "    " << *$1 << "\n";
    delete $1;
    $$ = 0;
  };

JumpTable : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    *$1 += " " + *$2.newTy + " " + *$3 + ", " + *$5.newTy + " " + *$6.val;
    $2.destroy(); delete $3; $5.destroy(); $6.destroy();
    $$ = $1;
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $2->insert(0, *$1.newTy + " " );
    *$2 += ", " + *$4.newTy + " " + *$5.val;
    $1.destroy(); $4.destroy(); $5.destroy();
    $$ = $2;
  };

Inst 
  : OptAssign InstVal {
    if (!$1->empty())
      *$1 += " = ";
    *$1 += *$2;
    delete $2;
    $$ = $1; 
  };

PHIList 
  : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    $3.val->insert(0, *$1.newTy + "[");
    *$3.val += "," + *$5.val + "]";
    $1.destroy(); $5.destroy();
    $$ = new std::string(*$3.val);
    $3.destroy();
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    *$1 += ", [" + *$4.val + "," + *$6.val + "]";
    $4.destroy(); $6.destroy();
    $$ = $1;
  };


ValueRefList 
  : ResolvedVal { 
    $$ = new ValueList();
    $$->push_back($1);
  }
  | ValueRefList ',' ResolvedVal {
    $1->push_back($3);
    $$ = $1;
  };

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE 
  : ValueRefList  { $$ = $1; }
  | /*empty*/ { $$ = new ValueList(); }
  ;

OptTailCall 
  : TAIL CALL {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | CALL 
  ;

InstVal : ArithmeticOps Types ValueRef ',' ValueRef {
    const char* op = getDivRemOpcode(*$1, $2); 
    $$ = new std::string(op);
    *$$ += " " + *$2.newTy + " " + *$3.val + ", " + *$5.val;
    delete $1; $2.destroy(); $3.destroy(); $5.destroy();
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    *$1 += " " + *$2.newTy + " " + *$3.val + ", " + *$5.val;
    $2.destroy(); $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | SetCondOps Types ValueRef ',' ValueRef {
#if UPGRADE_SETCOND_OPS
    *$1 = getCompareOp(*$1, $2);
#endif
    *$1 += " " + *$2.newTy + " " + *$3.val + ", " + *$5.val;
    $2.destroy(); $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | CompareOps Predicates Types ValueRef ',' ValueRef ')' {
    *$1 += " " + *$2 + " " + *$4.val + "," + *$6.val + ")";
    delete $2; $4.destroy(); $6.destroy();
    $$ = $1;
  }
  | NOT ResolvedVal {
    *$1 += " " + *$2.val;
    $2.destroy();
    $$ = $1;
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    const char* shiftop = $1->c_str();
    if (*$1 == "shr")
      shiftop = ($2.type.isUnsigned()) ? "lshr" : "ashr";
    $$ = new std::string(shiftop);
    *$$ += " " + *$2.val + ", " + *$4.val;
    delete $1; $2.destroy(); $4.destroy();
  }
  | CastOps ResolvedVal TO Types {
    std::string source = *$2.val;
    TypeInfo SrcTy = $2.type;
    TypeInfo DstTy = $4;
    ResolveType(DstTy);
    $$ = new std::string();
    if (*$1 == "cast") {
      *$$ +=  getCastUpgrade(source, SrcTy, DstTy, false);
    } else {
      *$$ += *$1 + " " + source + " to " + *DstTy.newTy;
    }
    delete $1; $2.destroy();
    delete $3; $4.destroy();
  }
  | SELECT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val + ", " + *$6.val;
    $2.destroy(); $4.destroy(); $6.destroy();
    $$ = $1;
  }
  | VAARG ResolvedVal ',' Types {
    *$1 += " " + *$2.val + ", " + *$4.newTy;
    $2.destroy(); $4.destroy();
    $$ = $1;
  }
  | EXTRACTELEMENT ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val;
    $2.destroy(); $4.destroy();
    $$ = $1;
  }
  | INSERTELEMENT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val + ", " + *$6.val;
    $2.destroy(); $4.destroy(); $6.destroy();
    $$ = $1;
  }
  | SHUFFLEVECTOR ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val + ", " + *$6.val;
    $2.destroy(); $4.destroy(); $6.destroy();
    $$ = $1;
  }
  | PHI_TOK PHIList {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | OptTailCall OptCallingConv TypesV ValueRef '(' ValueRefListE ')'  {
    if (!$2->empty())
      *$1 += " " + *$2;
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$3.newTy + " " + *$4.val + "(";
    for (unsigned i = 0; i < $6->size(); ++i) {
      ValueInfo& VI = (*$6)[i];
      *$1 += *VI.val;
      if (i+1 < $6->size())
        *$1 += ", ";
      VI.destroy();
    }
    *$1 += ")";
    delete $2; $3.destroy(); $4.destroy(); delete $6;
    $$ = $1;
  }
  | MemoryInst ;


// IndexList - List of indices for GEP based instructions...
IndexList 
  : ',' ValueRefList { $$ = $2; }
  | /* empty */ {  $$ = new ValueList(); }
  ;

OptVolatile 
  : VOLATILE 
  | /* empty */ { $$ = new std::string(); }
  ;

MemoryInst : MALLOC Types OptCAlign {
    *$1 += " " + *$2.newTy;
    if (!$3->empty())
      *$1 += " " + *$3;
    $2.destroy(); delete $3;
    $$ = $1;
  }
  | MALLOC Types ',' UINT ValueRef OptCAlign {
    *$1 += " " + *$2.newTy + ", " + *$4.newTy + " " + *$5.val;
    if (!$6->empty())
      *$1 += " " + *$6;
    $2.destroy(); $4.destroy(); $5.destroy(); delete $6;
    $$ = $1;
  }
  | ALLOCA Types OptCAlign {
    *$1 += " " + *$2.newTy;
    if (!$3->empty())
      *$1 += " " + *$3;
    $2.destroy(); delete $3;
    $$ = $1;
  }
  | ALLOCA Types ',' UINT ValueRef OptCAlign {
    *$1 += " " + *$2.newTy + ", " + *$4.newTy + " " + *$5.val;
    if (!$6->empty())
      *$1 += " " + *$6;
    $2.destroy(); $4.destroy(); $5.destroy(); delete $6;
    $$ = $1;
  }
  | FREE ResolvedVal {
    *$1 += " " + *$2.val;
    $2.destroy();
    $$ = $1;
  }
  | OptVolatile LOAD Types ValueRef {
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$2 + " " + *$3.newTy + " " + *$4.val;
    delete $2; $3.destroy(); $4.destroy();
    $$ = $1;
  }
  | OptVolatile STORE ResolvedVal ',' Types ValueRef {
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$2 + " " + *$3.val + ", " + *$5.newTy + " " + *$6.val;
    delete $2; $3.destroy(); $5.destroy(); $6.destroy();
    $$ = $1;
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    // Upgrade the indices
    for (unsigned i = 0; i < $4->size(); ++i) {
      ValueInfo& VI = (*$4)[i];
      if (VI.type.isUnsigned() && !VI.isConstant() && 
          VI.type.getBitWidth() < 64) {
        std::string* old = VI.val;
        *O << "    %gep_upgrade" << unique << " = zext " << *old 
           << " to ulong\n";
        VI.val = new std::string("ulong %gep_upgrade" + llvm::utostr(unique++));
        VI.type.oldTy = ULongTy;
        delete old;
      }
    }
    *$1 += " " + *$2.newTy + " " + *$3.val;
    for (unsigned i = 0; i < $4->size(); ++i) {
      ValueInfo& VI = (*$4)[i];
      *$1 += ", " + *VI.val;
      VI.destroy();
    }
    $2.destroy(); $3.destroy(); delete $4;
    $$ = $1;
  };

%%

int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << errMsg << '\n';
  exit(1);
}
