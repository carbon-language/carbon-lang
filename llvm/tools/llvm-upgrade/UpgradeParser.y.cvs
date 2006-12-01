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
#include <list>
#include <utility>
#include <iostream>

#define YYERROR_VERBOSE 1
#define YYINCLUDED_STDLIB_H
#define YYDEBUG 1

int yylex();                       // declaration" of xxx warnings.
int yyparse();
extern int yydebug;

static std::string CurFilename;
static std::ostream *O = 0;
std::istream* LexInput = 0;
unsigned SizeOfPointer = 32;

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

const char* getCastOpcode(TypeInfo& SrcTy, TypeInfo&DstTy) {
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
      assert(!"Casting pointer to other than pointer or int");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }
  return opcode;
}

%}

%file-prefix="UpgradeParser"

%union {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
}

%token <Type>   VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token <Type>   FLOAT DOUBLE LABEL OPAQUE
%token <String> ESINT64VAL EUINT64VAL SINTVAL UINTVAL FPVAL
%token <String> NULL_TOK UNDEF ZEROINITIALIZER TRUETOK FALSETOK
%token <String> TYPE VAR_ID LABELSTR STRINGCONSTANT
%token <String> IMPLEMENTATION BEGINTOK ENDTOK
%token <String> DECLARE GLOBAL CONSTANT SECTION VOLATILE
%token <String> TO DOTDOTDOT CONST INTERNAL LINKONCE WEAK 
%token <String> DLLIMPORT DLLEXPORT EXTERN_WEAK APPENDING
%token <String> NOT EXTERNAL TARGET TRIPLE ENDIAN POINTERSIZE LITTLE BIG
%token <String> ALIGN
%token <String> DEPLIBS CALL TAIL ASM_TOK MODULE SIDEEFFECT
%token <String> CC_TOK CCC_TOK CSRETCC_TOK FASTCC_TOK COLDCC_TOK
%token <String> X86_STDCALLCC_TOK X86_FASTCALLCC_TOK
%token <String> DATALAYOUT
%token <String> RET BR SWITCH INVOKE UNWIND UNREACHABLE
%token <String> ADD SUB MUL UDIV SDIV FDIV UREM SREM FREM AND OR XOR
%token <String> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comparators
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
%type <String> ValueRefList OptTailCall InstVal IndexList OptVolatile
%type <String> MemoryInst SymbolicValueRef OptSideEffect GlobalType
%type <String> FnDeclareLinkage BasicBlockList BigOrLittle AsmBlock
%type <String> Name ValueRef ValueRefListE ConstValueRef 
%type <String> ShiftOps SetCondOps LogicalOps ArithmeticOps CastOps 

%type <String> ConstVector

%type <Type> IntType SIntType UIntType FPType TypesV Types 
%type <Type> PrimType UpRTypesV UpRTypes

%type <String> IntVal EInt64Val 
%type <Const>  ConstVal

%type <Value> ResolvedVal

%start Module

%%

// Handle constant integer size restriction and conversion...
IntVal : SINTVAL | UINTVAL ;
EInt64Val : ESINT64VAL | EUINT64VAL;

// Operations that are notably excluded from this list include:
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
ArithmeticOps: ADD | SUB | MUL | UDIV | SDIV | FDIV | UREM | SREM | FREM;
LogicalOps   : AND | OR | XOR;
SetCondOps   : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE;
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
    *$1 += " = ";
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
UpRTypes : OPAQUE | PrimType 
         | SymbolicValueRef { 
           $$.newTy = $1; $$.oldTy = OpaqueTy;
         };

// Include derived types in the Types production.
//
UpRTypes : '\\' EUINT64VAL {                   // Type UpReference
    $2->insert(0, "\\");
    $$.newTy = $2;
    $$.oldTy = OpaqueTy;
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
  }
  | '<' EUINT64VAL 'x' UpRTypes '>' {          // Packed array type?
    $2->insert(0,"< ");
    *$2 += " x " + *$4.newTy + " >";
    delete $4.newTy;
    $$.newTy = $2;
    $$.oldTy = PackedTy;
  }
  | '{' TypeListI '}' {                        // Structure type?
    $2->insert(0, "{ ");
    *$2 += " }";
    $$.newTy = $2;
    $$.oldTy = StructTy;
  }
  | '{' '}' {                                  // Empty structure type?
    $$.newTy = new std::string("{ }");
    $$.oldTy = StructTy;
  }
  | UpRTypes '*' {                             // Pointer type?
    *$1.newTy += '*';
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
    *$$.cnst += " [ ]";
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
    // We must infer the cast opcode from the types of the operands. 
    const char *opcode = $1->c_str();
    if (*$1 == "cast")
      opcode = getCastOpcode($3.type, $5);
    $$ = new std::string(opcode);
    *$$ += "(" + *$3.cnst + " " + *$4 + " " + *$5.newTy + ")";
    delete $1; $3.destroy(); delete $4; $5.destroy();
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
    *$1 += "(" + *$3.cnst + " " + *$4 + ")";
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
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | LogicalOps '(' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | SetCondOps '(' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
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
  | ConstPool;

// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign TYPE TypesV {
    *O << *$2 << " " << *$3 << " " << *$4.newTy << "\n";
    // delete $2; delete $3; $4.destroy();
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
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5.cnst << " " 
       << *$6 << "\n";
    delete $2; delete $3; delete $4; $5.destroy(); delete $6; 
    $$ = 0;
  }
  | ConstPool OptAssign EXTERNAL GlobalType Types  GlobalVarAttributes {
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5.newTy 
       << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; $5.destroy(); delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign DLLIMPORT GlobalType Types  GlobalVarAttributes {
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5.newTy 
       << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; $5.destroy(); delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign EXTERN_WEAK GlobalType Types  GlobalVarAttributes {
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5.newTy 
       << " " << *$6 << "\n";
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

BEGIN : BEGINTOK {
    $$ = new std::string("begin");
  }
  | '{' { 
    $$ = new std::string ("{");
  }

FunctionHeader : OptLinkage FunctionHeaderH BEGIN {
  if (!$1->empty()) {
    *O << *$1 << " ";
  }
  *O << *$2 << " " << *$3 << "\n";
  delete $1; delete $2; delete $3;
  $$ = 0;
};

END : ENDTOK { $$ = new std::string("end"); }
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
ValueRef : SymbolicValueRef | ConstValueRef;


// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : Types ValueRef {
    $$.type = $1;
    $$.val = new std::string(*$1.newTy + " ");
    *$$.val += *$2;
    delete $2;
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
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3 << "\n";
    delete $1; $2.destroy(); delete $3;
    $$ = 0;
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3 << ", " 
       << *$5.newTy << " " << *$6 << ", " << *$8.newTy << " " << *$9 << "\n";
    delete $1; $2.destroy(); delete $3; $5.destroy(); delete $6; 
    $8.destroy(); delete $9;
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3 << ", " << *$5.newTy 
       << " " << *$6 << " [" << *$8 << " ]\n";
    delete $1; $2.destroy(); delete $3; $5.destroy(); delete $6; delete $8;
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    *O << "    " << *$1 << " " << *$2.newTy << " " << *$3 << ", " 
       << *$5.newTy << " " << *$6 << "[]\n";
    delete $1; $2.destroy(); delete $3; $5.destroy(); delete $6;
    $$ = 0;
  }
  | OptAssign INVOKE OptCallingConv TypesV ValueRef '(' ValueRefListE ')'
    TO LABEL ValueRef UNWIND LABEL ValueRef {
    *O << "    ";
    if (!$1->empty())
      *O << *$1;
    *O << *$2 << " " << *$3 << " " << *$4.newTy << " " << *$5 << " ("
       << *$7 << ") " << *$9 << " " << *$10.newTy << " " << *$11 << " " 
       << *$12 << " " << *$13.newTy << " " << *$14 << "\n";
    delete $1; delete $2; delete $3; $4.destroy(); delete $5; delete $7; 
    delete $9; $10.destroy(); delete $11; delete $12; $13.destroy(); 
    delete $14; 
    $$ = 0;
  }
  | UNWIND {
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
    *$1 += " " + *$2.newTy + " " + *$3 + ", " + *$5.newTy + " " + *$6;
    $2.destroy(); delete $3; $5.destroy(); delete $6;
    $$ = $1;
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $2->insert(0, *$1.newTy + " " );
    *$2 += ", " + *$4.newTy + " " + *$5;
    $1.destroy(); $4.destroy(); delete $5;
    $$ = $2;
  };

Inst 
  : OptAssign InstVal {
    *$1 += *$2;
    delete $2;
    $$ = $1; 
  };

PHIList 
  : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    $3->insert(0, *$1.newTy + "[");
    *$3 += "," + *$5 + "]";
    $1.destroy(); delete $5;
    $$ = $3;
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    *$1 += ", [" + *$4 + "," + *$6 + "]";
    delete $4; delete $6;
    $$ = $1;
  };


ValueRefList 
  : ResolvedVal { $$ = new std::string(*$1.val); $1.destroy(); }
  | ValueRefList ',' ResolvedVal {
    *$1 += ", " + *$3.val;
    $3.destroy();
    $$ = $1;
  };

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE 
  : ValueRefList 
  | /*empty*/ { $$ = new std::string(); }
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
    *$1 += " " + *$2.newTy + " " + *$3 + ", " + *$5;
    $2.destroy(); delete $3; delete $5;
    $$ = $1;
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    *$1 += " " + *$2.newTy + " " + *$3 + ", " + *$5;
    $2.destroy(); delete $3; delete $5;
    $$ = $1;
  }
  | SetCondOps Types ValueRef ',' ValueRef {
    *$1 += " " + *$2.newTy + " " + *$3 + ", " + *$5;
    $2.destroy(); delete $3; delete $5;
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
    const char *opcode = $1->c_str();
    if (*$1 == "cast")
      opcode = getCastOpcode($2.type, $4);
    $$ = new std::string(opcode);
    *$$ += *$2.val + " " + *$3 + " " + *$4.newTy; 
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
    *$1 += *$3.newTy + " " + *$4 + "(" + *$6 + ")";
    delete $2; $3.destroy(); delete $4; delete $6;
    $$ = $1;
  }
  | MemoryInst ;


// IndexList - List of indices for GEP based instructions...
IndexList 
  : ',' ValueRefList { 
    $2->insert(0, ", ");
    $$ = $2;
  } 
  | /* empty */ {  $$ = new std::string(); }
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
    *$1 += " " + *$2.newTy + ", " + *$4.newTy + " " + *$5;
    if (!$6->empty())
      *$1 += " " + *$6;
    $2.destroy(); $4.destroy(); delete $5; delete $6;
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
    *$1 += " " + *$2.newTy + ", " + *$4.newTy + " " + *$5;
    if (!$6->empty())
      *$1 += " " + *$6;
    $2.destroy(); $4.destroy(); delete $5; delete $6;
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
    *$1 += *$2 + " " + *$3.newTy + " " + *$4;
    delete $2; $3.destroy(); delete $4;
    $$ = $1;
  }
  | OptVolatile STORE ResolvedVal ',' Types ValueRef {
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$2 + " " + *$3.val + ", " + *$5.newTy + " " + *$6;
    delete $2; $3.destroy(); $5.destroy(); delete $6;
    $$ = $1;
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    *$1 += *$2.newTy + " " + *$3 + " " + *$4;
    $2.destroy(); delete $3; delete $4;
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
