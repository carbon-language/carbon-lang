/*===-- Lexer.l - Scanner for llvm assembly files --------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the flex scanner for LLVM assembly languages files.
//
//===----------------------------------------------------------------------===*/

%option prefix="llvmAsm"
%option yylineno
%option nostdinit
%option never-interactive
%option batch
%option noyywrap
%option nodefault
%option 8bit
%option outfile="Lexer.cpp"
%option ecs
%option noreject
%option noyymore

%{
#include "ParserInternals.h"
#include "llvm/Module.h"
#include <list>
#include "llvmAsmParser.h"
#include <cctype>
#include <cstdlib>

void set_scan_file(FILE * F){
  yy_switch_to_buffer(yy_create_buffer( F, YY_BUF_SIZE ) );
}
void set_scan_string (const char * str) {
  yy_scan_string (str);
}

// Construct a token value for a non-obsolete token
#define RET_TOK(type, Enum, sym) \
  llvmAsmlval.type = Instruction::Enum; \
  return sym

// Construct a token value for an obsolete token
#define RET_TY(CTYPE, SYM) \
  llvmAsmlval.PrimType = CTYPE;\
  return SYM

namespace llvm {

// TODO: All of the static identifiers are figured out by the lexer,
// these should be hashed to reduce the lexer size


// atoull - Convert an ascii string of decimal digits into the unsigned long
// long representation... this does not have to do input error checking,
// because we know that the input will be matched by a suitable regex...
//
static uint64_t atoull(const char *Buffer) {
  uint64_t Result = 0;
  for (; *Buffer; Buffer++) {
    uint64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
    if (Result < OldRes)   // Uh, oh, overflow detected!!!
      GenerateError("constant bigger than 64 bits detected!");
  }
  return Result;
}

static uint64_t HexIntToVal(const char *Buffer) {
  uint64_t Result = 0;
  for (; *Buffer; ++Buffer) {
    uint64_t OldRes = Result;
    Result *= 16;
    char C = *Buffer;
    if (C >= '0' && C <= '9')
      Result += C-'0';
    else if (C >= 'A' && C <= 'F')
      Result += C-'A'+10;
    else if (C >= 'a' && C <= 'f')
      Result += C-'a'+10;

    if (Result < OldRes)   // Uh, oh, overflow detected!!!
      GenerateError("constant bigger than 64 bits detected!");
  }
  return Result;
}


// HexToFP - Convert the ascii string in hexidecimal format to the floating
// point representation of it.
//
static double HexToFP(const char *Buffer) {
  // Behave nicely in the face of C TBAA rules... see:
  // http://www.nullstone.com/htmls/category/aliastyp.htm
  union {
    uint64_t UI;
    double FP;
  } UIntToFP;
  UIntToFP.UI = HexIntToVal(Buffer);

  assert(sizeof(double) == sizeof(uint64_t) &&
         "Data sizes incompatible on this target!");
  return UIntToFP.FP;   // Cast Hex constant to double
}


// UnEscapeLexed - Run through the specified buffer and change \xx codes to the
// appropriate character.  If AllowNull is set to false, a \00 value will cause
// an exception to be thrown.
//
// If AllowNull is set to true, the return value of the function points to the
// last character of the string in memory.
//
char *UnEscapeLexed(char *Buffer, bool AllowNull) {
  char *BOut = Buffer;
  for (char *BIn = Buffer; *BIn; ) {
    if (BIn[0] == '\\' && isxdigit(BIn[1]) && isxdigit(BIn[2])) {
      char Tmp = BIn[3]; BIn[3] = 0;     // Terminate string
      *BOut = (char)strtol(BIn+1, 0, 16);  // Convert to number
      if (!AllowNull && !*BOut)
        GenerateError("String literal cannot accept \\00 escape!");

      BIn[3] = Tmp;                  // Restore character
      BIn += 3;                      // Skip over handled chars
      ++BOut;
    } else {
      *BOut++ = *BIn++;
    }
  }

  return BOut;
}

} // End llvm namespace

using namespace llvm;

#define YY_NEVER_INTERACTIVE 1
%}



/* Comments start with a ; and go till end of line */
Comment    ;.*

/* Local Values and Type identifiers start with a % sign */
LocalVarName       %[-a-zA-Z$._][-a-zA-Z$._0-9]*

/* Global Value identifiers start with an @ sign */
GlobalVarName       @[-a-zA-Z$._][-a-zA-Z$._0-9]*

/* Label identifiers end with a colon */
Label       [-a-zA-Z$._0-9]+:
QuoteLabel \"[^\"]+\":

/* Quoted names can contain any character except " and \ */
StringConstant \"[^\"]*\"
AtStringConstant @\"[^\"]*\"
  
/* LocalVarID/GlobalVarID: match an unnamed local variable slot ID. */
LocalVarID     %[0-9]+
GlobalVarID    @[0-9]+

/* Integer types are specified with i and a bitwidth */
IntegerType i[0-9]+

/* E[PN]Integer: match positive and negative literal integer values. */
PInteger   [0-9]+
NInteger  -[0-9]+

/* FPConstant - A Floating point constant.
 */
FPConstant [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?

/* HexFPConstant - Floating point constant represented in IEEE format as a
 *  hexadecimal number for when exponential notation is not precise enough.
 */
HexFPConstant 0x[0-9A-Fa-f]+

/* HexIntConstant - Hexadecimal constant generated by the CFE to avoid forcing
 * it to deal with 64 bit numbers.
 */
HexIntConstant [us]0x[0-9A-Fa-f]+

%%

{Comment}       { /* Ignore comments for now */ }

begin           { return BEGINTOK; }
end             { return ENDTOK; }
true            { return TRUETOK;  }
false           { return FALSETOK; }
declare         { return DECLARE; }
define          { return DEFINE; }
global          { return GLOBAL; }
constant        { return CONSTANT; }
internal        { return INTERNAL; }
linkonce        { return LINKONCE; }
weak            { return WEAK; }
appending       { return APPENDING; }
dllimport       { return DLLIMPORT; }
dllexport       { return DLLEXPORT; }
hidden          { return HIDDEN; }
protected       { return PROTECTED; }
extern_weak     { return EXTERN_WEAK; }
external        { return EXTERNAL; }
thread_local    { return THREAD_LOCAL; }
zeroinitializer { return ZEROINITIALIZER; }
\.\.\.          { return DOTDOTDOT; }
undef           { return UNDEF; }
null            { return NULL_TOK; }
to              { return TO; }
tail            { return TAIL; }
target          { return TARGET; }
triple          { return TRIPLE; }
deplibs         { return DEPLIBS; }
datalayout      { return DATALAYOUT; }
volatile        { return VOLATILE; }
align           { return ALIGN;  }
section         { return SECTION; }
alias           { return ALIAS; }
module          { return MODULE; }
asm             { return ASM_TOK; }
sideeffect      { return SIDEEFFECT; }

cc              { return CC_TOK; }
ccc             { return CCC_TOK; }
fastcc          { return FASTCC_TOK; }
coldcc          { return COLDCC_TOK; }
x86_stdcallcc   { return X86_STDCALLCC_TOK; }
x86_fastcallcc  { return X86_FASTCALLCC_TOK; }

inreg           { return INREG; }
sret            { return SRET;  }
nounwind        { return NOUNWIND; }
noreturn        { return NORETURN; }

void            { RET_TY(Type::VoidTy,  VOID);  }
float           { RET_TY(Type::FloatTy, FLOAT); }
double          { RET_TY(Type::DoubleTy,DOUBLE);}
label           { RET_TY(Type::LabelTy, LABEL); }
type            { return TYPE;   }
opaque          { return OPAQUE; }
{IntegerType}   { uint64_t NumBits = atoull(yytext+1);
                  if (NumBits < IntegerType::MIN_INT_BITS || 
                      NumBits > IntegerType::MAX_INT_BITS)
                    GenerateError("Bitwidth for integer type out of range!");
                  const Type* Ty = IntegerType::get(NumBits);
                  RET_TY(Ty, INTTYPE);
                }

add             { RET_TOK(BinaryOpVal, Add, ADD); }
sub             { RET_TOK(BinaryOpVal, Sub, SUB); }
mul             { RET_TOK(BinaryOpVal, Mul, MUL); }
udiv            { RET_TOK(BinaryOpVal, UDiv, UDIV); }
sdiv            { RET_TOK(BinaryOpVal, SDiv, SDIV); }
fdiv            { RET_TOK(BinaryOpVal, FDiv, FDIV); }
urem            { RET_TOK(BinaryOpVal, URem, UREM); }
srem            { RET_TOK(BinaryOpVal, SRem, SREM); }
frem            { RET_TOK(BinaryOpVal, FRem, FREM); }
shl             { RET_TOK(BinaryOpVal, Shl, SHL); }
lshr            { RET_TOK(BinaryOpVal, LShr, LSHR); }
ashr            { RET_TOK(BinaryOpVal, AShr, ASHR); }
and             { RET_TOK(BinaryOpVal, And, AND); }
or              { RET_TOK(BinaryOpVal, Or , OR ); }
xor             { RET_TOK(BinaryOpVal, Xor, XOR); }
icmp            { RET_TOK(OtherOpVal,  ICmp,  ICMP); }
fcmp            { RET_TOK(OtherOpVal,  FCmp,  FCMP); }

eq              { return EQ;  }
ne              { return NE;  }
slt             { return SLT; }
sgt             { return SGT; }
sle             { return SLE; }
sge             { return SGE; }
ult             { return ULT; }
ugt             { return UGT; }
ule             { return ULE; }
uge             { return UGE; }
oeq             { return OEQ; }
one             { return ONE; }
olt             { return OLT; }
ogt             { return OGT; }
ole             { return OLE; }
oge             { return OGE; }
ord             { return ORD; }
uno             { return UNO; }
ueq             { return UEQ; }
une             { return UNE; }

phi             { RET_TOK(OtherOpVal, PHI, PHI_TOK); }
call            { RET_TOK(OtherOpVal, Call, CALL); }
trunc           { RET_TOK(CastOpVal, Trunc, TRUNC); }
zext            { RET_TOK(CastOpVal, ZExt, ZEXT); }
sext            { RET_TOK(CastOpVal, SExt, SEXT); }
fptrunc         { RET_TOK(CastOpVal, FPTrunc, FPTRUNC); }
fpext           { RET_TOK(CastOpVal, FPExt, FPEXT); }
uitofp          { RET_TOK(CastOpVal, UIToFP, UITOFP); }
sitofp          { RET_TOK(CastOpVal, SIToFP, SITOFP); }
fptoui          { RET_TOK(CastOpVal, FPToUI, FPTOUI); }
fptosi          { RET_TOK(CastOpVal, FPToSI, FPTOSI); }
inttoptr        { RET_TOK(CastOpVal, IntToPtr, INTTOPTR); }
ptrtoint        { RET_TOK(CastOpVal, PtrToInt, PTRTOINT); }
bitcast         { RET_TOK(CastOpVal, BitCast, BITCAST); }
select          { RET_TOK(OtherOpVal, Select, SELECT); }
va_arg          { RET_TOK(OtherOpVal, VAArg , VAARG); }
ret             { RET_TOK(TermOpVal, Ret, RET); }
br              { RET_TOK(TermOpVal, Br, BR); }
switch          { RET_TOK(TermOpVal, Switch, SWITCH); }
invoke          { RET_TOK(TermOpVal, Invoke, INVOKE); }
unwind          { RET_TOK(TermOpVal, Unwind, UNWIND); }
unreachable     { RET_TOK(TermOpVal, Unreachable, UNREACHABLE); }

malloc          { RET_TOK(MemOpVal, Malloc, MALLOC); }
alloca          { RET_TOK(MemOpVal, Alloca, ALLOCA); }
free            { RET_TOK(MemOpVal, Free, FREE); }
load            { RET_TOK(MemOpVal, Load, LOAD); }
store           { RET_TOK(MemOpVal, Store, STORE); }
getelementptr   { RET_TOK(MemOpVal, GetElementPtr, GETELEMENTPTR); }

extractelement  { RET_TOK(OtherOpVal, ExtractElement, EXTRACTELEMENT); }
insertelement   { RET_TOK(OtherOpVal, InsertElement, INSERTELEMENT); }
shufflevector   { RET_TOK(OtherOpVal, ShuffleVector, SHUFFLEVECTOR); }


{LocalVarName}  {
                  UnEscapeLexed(yytext+1);
                  llvmAsmlval.StrVal = strdup(yytext+1);             // Skip %
                  return LOCALVAR;
                }
{GlobalVarName} {
                  UnEscapeLexed(yytext+1);
                  llvmAsmlval.StrVal = strdup(yytext+1);             // Skip @
                  return GLOBALVAR;
                }
{Label}         {
                  yytext[strlen(yytext)-1] = 0;  // nuke colon
                  UnEscapeLexed(yytext);
                  llvmAsmlval.StrVal = strdup(yytext);
                  return LABELSTR;
                }
{QuoteLabel}    {
                  yytext[strlen(yytext)-2] = 0;  // nuke colon, end quote
                  UnEscapeLexed(yytext+1);
                  llvmAsmlval.StrVal = strdup(yytext+1);
                  return LABELSTR;
                }

{StringConstant} { // Note that we cannot unescape a string constant here!  The
                   // string constant might contain a \00 which would not be
                   // understood by the string stuff.  It is valid to make a
                   // [sbyte] c"Hello World\00" constant, for example.
                   //
                   yytext[strlen(yytext)-1] = 0;           // nuke end quote
                   llvmAsmlval.StrVal = strdup(yytext+1);  // Nuke start quote
                   return STRINGCONSTANT;
                 }
{AtStringConstant} {
                     yytext[strlen(yytext)-1] = 0;           // nuke end quote
                     llvmAsmlval.StrVal = strdup(yytext+2);  // Nuke @, quote
                     return ATSTRINGCONSTANT;
                   }

{PInteger}      { int len = strlen(yytext); 
                  uint32_t numBits = ((len * 64) / 19) + 1;
                  APInt Tmp(numBits, yytext, len, 10);
                  uint32_t activeBits = Tmp.getActiveBits();
                  if (activeBits > 0 && activeBits < numBits)
                    Tmp.trunc(activeBits);
                  if (Tmp.getBitWidth() > 64) {
                    llvmAsmlval.APIntVal = new APInt(Tmp);
                    return EUAPINTVAL; 
                  } else {
                    llvmAsmlval.UInt64Val = Tmp.getZExtValue();
                    return EUINT64VAL;
                  }
                }
{NInteger}      { int len = strlen(yytext); 
                  uint32_t numBits = (((len-1) * 64) / 19) + 2;
                  APInt Tmp(numBits, yytext, len, 10);
                  uint32_t minBits = Tmp.getMinSignedBits();
                  if (minBits > 0 && minBits < numBits)
                    Tmp.trunc(minBits);
                  if (Tmp.getBitWidth() > 64) {
                    llvmAsmlval.APIntVal = new APInt(Tmp);
                    return ESAPINTVAL;
                  } else {
                    llvmAsmlval.SInt64Val = Tmp.getSExtValue();
                    return ESINT64VAL;
                  }
                }

{HexIntConstant} { int len = strlen(yytext+3) - 3;
                   uint32_t bits = len * 4;
                   APInt Tmp(bits, yytext+3, len, 16);
                   uint32_t activeBits = Tmp.getActiveBits();
                   if (activeBits > 0 && activeBits < bits)
                     Tmp.trunc(activeBits);
                   if (Tmp.getBitWidth() > 64) {
                     llvmAsmlval.APIntVal = new APInt(Tmp);
                     return yytext[0] == 's' ? ESAPINTVAL : EUAPINTVAL;
                   } else if (yytext[0] == 's') {
                     llvmAsmlval.SInt64Val = Tmp.getSExtValue();
                     return ESINT64VAL;
                   } else {
                     llvmAsmlval.UInt64Val = Tmp.getZExtValue();
                     return EUINT64VAL;
                   }
                 }

{LocalVarID}     {
                  uint64_t Val = atoull(yytext+1);
                  if ((unsigned)Val != Val)
                    GenerateError("Invalid value number (too large)!");
                  llvmAsmlval.UIntVal = unsigned(Val);
                  return LOCALVAL_ID;
                }
{GlobalVarID}   {
                  uint64_t Val = atoull(yytext+1);
                  if ((unsigned)Val != Val)
                    GenerateError("Invalid value number (too large)!");
                  llvmAsmlval.UIntVal = unsigned(Val);
                  return GLOBALVAL_ID;
                }

{FPConstant}    { llvmAsmlval.FPVal = atof(yytext); return FPVAL; }
{HexFPConstant} { llvmAsmlval.FPVal = HexToFP(yytext); return FPVAL; }

<<EOF>>         {
                  /* Make sure to free the internal buffers for flex when we are
                   * done reading our input!
                   */
                  yy_delete_buffer(YY_CURRENT_BUFFER);
                  return EOF;
                }

[ \r\t\n]       { /* Ignore whitespace */ }
.               { return yytext[0]; }

%%
