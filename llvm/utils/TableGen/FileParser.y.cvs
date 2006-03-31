//===-- FileParser.y - Parser for TableGen files ----------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file implements the bison parser for Table Generator files...
//
//===----------------------------------------------------------------------===//

%{
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cstdio>
#define YYERROR_VERBOSE 1

int yyerror(const char *ErrorMsg);
int yylex();

namespace llvm {

extern int Filelineno;
static Record *CurRec = 0;
static bool ParsingTemplateArgs = false;

typedef std::pair<Record*, std::vector<Init*>*> SubClassRefTy;

struct LetRecord {
  std::string Name;
  std::vector<unsigned> Bits;
  Init *Value;
  bool HasBits;
  LetRecord(const std::string &N, std::vector<unsigned> *B, Init *V)
    : Name(N), Value(V), HasBits(B != 0) {
    if (HasBits) Bits = *B;
  }
};

static std::vector<std::vector<LetRecord> > LetStack;


extern std::ostream &err();

static void addValue(const RecordVal &RV) {
  if (RecordVal *ERV = CurRec->getValue(RV.getName())) {
    // The value already exists in the class, treat this as a set...
    if (ERV->setValue(RV.getValue())) {
      err() << "New definition of '" << RV.getName() << "' of type '"
            << *RV.getType() << "' is incompatible with previous "
            << "definition of type '" << *ERV->getType() << "'!\n";
      exit(1);
    }
  } else {
    CurRec->addValue(RV);
  }
}

static void addSuperClass(Record *SC) {
  if (CurRec->isSubClassOf(SC)) {
    err() << "Already subclass of '" << SC->getName() << "'!\n";
    exit(1);
  }
  CurRec->addSuperClass(SC);
}

static void setValue(const std::string &ValName, 
                     std::vector<unsigned> *BitList, Init *V) {
  if (!V) return;

  RecordVal *RV = CurRec->getValue(ValName);
  if (RV == 0) {
    err() << "Value '" << ValName << "' unknown!\n";
    exit(1);
  }

  // Do not allow assignments like 'X = X'.  This will just cause infinite loops
  // in the resolution machinery.
  if (!BitList)
    if (VarInit *VI = dynamic_cast<VarInit*>(V))
      if (VI->getName() == ValName)
        return;
  
  // If we are assigning to a subset of the bits in the value... then we must be
  // assigning to a field of BitsRecTy, which must have a BitsInit
  // initializer...
  //
  if (BitList) {
    BitsInit *CurVal = dynamic_cast<BitsInit*>(RV->getValue());
    if (CurVal == 0) {
      err() << "Value '" << ValName << "' is not a bits type!\n";
      exit(1);
    }

    // Convert the incoming value to a bits type of the appropriate size...
    Init *BI = V->convertInitializerTo(new BitsRecTy(BitList->size()));
    if (BI == 0) {
      V->convertInitializerTo(new BitsRecTy(BitList->size()));
      err() << "Initializer '" << *V << "' not compatible with bit range!\n";
      exit(1);
    }

    // We should have a BitsInit type now...
    assert(dynamic_cast<BitsInit*>(BI) != 0 || &(std::cerr << *BI) == 0);
    BitsInit *BInit = (BitsInit*)BI;

    BitsInit *NewVal = new BitsInit(CurVal->getNumBits());

    // Loop over bits, assigning values as appropriate...
    for (unsigned i = 0, e = BitList->size(); i != e; ++i) {
      unsigned Bit = (*BitList)[i];
      if (NewVal->getBit(Bit)) {
        err() << "Cannot set bit #" << Bit << " of value '" << ValName
              << "' more than once!\n";
        exit(1);
      }
      NewVal->setBit(Bit, BInit->getBit(i));
    }

    for (unsigned i = 0, e = CurVal->getNumBits(); i != e; ++i)
      if (NewVal->getBit(i) == 0)
        NewVal->setBit(i, CurVal->getBit(i));

    V = NewVal;
  }

  if (RV->setValue(V)) {
    err() << "Value '" << ValName << "' of type '" << *RV->getType()
	  << "' is incompatible with initializer '" << *V << "'!\n";
    exit(1);
  }
}

// addSubClass - Add SC as a subclass to CurRec, resolving TemplateArgs as SC's
// template arguments.
static void addSubClass(Record *SC, const std::vector<Init*> &TemplateArgs) {
  // Add all of the values in the subclass into the current class...
  const std::vector<RecordVal> &Vals = SC->getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    addValue(Vals[i]);

  const std::vector<std::string> &TArgs = SC->getTemplateArgs();

  // Ensure that an appropriate number of template arguments are specified...
  if (TArgs.size() < TemplateArgs.size()) {
    err() << "ERROR: More template args specified than expected!\n";
    exit(1);
  } else {    // This class expects template arguments...
    // Loop over all of the template arguments, setting them to the specified
    // value or leaving them as the default if necessary.
    for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
      if (i < TemplateArgs.size()) {  // A value is specified for this temp-arg?
        // Set it now.
        setValue(TArgs[i], 0, TemplateArgs[i]);

        // Resolve it next.
        CurRec->resolveReferencesTo(CurRec->getValue(TArgs[i]));
                                    
        
        // Now remove it.
        CurRec->removeValue(TArgs[i]);

      } else if (!CurRec->getValue(TArgs[i])->getValue()->isComplete()) {
        err() << "ERROR: Value not specified for template argument #"
              << i << " (" << TArgs[i] << ") of subclass '" << SC->getName()
              << "'!\n";
        exit(1);
      }
    }
  }

  // Since everything went well, we can now set the "superclass" list for the
  // current record.
  const std::vector<Record*> &SCs  = SC->getSuperClasses();
  for (unsigned i = 0, e = SCs.size(); i != e; ++i)
    addSuperClass(SCs[i]);
  addSuperClass(SC);
}

} // End llvm namespace

using namespace llvm;

%}

%union {
  std::string*                StrVal;
  int                         IntVal;
  llvm::RecTy*                Ty;
  llvm::Init*                 Initializer;
  std::vector<llvm::Init*>*   FieldList;
  std::vector<unsigned>*      BitList;
  llvm::Record*               Rec;
  SubClassRefTy*              SubClassRef;
  std::vector<SubClassRefTy>* SubClassList;
  std::vector<std::pair<llvm::Init*, std::string> >* DagValueList;
};

%token INT BIT STRING BITS LIST CODE DAG CLASS DEF FIELD LET IN
%token SHLTOK SRATOK SRLTOK STRCONCATTOK
%token <IntVal>      INTVAL
%token <StrVal>      ID VARNAME STRVAL CODEFRAGMENT

%type <Ty>           Type
%type <Rec>          ClassInst DefInst Object ObjectBody ClassID

%type <SubClassRef>  SubClassRef
%type <SubClassList> ClassList ClassListNE
%type <IntVal>       OptPrefix
%type <Initializer>  Value OptValue IDValue
%type <DagValueList> DagArgList DagArgListNE
%type <FieldList>    ValueList ValueListNE
%type <BitList>      BitList OptBitList RBitList
%type <StrVal>       Declaration OptID OptVarName ObjectName

%start File

%%

ClassID : ID {
    $$ = Records.getClass(*$1);
    if ($$ == 0) {
      err() << "Couldn't find class '" << *$1 << "'!\n";
      exit(1);
    }
    delete $1;
  };


// TableGen types...
Type : STRING {                       // string type
    $$ = new StringRecTy();
  } | BIT {                           // bit type
    $$ = new BitRecTy();
  } | BITS '<' INTVAL '>' {           // bits<x> type
    $$ = new BitsRecTy($3);
  } | INT {                           // int type
    $$ = new IntRecTy();
  } | LIST '<' Type '>'    {          // list<x> type
    $$ = new ListRecTy($3);
  } | CODE {                          // code type
    $$ = new CodeRecTy();
  } | DAG {                           // dag type
    $$ = new DagRecTy();
  } | ClassID {                       // Record Type
    $$ = new RecordRecTy($1);
  };

OptPrefix : /*empty*/ { $$ = 0; } | FIELD { $$ = 1; };

OptValue : /*empty*/ { $$ = 0; } | '=' Value { $$ = $2; };

IDValue : ID {
  if (const RecordVal *RV = (CurRec ? CurRec->getValue(*$1) : 0)) {
    $$ = new VarInit(*$1, RV->getType());
  } else if (CurRec && CurRec->isTemplateArg(CurRec->getName()+":"+*$1)) {
    const RecordVal *RV = CurRec->getValue(CurRec->getName()+":"+*$1);
    assert(RV && "Template arg doesn't exist??");
    $$ = new VarInit(CurRec->getName()+":"+*$1, RV->getType());
  } else if (Record *D = Records.getDef(*$1)) {
    $$ = new DefInit(D);
  } else {
    err() << "Variable not defined: '" << *$1 << "'!\n";
    exit(1);
  }
  
  delete $1;
};

Value : IDValue {
    $$ = $1;
  } | INTVAL {
    $$ = new IntInit($1);
  } | STRVAL {
    $$ = new StringInit(*$1);
    delete $1;
  } | CODEFRAGMENT {
    $$ = new CodeInit(*$1);
    delete $1;
  } | '?' {
    $$ = new UnsetInit();
  } | '{' ValueList '}' {
    BitsInit *Init = new BitsInit($2->size());
    for (unsigned i = 0, e = $2->size(); i != e; ++i) {
      struct Init *Bit = (*$2)[i]->convertInitializerTo(new BitRecTy());
      if (Bit == 0) {
        err() << "Element #" << i << " (" << *(*$2)[i]
       	      << ") is not convertable to a bit!\n";
        exit(1);
      }
      Init->setBit($2->size()-i-1, Bit);
    }
    $$ = Init;
    delete $2;
  } | ID '<' ValueListNE '>' {
    // This is a CLASS<initvalslist> expression.  This is supposed to synthesize
    // a new anonymous definition, deriving from CLASS<initvalslist> with no
    // body.
    Record *Class = Records.getClass(*$1);
    if (!Class) {
      err() << "Expected a class, got '" << *$1 << "'!\n";
      exit(1);
    }
    delete $1;
    
    static unsigned AnonCounter = 0;
    Record *OldRec = CurRec;  // Save CurRec.
    
    // Create the new record, set it as CurRec temporarily.
    CurRec = new Record("anonymous.val."+utostr(AnonCounter++));
    addSubClass(Class, *$3);    // Add info about the subclass to CurRec.
    delete $3;  // Free up the template args.
    
    CurRec->resolveReferences();
    
    Records.addDef(CurRec);
    
    // The result of the expression is a reference to the new record.
    $$ = new DefInit(CurRec);
    
    // Restore the old CurRec
    CurRec = OldRec;
  } | Value '{' BitList '}' {
    $$ = $1->convertInitializerBitRange(*$3);
    if ($$ == 0) {
      err() << "Invalid bit range for value '" << *$1 << "'!\n";
      exit(1);
    }
    delete $3;
  } | '[' ValueList ']' {
    $$ = new ListInit(*$2);
    delete $2;
  } | Value '.' ID {
    if (!$1->getFieldType(*$3)) {
      err() << "Cannot access field '" << *$3 << "' of value '" << *$1 << "!\n";
      exit(1);
    }
    $$ = new FieldInit($1, *$3);
    delete $3;
  } | '(' IDValue DagArgList ')' {
    $$ = new DagInit($2, *$3);
    delete $3;
  } | Value '[' BitList ']' {
    std::reverse($3->begin(), $3->end());
    $$ = $1->convertInitListSlice(*$3);
    if ($$ == 0) {
      err() << "Invalid list slice for value '" << *$1 << "'!\n";
      exit(1);
    }
    delete $3;
  } | SHLTOK '(' Value ',' Value ')' {
    $$ = (new BinOpInit(BinOpInit::SHL, $3, $5))->Fold();
  } | SRATOK '(' Value ',' Value ')' {
    $$ = (new BinOpInit(BinOpInit::SRA, $3, $5))->Fold();
  } | SRLTOK '(' Value ',' Value ')' {
    $$ = (new BinOpInit(BinOpInit::SRL, $3, $5))->Fold();
  } | STRCONCATTOK '(' Value ',' Value ')' {
    $$ = (new BinOpInit(BinOpInit::STRCONCAT, $3, $5))->Fold();
  };

OptVarName : /* empty */ {
    $$ = new std::string();
  }
  | ':' VARNAME {
    $$ = $2;
  };

DagArgListNE : Value OptVarName {
    $$ = new std::vector<std::pair<Init*, std::string> >();
    $$->push_back(std::make_pair($1, *$2));
    delete $2;
  }
  | DagArgListNE ',' Value OptVarName {
    $1->push_back(std::make_pair($3, *$4));
    delete $4;
    $$ = $1;
  };

DagArgList : /*empty*/ {
    $$ = new std::vector<std::pair<Init*, std::string> >();
  }
  | DagArgListNE { $$ = $1; };


RBitList : INTVAL {
    $$ = new std::vector<unsigned>();
    $$->push_back($1);
  } | INTVAL '-' INTVAL {
    if ($1 < 0 || $3 < 0) {
      err() << "Invalid range: " << $1 << "-" << $3 << "!\n";
      exit(1);
    }
    $$ = new std::vector<unsigned>();
    if ($1 < $3) {
      for (int i = $1; i <= $3; ++i)
        $$->push_back(i);
    } else {
      for (int i = $1; i >= $3; --i)
        $$->push_back(i);
    }
  } | INTVAL INTVAL {
    $2 = -$2;
    if ($1 < 0 || $2 < 0) {
      err() << "Invalid range: " << $1 << "-" << $2 << "!\n";
      exit(1);
    }
    $$ = new std::vector<unsigned>();
    if ($1 < $2) {
      for (int i = $1; i <= $2; ++i)
        $$->push_back(i);
    } else {
      for (int i = $1; i >= $2; --i)
        $$->push_back(i);
    }
  } | RBitList ',' INTVAL {
    ($$=$1)->push_back($3);
  } | RBitList ',' INTVAL '-' INTVAL {
    if ($3 < 0 || $5 < 0) {
      err() << "Invalid range: " << $3 << "-" << $5 << "!\n";
      exit(1);
    }
    $$ = $1;
    if ($3 < $5) {
      for (int i = $3; i <= $5; ++i)
        $$->push_back(i);
    } else {
      for (int i = $3; i >= $5; --i)
        $$->push_back(i);
    }
  } | RBitList ',' INTVAL INTVAL {
    $4 = -$4;
    if ($3 < 0 || $4 < 0) {
      err() << "Invalid range: " << $3 << "-" << $4 << "!\n";
      exit(1);
    }
    $$ = $1;
    if ($3 < $4) {
      for (int i = $3; i <= $4; ++i)
        $$->push_back(i);
    } else {
      for (int i = $3; i >= $4; --i)
        $$->push_back(i);
    }
  };

BitList : RBitList { $$ = $1; std::reverse($1->begin(), $1->end()); };

OptBitList : /*empty*/ { $$ = 0; } | '{' BitList '}' { $$ = $2; };



ValueList : /*empty*/ {
    $$ = new std::vector<Init*>();
  } | ValueListNE {
    $$ = $1;
  };

ValueListNE : Value {
    $$ = new std::vector<Init*>();
    $$->push_back($1);
  } | ValueListNE ',' Value {
    ($$ = $1)->push_back($3);
  };

Declaration : OptPrefix Type ID OptValue {
  std::string DecName = *$3;
  if (ParsingTemplateArgs)
    DecName = CurRec->getName() + ":" + DecName;

  addValue(RecordVal(DecName, $2, $1));
  setValue(DecName, 0, $4);
  $$ = new std::string(DecName);
};

BodyItem : Declaration ';' {
  delete $1;
} | LET ID OptBitList '=' Value ';' {
  setValue(*$2, $3, $5);
  delete $2;
  delete $3;
};

BodyList : /*empty*/ | BodyList BodyItem;
Body : ';' | '{' BodyList '}';

SubClassRef : ClassID {
    $$ = new SubClassRefTy($1, new std::vector<Init*>());
  } | ClassID '<' ValueListNE '>' {
    $$ = new SubClassRefTy($1, $3);
  };

ClassListNE : SubClassRef {
    $$ = new std::vector<SubClassRefTy>();
    $$->push_back(*$1);
    delete $1;
  }
  | ClassListNE ',' SubClassRef {
    ($$=$1)->push_back(*$3);
    delete $3;
  };

ClassList : /*empty */ {
    $$ = new std::vector<SubClassRefTy>();
  }
  | ':' ClassListNE {
    $$ = $2;
  };

DeclListNE : Declaration {
  CurRec->addTemplateArg(*$1);
  delete $1;
} | DeclListNE ',' Declaration {
  CurRec->addTemplateArg(*$3);
  delete $3;
};

TemplateArgList : '<' DeclListNE '>' {};
OptTemplateArgList : /*empty*/ | TemplateArgList;

OptID : ID { $$ = $1; } | /*empty*/ { $$ = new std::string(); };

ObjectName : OptID {
  static unsigned AnonCounter = 0;
  if ($1->empty())
    *$1 = "anonymous."+utostr(AnonCounter++);
  $$ = $1;
};

ClassName : ObjectName {
  // If a class of this name already exists, it must be a forward ref.
  if ((CurRec = Records.getClass(*$1))) {
    // If the body was previously defined, this is an error.
    if (!CurRec->getValues().empty() ||
        !CurRec->getSuperClasses().empty() ||
        !CurRec->getTemplateArgs().empty()) {
      err() << "Class '" << CurRec->getName() << "' already defined!\n";
      exit(1);
    }
  } else {
    // If this is the first reference to this class, create and add it.
    CurRec = new Record(*$1);
    Records.addClass(CurRec);
  }
  delete $1;
};

DefName : ObjectName {
  CurRec = new Record(*$1);
  delete $1;
  
  // Ensure redefinition doesn't happen.
  if (Records.getDef(CurRec->getName())) {
    err() << "Def '" << CurRec->getName() << "' already defined!\n";
    exit(1);
  }
  Records.addDef(CurRec);
};

ObjectBody : ClassList {
           for (unsigned i = 0, e = $1->size(); i != e; ++i) {
             addSubClass((*$1)[i].first, *(*$1)[i].second);
             // Delete the template arg values for the class
             delete (*$1)[i].second;
           }
           delete $1;   // Delete the class list...
  
           // Process any variables on the set stack...
           for (unsigned i = 0, e = LetStack.size(); i != e; ++i)
             for (unsigned j = 0, e = LetStack[i].size(); j != e; ++j)
               setValue(LetStack[i][j].Name,
                        LetStack[i][j].HasBits ? &LetStack[i][j].Bits : 0,
                        LetStack[i][j].Value);
         } Body {
           $$ = CurRec;
           CurRec = 0;
         };

ClassInst : CLASS ClassName {
                ParsingTemplateArgs = true;
            } OptTemplateArgList {
                ParsingTemplateArgs = false;
            } ObjectBody {
        $$ = $6;
     };

DefInst : DEF DefName ObjectBody {
  $3->resolveReferences();

  // If ObjectBody has template arguments, it's an error.
  assert($3->getTemplateArgs().empty() && "How'd this get template args?");
  $$ = $3;
};


Object : ClassInst | DefInst;

LETItem : ID OptBitList '=' Value {
  LetStack.back().push_back(LetRecord(*$1, $2, $4));
  delete $1; delete $2;
};

LETList : LETItem | LETList ',' LETItem;

// LETCommand - A 'LET' statement start...
LETCommand : LET { LetStack.push_back(std::vector<LetRecord>()); } LETList IN;

// Support Set commands wrapping objects... both with and without braces.
Object : LETCommand '{' ObjectList '}' {
    LetStack.pop_back();
  }
  | LETCommand Object {
    LetStack.pop_back();
  };

ObjectList : Object {} | ObjectList Object {};

File : ObjectList {};

%%

int yyerror(const char *ErrorMsg) {
  err() << "Error parsing: " << ErrorMsg << "\n";
  exit(1);
}
