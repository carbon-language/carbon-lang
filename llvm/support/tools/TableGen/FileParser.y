//===-- FileParser.y - Parser for TableGen files ----------------*- C++ -*-===//
//
//  This file implements the bison parser for Table Generator files...
//
//===------------------------------------------------------------------------=//

%{
#include "Record.h"
#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#define YYERROR_VERBOSE 1

int yyerror(const char *ErrorMsg);
int yylex();
extern FILE *Filein;
extern int Filelineno;
int Fileparse();
static Record *CurRec = 0;

typedef std::pair<Record*, std::vector<Init*>*> SubClassRefTy;

static std::vector<std::pair<std::pair<std::string, std::vector<unsigned>*>,
                             Init*> > SetStack;

void ParseFile() {
  FILE *F = stdin;

  Filein = F;
  Filelineno = 1;
  Fileparse();
  Filein = stdin;
}

static std::ostream &err() {
  return std::cerr << "Parsing Line #" << Filelineno << ": ";
}

static void addValue(const RecordVal &RV) {
  if (CurRec->getValue(RV.getName())) {
    err() << "Value '" << RV.getName() << "' multiply defined!\n";
    abort();
  }

  CurRec->addValue(RV);
}

static void addSuperClass(Record *SC) {
  if (CurRec->isSubClassOf(SC)) {
    err() << "Already subclass of '" << SC->getName() << "'!\n";
    abort();
  }
  CurRec->addSuperClass(SC);
}

static void setValue(const std::string &ValName, 
		     std::vector<unsigned> *BitList, Init *V) {
  if (!V) return ;

  RecordVal *RV = CurRec->getValue(ValName);
  if (RV == 0) {
    err() << "Value '" << ValName << "' unknown!\n";
    abort();
  }
  
  // If we are assigning to a subset of the bits in the value... then we must be
  // assigning to a field of BitsRecTy, which must have a BitsInit
  // initializer...
  //
  if (BitList) {
    BitsInit *CurVal = dynamic_cast<BitsInit*>(RV->getValue());
    if (CurVal == 0) {
      err() << "Value '" << ValName << "' is not a bits type!\n";
      abort();
    }

    // Convert the incoming value to a bits type of the appropriate size...
    Init *BI = V->convertInitializerTo(new BitsRecTy(BitList->size()));
    if (BI == 0) {
      V->convertInitializerTo(new BitsRecTy(BitList->size()));
      err() << "Initializer '" << *V << "' not compatible with bit range!\n";
      abort();
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
        abort();
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
    abort();
  }
}

static void addSubClass(Record *SC, const std::vector<Init*> &TemplateArgs) {
  // Add all of the values in the subclass into the current class...
  const std::vector<RecordVal> &Vals = SC->getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    addValue(Vals[i]);

  const std::vector<std::string> &TArgs = SC->getTemplateArgs();

  // Ensure that an appropriate number of template arguments are specified...
  if (TArgs.size() < TemplateArgs.size()) {
    err() << "ERROR: More template args specified than expected!\n";
    abort();
  } else {    // This class expects template arguments...
    // Loop over all of the template arguments, setting them to the specified
    // value or leaving them as the default as neccesary.
    for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
      if (i < TemplateArgs.size()) {  // A value is specified for this temp-arg?
	// Set it now.
	setValue(TArgs[i], 0, TemplateArgs[i]);
      } else if (!CurRec->getValue(TArgs[i])->getValue()->isComplete()) {
	err() << "ERROR: Value not specified for template argument #"
	      << i << " (" << TArgs[i] << ") of subclass '" << SC->getName()
	      << "'!\n";
	abort();
      }
    }
  }


  // Since everything went well, we can now set the "superclass" list for the
  // current record.
  const std::vector<Record*>   &SCs  = SC->getSuperClasses();
  for (unsigned i = 0, e = SCs.size(); i != e; ++i)
    addSuperClass(SCs[i]);
  addSuperClass(SC);
}


%}

%union {
  std::string          *StrVal;
  int                   IntVal;
  RecTy                *Ty;
  Init                 *Initializer;
  std::vector<Init*>   *FieldList;
  std::vector<Record*> *RecPtr;
  std::vector<unsigned>*BitList;
  Record               *Rec;
  SubClassRefTy        *SubClassRef;
  std::vector<SubClassRefTy> *SubClassList;
};

%token INT BIT STRING BITS LIST CLASS DEF FIELD SET IN
%token <IntVal>      INTVAL
%token <StrVal>      ID STRVAL

%type <Ty>           Type
%type <RecPtr>       DefList DefListNE
%type <Rec>          ClassInst DefInst Object ObjectBody ClassID DefID

%type <SubClassRef>  SubClassRef
%type <SubClassList> ClassList ClassListNE
%type <IntVal>       OptPrefix
%type <Initializer>  Value OptValue
%type <FieldList>    ValueList ValueListNE
%type <BitList>      BitList OptBitList RBitList
%type <StrVal>       Declaration

%start File
%%

ClassID : ID {
    $$ = Records.getClass(*$1);
    if ($$ == 0) {
      err() << "Couldn't find class '" << *$1 << "'!\n";
      abort();
    }
    delete $1;
  };

DefID : ID {
    $$ = Records.getDef(*$1);
    if ($$ == 0) {
      err() << "Couldn't find def '" << *$1 << "'!\n";
      abort();
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
  } | LIST '<' ClassID '>' {          // list<x> type
    $$ = new ListRecTy($3);
  } | ClassID {                       // Record Type
    $$ = new RecordRecTy($1);
  };

OptPrefix : /*empty*/ { $$ = 0; } | FIELD { $$ = 1; };

OptValue : /*empty*/ { $$ = 0; } | '=' Value { $$ = $2; };

Value : INTVAL {
    $$ = new IntInit($1);
  } | STRVAL {
    $$ = new StringInit(*$1);
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
	abort();
      }
      Init->setBit($2->size()-i-1, Bit);
    }
    $$ = Init;
    delete $2;
  } | ID {
    if (const RecordVal *RV = CurRec->getValue(*$1)) {
      $$ = new VarInit(*$1, RV->getType());
    } else if (Record *D = Records.getDef(*$1)) {
      $$ = new DefInit(D);
    } else {
      err() << "Variable not defined: '" << *$1 << "'!\n";
      abort();
    }
    
    delete $1;
  } | Value '{' BitList '}' {
    $$ = $1->convertInitializerBitRange(*$3);
    if ($$ == 0) {
      err() << "Invalid bit range for value '" << *$1 << "'!\n";
      abort();
    }
    delete $3;
  } | '[' DefList ']' {
    $$ = new ListInit(*$2);
    delete $2;
  } | Value '.' ID {
    if (!$1->getFieldType(*$3)) {
      err() << "Cannot access field '" << *$3 << "' of value '" << *$1 << "!\n";
      abort();
    }
    $$ = new FieldInit($1, *$3);
    delete $3;
  };

DefList : /*empty */ {
    $$ = new std::vector<Record*>();
  } | DefListNE {
    $$ = $1;
  };
DefListNE : DefID {
    $$ = new std::vector<Record*>();
    $$->push_back($1);
  } | DefListNE ',' DefID {
    ($$=$1)->push_back($3);
  };


RBitList : INTVAL {
    $$ = new std::vector<unsigned>();
    $$->push_back($1);
  } | INTVAL '-' INTVAL {
    if ($1 < $3 || $1 < 0 || $3 < 0) {
      err() << "Invalid bit range: " << $1 << "-" << $3 << "!\n";
      abort();
    }
    $$ = new std::vector<unsigned>();
    for (int i = $1; i >= $3; --i)
      $$->push_back(i);
  } | INTVAL INTVAL {
    $2 = -$2;
    if ($1 < $2 || $1 < 0 || $2 < 0) {
      err() << "Invalid bit range: " << $1 << "-" << $2 << "!\n";
      abort();
    }
    $$ = new std::vector<unsigned>();
    for (int i = $1; i >= $2; --i)
      $$->push_back(i);
  } | RBitList ',' INTVAL {
    ($$=$1)->push_back($3);
  } | RBitList ',' INTVAL '-' INTVAL {
    if ($3 < $5 || $3 < 0 || $5 < 0) {
      err() << "Invalid bit range: " << $3 << "-" << $5 << "!\n";
      abort();
    }
    $$ = $1;
    for (int i = $3; i >= $5; --i)
      $$->push_back(i);
  } | RBitList ',' INTVAL INTVAL {
    $4 = -$4;
    if ($3 < $4 || $3 < 0 || $4 < 0) {
      err() << "Invalid bit range: " << $3 << "-" << $4 << "!\n";
      abort();
    }
    $$ = $1;
    for (int i = $3; i >= $4; --i)
      $$->push_back(i);
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
  addValue(RecordVal(*$3, $2, $1));
  setValue(*$3, 0, $4);
  $$ = $3;
};

BodyItem : Declaration ';' {
  delete $1;
} | SET ID OptBitList '=' Value ';' {
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

ObjectBody : ID {
           CurRec = new Record(*$1);
           delete $1;
         } OptTemplateArgList ClassList {
           for (unsigned i = 0, e = $4->size(); i != e; ++i) {
	     addSubClass((*$4)[i].first, *(*$4)[i].second);
	     delete (*$4)[i].second;  // Delete the template list
	   }
           delete $4;

	   // Process any variables on the set stack...
	   for (unsigned i = 0, e = SetStack.size(); i != e; ++i)
	     setValue(SetStack[i].first.first, SetStack[i].first.second,
		      SetStack[i].second);
         } Body {
  CurRec->resolveReferences();
  $$ = CurRec;
  CurRec = 0;
};

ClassInst : CLASS ObjectBody {
  if (Records.getClass($2->getName())) {
    err() << "Class '" << $2->getName() << "' already defined!\n";
    abort();
  }
  Records.addClass($$ = $2);
};

DefInst : DEF ObjectBody {
  // TODO: If ObjectBody has template arguments, it's an error.
  if (Records.getDef($2->getName())) {
    err() << "Def '" << $2->getName() << "' already defined!\n";
    abort();
  }
  Records.addDef($$ = $2);
};


Object : ClassInst | DefInst;

// SETCommand - A 'SET' statement start...
SETCommand : SET ID OptBitList '=' Value IN {
  SetStack.push_back(std::make_pair(std::make_pair(*$2, $3), $5));
  delete $2;
};

// Support Set commands wrapping objects... both with and without braces.
Object : SETCommand '{' ObjectList '}' {
    delete SetStack.back().first.second; // Delete OptBitList
    SetStack.pop_back();
  }
  | SETCommand Object {
    delete SetStack.back().first.second; // Delete OptBitList
    SetStack.pop_back();
  };

ObjectList : Object {} | ObjectList Object {};

File : ObjectList {};

%%

int yyerror(const char *ErrorMsg) {
  err() << "Error parsing: " << ErrorMsg << "\n";
  abort();
}
