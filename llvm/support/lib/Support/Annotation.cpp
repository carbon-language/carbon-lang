//===-- Annotation.cpp - Implement the Annotation Classes --------*- C++ -*--=//
//
// This file implements the AnnotationManager class.
//
//===----------------------------------------------------------------------===//

#include <map>
#include "Support/Annotation.h"
using std::string;
using std::map;
using std::pair;
using std::make_pair;

typedef map<const string, unsigned> IDMapType;
static unsigned IDCounter = 0;  // Unique ID counter

// Static member to ensure initialiation on demand.
static IDMapType &getIDMap() { static IDMapType TheMap; return TheMap; }

// On demand annotation creation support...
typedef Annotation *(*AnnFactory)(AnnotationID, const Annotable *, void *);
typedef map<unsigned, pair<AnnFactory,void*> > FactMapType;

static FactMapType *TheFactMap = 0;
static FactMapType &getFactMap() {
  if (TheFactMap == 0)
    TheFactMap = new FactMapType();
  return *TheFactMap;
}

static void eraseFromFactMap(unsigned ID) {
  assert(TheFactMap && "No entries found!");
  TheFactMap->erase(ID);
  if (TheFactMap->empty()) {   // Delete when empty
    delete TheFactMap;
    TheFactMap = 0;
  }
}


AnnotationID AnnotationManager::getID(const string &Name) {  // Name -> ID
  IDMapType::iterator I = getIDMap().find(Name);
  if (I == getIDMap().end()) {
    getIDMap()[Name] = IDCounter++;   // Add a new element
    return IDCounter-1;
  }
  return I->second;
}

// getID - Name -> ID + registration of a factory function for demand driven
// annotation support.
AnnotationID AnnotationManager::getID(const string &Name, Factory Fact,
				      void *Data) {
  AnnotationID Result(getID(Name));
  registerAnnotationFactory(Result, Fact, Data);
  return Result;		      
}


// getName - This function is especially slow, but that's okay because it should
// only be used for debugging.
//
const string &AnnotationManager::getName(AnnotationID ID) {        // ID -> Name
  IDMapType &TheMap = getIDMap();
  for (IDMapType::iterator I = TheMap.begin(); ; ++I) {
    assert(I != TheMap.end() && "Annotation ID is unknown!");
    if (I->second == ID.ID) return I->first;
  }
}


// registerAnnotationFactory - This method is used to register a callback
// function used to create an annotation on demand if it is needed by the 
// Annotable::findOrCreateAnnotation method.
//
void AnnotationManager::registerAnnotationFactory(AnnotationID ID, 
						  AnnFactory F,
						  void *ExtraData) {
  if (F)
    getFactMap()[ID.ID] = make_pair(F, ExtraData);
  else
    eraseFromFactMap(ID.ID);
}

// createAnnotation - Create an annotation of the specified ID for the
// specified object, using a register annotation creation function.
//
Annotation *AnnotationManager::createAnnotation(AnnotationID ID, 
						const Annotable *Obj) {
  FactMapType::iterator I = getFactMap().find(ID.ID);
  if (I == getFactMap().end()) return 0;
  return I->second.first(ID, Obj, I->second.second);
}
