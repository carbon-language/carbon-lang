//===-- Annotation.cpp - Implement the Annotation Classes --------*- C++ -*--=//
//
// This file implements the AnnotationManager class.
//
//===----------------------------------------------------------------------===//

#include <map>
#include "llvm/Annotation.h"

typedef map<const string, unsigned> IDMapType;
static unsigned IDCounter = 0;  // Unique ID counter

// Static member to ensure initialiation on demand.
static IDMapType &getIDMap() { static IDMapType TheMap; return TheMap; }

// On demand annotation creation support...
typedef Annotation *(*AnnFactory)(AnnotationID, Annotable *);
typedef map<unsigned, AnnFactory> FactMapType;
static FactMapType &getFactMap() { static FactMapType FactMap; return FactMap; }


AnnotationID AnnotationManager::getID(const string &Name) {  // Name -> ID
  IDMapType::iterator I = getIDMap().find(Name);
  if (I == getIDMap().end()) {
    getIDMap()[Name] = IDCounter++;   // Add a new element
    return IDCounter-1;
  }
  return I->second;
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
						  AnnFactory F) {
  if (F)
    getFactMap()[ID.ID] = F;
  else
    getFactMap().erase(ID.ID);
}

// createAnnotation - Create an annotation of the specified ID for the
// specified object, using a register annotation creation function.
//
Annotation *AnnotationManager::createAnnotation(AnnotationID ID, 
						Annotable *Obj) {
  FactMapType::iterator I = getFactMap().find(ID.ID);
  if (I == getFactMap().end()) return 0;
  return I->second(ID, Obj);
}
