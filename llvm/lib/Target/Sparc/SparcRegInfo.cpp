#include "llvm/CodeGen/IGNode.h"
#include "SparcRegInfo.h"

#include "llvm/Target/Sparc.h"

//-----------------------------------------------------------------------------
// Int Register Class
//-----------------------------------------------------------------------------

void SparcIntRegClass::colorIGNode(IGNode * Node, bool IsColorUsedArr[]) const 
{

  /* Algorithm:
  Record the color of all neighbors.

  If there is no call interf, try to allocate volatile, then non volatile
  If there is call interf, try to allocate non-volatile. If that fails
     try to allocate a volatile and insert save across calls
  If both above fail, spill.

  */

  unsigned NumNeighbors =  Node->getNumOfNeighbors();   // total # of neighbors

  for(unsigned n=0; n < NumNeighbors; n++) {            // for each neigh 
    IGNode *NeighIGNode = Node->getAdjIGNode(n);
    if( NeighIGNode->hasColor() ) {                     // if neigh has a color
      IsColorUsedArr[ NeighIGNode->getColor() ] = true; // record that color
    }
  }



  unsigned SearchStart;                 // start pos of color in pref-order
  bool ColorFound= false;               // have we found a color yet?

  //if this Node is between calls
  if( Node->getNumOfCallInterferences() == 0) { 

    // start with volatiles (we can  allocate volatiles safely)
    SearchStart = SparcIntRegOrder::StartOfAllRegs;  
  }
  else {           
    // start with non volatiles (no non-volatiles)
    SearchStart =  SparcIntRegOrder::StartOfNonVolatileRegs;  
  }

  unsigned c=0;                         // color
 
  // find first unused color
  for( c=SearchStart; c < SparcIntRegOrder::NumOfAvailRegs; c++) { 
    if( ! IsColorUsedArr[ c ] ) { ColorFound = true; break; }
  }

  if( ColorFound) 
    Node->setColor(c);                  // first color found in preffered order

  // if color is not found because of call interference
  // try even finding a volatile color and insert save across calls
  else if( Node->getNumOfCallInterferences() ) 
  { 
    // start from 0 - try to find even a volatile this time
    SearchStart = SparcIntRegOrder::StartOfAllRegs;  

    // find first unused volatile color
    for(c=SearchStart; c < SparcIntRegOrder::StartOfNonVolatileRegs; c++) { 
      if( ! IsColorUsedArr[ c ] ) { ColorFound = true; break; }
    }

    if( ColorFound) { 
      Node->setColor(c);  
      // since LR span across calls, must save across calls 
      Node->markForSaveAcrossCalls();       
    }

  }

  // If we couldn't find a color regardless of call interference - i.e., we
  // don't have either a volatile or non-volatile color left
  if( !ColorFound )  
    Node->markForSpill();               // no color found - must spill


  if( DEBUG_RA)                  
    UltraSparcRegInfo::printReg( Node->getParentLR() );

}






//-----------------------------------------------------------------------------
// Float Register Class
//-----------------------------------------------------------------------------

// find the first available color in the range [Start,End] depending on the
// type of the Node (i.e., float/double)

int SparcFloatRegClass::findFloatColor(const IGNode *const Node, unsigned Start,
 				       unsigned End, 
				       bool IsColorUsedArr[] ) const
{

  bool ColorFound = false;
  unsigned c;

  if( Node->getTypeID() == Type::DoubleTyID ) { 
      
    // find first unused color for a double 
    for( c=Start; c < End ;c+= 2){
      if( ! IsColorUsedArr[ c ] &&  ! IsColorUsedArr[ c+1 ]) 
	{ ColorFound=true;  break; }
    }
    
  } else {
    
    // find first unused color for a single
    for( c=Start; c < End; c++) { 
      if( ! IsColorUsedArr[ c ] ) { ColorFound=true;  break; }
    }
  }
  
  if( ColorFound ) return c;
  else return -1;
}





void SparcFloatRegClass::colorIGNode(IGNode * Node,bool IsColorUsedArr[]) const
{

  /* Algorithm:

     If the LR is a double try to allocate f32 - f63
     If the above fails or LR is single precision
        If the LR does not interfere with a call
	   start allocating from f0
	Else start allocating from f6
     If a color is still not found because LR interferes with a call
        Search in f0 - f6. If found mark for spill across calls.
     If a color is still not fond, mark for spilling
  */


  unsigned NumNeighbors =  Node->getNumOfNeighbors();   // total # of neighbors

  for(unsigned n=0; n < NumNeighbors; n++) {            // for each neigh 
    IGNode *NeighIGNode = Node->getAdjIGNode(n);
    if( NeighIGNode->hasColor() ) {                     // if neigh has a color
      IsColorUsedArr[ NeighIGNode->getColor() ] = true; // record that color
      if( NeighIGNode->getTypeID() == Type::DoubleTyID )
	IsColorUsedArr[ (NeighIGNode->getColor()) + 1 ] = true;  
    }
  }

  int ColorFound = -1;               // have we found a color yet?
  unsigned NumOfCallInterf = Node->getNumOfCallInterferences();

  // if value is a double - search the double only reigon (f32 - f63)
  if( Node->getTypeID() == Type::DoubleTyID )       
    ColorFound = findFloatColor( Node, 32, 64, IsColorUsedArr );
    

  if( ColorFound >= 0 ) {
    Node->setColor(ColorFound);                
    if( DEBUG_RA) UltraSparcRegInfo::printReg( Node->getParentLR() );
    return;
  }

  else { // the above fails or LR is single precision

    unsigned SearchStart;                 // start pos of color in pref-order

    //if this Node is between calls (i.e., no call interferences )
    if( ! NumOfCallInterf ) {
      // start with volatiles (we can  allocate volatiles safely)
      SearchStart = SparcFloatRegOrder::StartOfAllRegs;  
    }
    else {           
      // start with non volatiles (no non-volatiles)
      SearchStart =  SparcFloatRegOrder::StartOfNonVolatileRegs;  
    }
    
    ColorFound = findFloatColor( Node, SearchStart, 32, IsColorUsedArr );

  }

  if( ColorFound >= 0 ) {
    Node->setColor(ColorFound);                  
    if( DEBUG_RA) UltraSparcRegInfo::printReg( Node->getParentLR() );
    return;
  }

  else if( NumOfCallInterf ) { 

    // We are here because there is a call interference and no non-volatile
    // color could be found.
    // Now try to allocate even a volatile color

    ColorFound = findFloatColor( Node, SparcFloatRegOrder::StartOfAllRegs, 
				SparcFloatRegOrder::StartOfNonVolatileRegs,
				IsColorUsedArr);
  }

  if( ColorFound >= 0 ) {
    Node->setColor(ColorFound);         // first color found in preffered order
    Node->markForSaveAcrossCalls();  
    if( DEBUG_RA) UltraSparcRegInfo::printReg( Node->getParentLR() );
    return;
  }

  else {
    Node->markForSpill();               // no color found - must spill
    if( DEBUG_RA) UltraSparcRegInfo::printReg( Node->getParentLR() );
  }
  

}




















#if 0

//-----------------------------------------------------------------------------
// Float Register Class
//-----------------------------------------------------------------------------

void SparcFloatRegClass::colorIGNode(IGNode * Node,bool IsColorUsedArr[]) const
{

  /* Algorithm:
  Record the color of all neighbors.

  Single precision can use f0 - f31
  Double precision can use f0 - f63

  if LR is a double, try to allocate f32 - f63.
  if the above attempt fails, or Value is single presion, try to allcoate 
    f0 - f31.

      */

  unsigned NumNeighbors =  Node->getNumOfNeighbors();   // total # of neighbors

  for(unsigned n=0; n < NumNeighbors; n++) {            // for each neigh 
    IGNode *NeighIGNode = Node->getAdjIGNode(n);
    if( NeighIGNode->hasColor() ) {                     // if neigh has a color
      IsColorUsedArr[ NeighIGNode->getColor() ] = true; // record that color
      if( NeighIGNode->getTypeID() == Type::DoubleTyID )
	IsColorUsedArr[ (NeighIGNode->getColor()) + 1 ] = true;  
    }
  }


  unsigned SearchStart;                 // start pos of color in pref-order
  bool ColorFound= false;               // have we found a color yet?
  unsigned c;    


  if( Node->getTypeID() == Type::DoubleTyID ) {        // if value is a double

    // search the double only reigon (f32 - f63)
     for( c=32; c < 64; c+= 2) { 
      if( ! IsColorUsedArr[ c ] ) { ColorFound = true; break; }
    }

     // search f0 - f31 region
    if( ! ColorFound )  {                // if color not found
     for( c=0; c < 32; c+= 2) { 
      if( ! IsColorUsedArr[ c ] ) { ColorFound = true; break; }
     }
    }

  }

  else {  // value is Single

    for( c=0; c < 32; c++) { 
      if( ! IsColorUsedArr[ c ] ) { ColorFound = true; break; }
    }
  }
  

  if( ColorFound) 
    Node->setColor(c);                  // first color found in preferred order
  else
    Node->markForSpill();               // no color found - must spill


  if( DEBUG_RA)                  
    UltraSparcRegInfo::printReg( Node->getParentLR() );

}

#endif
