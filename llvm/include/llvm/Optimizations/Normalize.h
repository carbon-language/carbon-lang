// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	Normalize.h
// 
// Purpose:
//	Transformations to normalize LLVM code to simplify later passes:
//	-- Insert loads of constants that are arguments to PHI
//	   in the appropriate predecessor basic block.
//	
// History:
//	8/25/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_OPT_NORMALIZE_H
#define LLVM_OPT_NORMALIZE_H

//************************** System Include Files ***************************/


//*************************** User Include Files ***************************/


//************************* Forward Declarations ***************************/

class Method;

//************************** External Functions ****************************/


void	NormalizePhiConstantArgs	(Method* method);


//**************************************************************************/

#endif LLVM_OPT_NORMALIZE_H
