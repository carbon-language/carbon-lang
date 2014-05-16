//===-- MIUtilSetID.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilSetID.h
//
// Overview:	CMIUtilSetID interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Third party headers:
#include <set>

// In-house headers:
#include "MIUtilString.h"  
#include "MICmnBase.h"

//++ ============================================================================
// Details:	MI common code utility class. Set type container used to handle 
//			unique ID registration.
// Gotchas:	None.
// Authors:	Illya Rudkin 17/02/2014.
// Changes:	None.
//--
class CMIUtilSetID 
:	public std::set< CMIUtilString >
,	public CMICmnBase
{
// Methods:
public:
	/* ctor */	CMIUtilSetID( void );
	
	bool	Register( const CMIUtilString & vId );
	bool	Unregister( const CMIUtilString & vId );
	bool	HaveAlready( const CMIUtilString & vId ) const;
	bool	IsValid( const CMIUtilString & vId ) const;

// Overrideable:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMIUtilSetID( void );
};
