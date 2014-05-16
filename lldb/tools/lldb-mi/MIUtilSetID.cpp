//===-- MIUtilSetID.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilSetID.cpp
//
// Overview:	CMIUtilSetID interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MIUtilSetID.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilSetID constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilSetID::CMIUtilSetID( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilSetID destructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilSetID::~CMIUtilSetID( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Register an ID.
// Type:	Method.
// Args:	vId	- (R) Unique ID i.e. GUID.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIUtilSetID::Register( const CMIUtilString & vId )
{
	if( !IsValid( vId ) )
	{
		SetErrorDescription( CMIUtilString::Format( "ID '%s' invalid", vId.c_str() ) );
		return MIstatus::failure;
	}

	if( HaveAlready( vId ) )
		return MIstatus::success;

	insert( vId );

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Unregister an ID.
// Type:	Method.
// Args:	vId	- (R) Unique ID i.e. GUID.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMIUtilSetID::Unregister( const CMIUtilString & vId )
{
	if( !IsValid( vId ) )
	{
		SetErrorDescription( CMIUtilString::Format( "ID '%s' invalid", vId.c_str() ) );
		return MIstatus::failure;
	}

	erase( vId );

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Check an ID is already registered.
// Type:	Method.
// Args:	vId	- (R) Unique ID i.e. GUID.
// Return:	True - registered.
//			False - not found.
// Throws:	None.
//--
bool CMIUtilSetID::HaveAlready( const CMIUtilString & vId ) const
{
	const_iterator it = find( vId );
	if( it != end() )
		return true;
	
	return false;
}

//++ ------------------------------------------------------------------------------------
// Details:	Check the ID is valid to be registered.
// Type:	Method.
// Args:	vId	- (R) Unique ID i.e. GUID.
// Return:	True - valid.
//			False - not valid.
// Throws:	None.
//--
bool CMIUtilSetID::IsValid( const CMIUtilString & vId ) const
{
	bool bValid = true;

	if( vId.empty() )
		bValid = false;
	
	return bValid;
}

