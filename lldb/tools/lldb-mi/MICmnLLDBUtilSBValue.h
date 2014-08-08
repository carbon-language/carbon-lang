//===-- MICmnLLDBUtilSBValue.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmnLLDBUtilSBValue.h
//
// Overview:	CMICmnLLDBUtilSBValue interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Third Party Headers:
#include <lldb/API/SBValue.h>

// In-house headers:
#include "MIDataTypes.h"

// Declerations:
class CMIUtilString;

//++ ============================================================================
// Details:	Utility helper class to lldb::SBValue. Using a lldb::SBValue extract
//			value object information to help form verbose debug information.
// Gotchas:	None.
// Authors:	Illya Rudkin 08/07/2014.
// Changes:	None.
//--
class CMICmnLLDBUtilSBValue
{
// Methods:
public:
	/* ctor */	CMICmnLLDBUtilSBValue( const lldb::SBValue & vrValue, const bool vbHandleCharType = false );
	/* dtor */	~CMICmnLLDBUtilSBValue( void );
	//
	CMIUtilString	GetName( void ) const;
	CMIUtilString	GetValue( void ) const;
	CMIUtilString	GetValueCString( void ) const;
	CMIUtilString	GetChildValueCString( void ) const;
	CMIUtilString	GetTypeName( void ) const;
	CMIUtilString	GetTypeNameDisplay( void ) const;
	bool			IsCharType( void ) const;
	bool			IsChildCharType( void ) const;
	bool			IsLLDBVariable( void ) const;
	bool			IsNameUnknown( void ) const;
	bool			IsValueUnknown( void ) const;
	bool			IsValid( void ) const;
	bool			HasName( void ) const;

// Methods:
private:
	CMIUtilString	ReadCStringFromHostMemory( const lldb::SBValue & vrValueObj ) const;

// Attributes:
private:
	lldb::SBValue & m_rValue;
	const MIchar *	m_pUnkwn;
	bool			m_bValidSBValue;	// True = SBValue is a valid object, false = not valid.
	bool			m_bHandleCharType;	// True = Yes return text molding to char type, false = just return data.
};
